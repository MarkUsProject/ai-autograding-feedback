import base64
import importlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def encode_image(image_path: os.PathLike) -> bytes:
    """Encodes the image found at {image_path} to a base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_images(input_notebook_path: os.PathLike, output_directory: os.PathLike, output_name: str) -> List[Path]:
    image_paths = []
    with open(input_notebook_path, "r") as file:
        notebook = json.load(file)
        os.makedirs(output_directory, exist_ok=True)
        for cell_number, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                # Choosing the saved image's name
                source = cell["source"]
                if len(source) > 0 and source[0].startswith("# Question"):
                    # Cell header for file names
                    question_name = source[0][2:-1]
                elif "markus_question_name" in cell["metadata"]:
                    # Cell metadata tag for saved image file name
                    question_name = cell["metadata"]["markus_question_name"]
                else:
                    continue

                # Find images and save them
                image_count = 0
                for output in cell["outputs"]:
                    for file_type, data in output["data"].items():
                        if "image/" in file_type:
                            ext = file_type.split("/")[-1]
                            image_filename = f"{output_name}.{ext}"
                            os.makedirs(
                                os.path.join(output_directory, question_name, str(image_count)),
                                exist_ok=True,
                            )
                            image_path = os.path.join(
                                output_directory,
                                question_name,
                                str(image_count),
                                image_filename,
                            )
                            image_count += 1

                            image_data = base64.b64decode(data)
                            with open(image_path, "wb") as img_file:
                                img_file.write(image_data)
                            image_paths.append(os.path.abspath(image_path))

                # Save question context (source of previous cell)
                if cell_number >= 1 and notebook["cells"][cell_number - 1]["cell_type"] == "markdown":
                    question_context_data = "".join(notebook["cells"][cell_number - 1]["source"])
                    question_context_filename = "context.txt"
                    os.makedirs(os.path.join(output_directory, question_name), exist_ok=True)
                    question_context_path = os.path.join(output_directory, question_name, question_context_filename)
                    with open(question_context_path, "w") as txt_file:
                        txt_file.write(question_context_data)
    return image_paths


def extract_qmd_python_chunks_with_context(qmd_path: str) -> List[Dict[str, Any]]:
    """
    Extract ONLY Python code chunks from a QMD and annotate each with context from # / ## headings.
    Supports ```{python ...}, ```python, and ~~~ variants. Skips YAML front matter.
    """
    _PY_CHUNK_START = re.compile(
        r"""^(
                ```\{python\b[^}]*\}\s*$ |   # ```{python ...}
                ```python\s*$            |   # ```python
                ~~~\{python\b[^}]*\}\s*$ |   # ~~~{python ...}
                ~~~python\s*$                # ~~~python
            )""",
        re.IGNORECASE | re.VERBOSE,
    )
    _FENCE_END_TICKS = re.compile(r"^```\s*$")
    _FENCE_END_TILDES = re.compile(r"^~~~\s*$")
    _H1 = re.compile(r"^#\s+(.*)$")
    _H2 = re.compile(r"^##\s+(.*)$")

    qp = Path(qmd_path)
    if not qp.exists():
        raise FileNotFoundError(f"QMD file not found: {qmd_path}")

    lines = qp.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Skip YAML front matter if present
    i = 0
    if lines and re.match(r"^---\s*$", lines[0]):
        i = 1
        while i < len(lines) and not re.match(r"^---\s*$", lines[i]):
            i += 1
        i = min(i + 1, len(lines))

    current_main = None
    current_sub = None
    chunks = []
    in_py = False
    cur = []
    start_line = 0
    fence_kind = None  # "```" or "~~~"

    for i, raw in enumerate(lines):
        line = raw.rstrip("\n")

        if not in_py and _PY_CHUNK_START.match(line):
            in_py = True
            cur = []
            start_line = i
            fence_kind = "~~~" if line.strip().startswith("~~~") else "```"
            continue

        if in_py:
            if (fence_kind == "```" and _FENCE_END_TICKS.match(line)) or (
                fence_kind == "~~~" and _FENCE_END_TILDES.match(line)
            ):
                in_py = False
                fence_kind = None
                context = (
                    f"{current_main}__{current_sub}" if current_main and current_sub else (current_main or "unknown")
                )
                if cur:
                    chunks.append(
                        {
                            "context": context,
                            "code": cur[:],
                            "start_line": start_line + 1,  # 1-based
                        }
                    )
                continue
            else:
                cur.append(raw)
                continue

        m1 = _H1.match(line)
        if m1:
            current_main = _clean_heading_text(m1.group(1))
            current_sub = None
            continue
        m2 = _H2.match(line)
        if m2:
            current_sub = _clean_heading_text(m2.group(1))
            continue

    return chunks


def extract_qmd_python_images(qmd_path: str, output_dir: Optional[str] = None, dpi: int = 120) -> List[str]:
    """
    Runs the python blocks of code and saves images using matplotlib.
    Args:
        qmd_path: path to qmd file to extract images from
        output_dir(Optional): directory to save images to
        dpi: The resolution of the output image

    Returns:
        List[str]: Returns a list of image paths

    """
    chunks = extract_qmd_python_chunks_with_context(qmd_path)
    if not chunks:
        return []

    outdir = Path(output_dir or tempfile.mkdtemp(prefix="qmd_py_imgs_"))
    outdir.mkdir(parents=True, exist_ok=True)

    mpl = importlib.import_module("matplotlib")
    mpl.use("Agg", force=True)
    plt = importlib.import_module("matplotlib.pyplot")

    saved_files = []
    per_context_counter = {}
    current_context = "unknown"

    def _save_fig_with_context(fig, ctx: str):
        try:
            fig.canvas.draw()  # force render
        except Exception as e:
            print(f"[warn] could not draw canvas: {e}")

        cnt = per_context_counter.get(ctx, 0) + 1
        per_context_counter[ctx] = cnt
        fname = f"plot__{ctx}__{cnt:03d}.png"
        path = outdir / fname

        fig.savefig(path.as_posix(), dpi=dpi, bbox_inches="tight")
        saved_files.append(path.as_posix())

    # mirror user savefig
    _orig_savefig = plt.savefig

    def _mirror_savefig(*args, **kwargs):
        nonlocal current_context, saved_fignums_this_chunk
        ctx = current_context
        _orig_savefig(*args, **kwargs)
        fig = plt.gcf()
        fnum = fig.number
        if fnum not in saved_fignums_this_chunk:
            _save_fig_with_context(fig, ctx)
            saved_fignums_this_chunk.add(fnum)

    plt.savefig = _mirror_savefig

    exec_env = {"__name__": "__qmd_exec__", "plt": plt}

    for ch in chunks:
        current_context = ch["context"].replace(os.sep, "_")
        per_context_counter.setdefault(current_context, 0)

        fignums_before = set(plt.get_fignums())

        created_figs = []
        saved_fignums_this_chunk = set()

        _orig_figure = plt.figure
        _orig_subplots = plt.subplots

        def _wrapped_figure(*args, **kwargs):
            fig = _orig_figure(*args, **kwargs)
            created_figs.append((fig.number, fig))
            return fig

        def _wrapped_subplots(*args, **kwargs):
            fig, ax = _orig_subplots(*args, **kwargs)
            created_figs.append((fig.number, fig))
            return fig, ax

        plt.figure = _wrapped_figure
        plt.subplots = _wrapped_subplots

        code = "\n".join(ch["code"])
        try:
            exec(compile(code, filename=f"{qmd_path}:{ch['start_line']}", mode="exec"), exec_env, exec_env)
        except Exception:
            pass
        finally:
            plt.figure = _orig_figure
            plt.subplots = _orig_subplots

        fignums_after = set(plt.get_fignums())
        new_fignums = sorted(fignums_after - fignums_before)

        for fnum, fig in created_figs:
            if fnum in saved_fignums_this_chunk:
                continue
            _save_fig_with_context(fig, current_context)
            saved_fignums_this_chunk.add(fnum)

        # save any remaining new fig numbers (re-open only if not saved)
        for fnum in new_fignums:
            if fnum in saved_fignums_this_chunk:
                continue
            fig = plt.figure(fnum)
            _save_fig_with_context(fig, current_context)
            saved_fignums_this_chunk.add(fnum)

        # close figs we touched
        for _, fig in created_figs:
            try:
                plt.close(fig)
            except Exception:
                pass
        for fnum in new_fignums:
            try:
                plt.close(plt.figure(fnum))
            except Exception:
                pass

    plt.savefig = _orig_savefig

    seen, uniq = set(), []
    for p in saved_files:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _clean_heading_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s
