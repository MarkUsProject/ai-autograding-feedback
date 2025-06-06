import os
import sys
from pathlib import Path
from typing import Callable, Tuple

from .helpers.arg_options import model_mapping
from .helpers.file_converter import rename_files
from .helpers.template_utils import render_prompt_template

EXPECTED_SUFFIXES = ["_solution", "test_output", "_submission"]


def process_code(args, prompt: str) -> Tuple[str, str]:
    """
    Processes assignment files and generates a response using the selected model.

    Depending on the submission type (Jupyter or Python), it loads the appropriate
    files from the assignment folder, modifies the prompt with file references,
    and invokes the selected model to generate a response.

    Args:
        args: Command-line argument namespace containing submission_type, assignment, model, scope, and question.
        prompt (str): The initial user prompt to be modified and passed to the model.

    Returns:
        Tuple[str, str]: A tuple containing the final request string and the model's generated response.

    Raises:
        FileNotFoundError: If the submission, solution, or test_output file does not exist.
        SystemExit: If an invalid model is selected.
    """
    if not os.path.isfile(args.submission):
        raise FileNotFoundError(f"Submission file '{args.submission}' not found.")
    submission_file = Path(args.submission)

    solution_file = None
    if args.solution:
        if not os.path.isfile(args.solution):
            raise FileNotFoundError(f"Solution file '{args.solution}' not found.")
        solution_file = Path(args.solution)

    test_output_file = None

    if args.test_output:
        if not os.path.isfile(args.test_output):
            raise FileNotFoundError(f"Test output file '{args.test_output}' not found.")
        test_output_file = Path(args.test_output)

    if args.submission_type == "jupyter":
        # convert submission notebooks to .txt if needed
        ensure_txt_file(args.submission, rename_files)
        submission_file = Path(args.submission.replace(".ipynb", ".txt"))
        if solution_file:
            # convert solution notebooks to .txt if needed
            ensure_txt_file(args.solution, rename_files)
            solution_file = Path(args.solution.replace(".ipynb", ".txt"))

    elif args.submission_type != "python":
        raise SystemExit(f"Invalid submission type '{args.submission_type}'.")

    assignment_files = [f for f in [args.submission, args.solution, args.test_output] if f]

    prompt = render_prompt_template(prompt, assignment_files=assignment_files)

    if args.model in model_mapping:
        model = model_mapping[args.model]()
    else:
        print("Invalid model selected for code scope.")
        sys.exit(1)

    if args.scope == "code":
        if args.question:
            request, response = model.generate_response(
                prompt=prompt,
                submission_file=submission_file,
                solution_file=solution_file,
                test_output=test_output_file,
                question_num=args.question,
                llama_mode=args.llama_mode,
            )
        else:
            request, response = model.generate_response(
                prompt=prompt,
                submission_file=submission_file,
                solution_file=solution_file,
                test_output=test_output_file,
                llama_mode=args.llama_mode
            )

    return request, response


def ensure_txt_file(file_path: str, rename_function: Callable[[Path], None]) -> None:
    """Ensures a .txt version of the given file exists if needed (e.g., for Jupyter .ipynb files).

    Args:
        file_path (str): Path to the assignment directory.
        rename_function (Callable): A function that handles renaming or converting files.

    Returns:
        None
    """
    txt_file_path = file_path.replace(".ipynb", ".txt")
    if not os.path.exists(txt_file_path) and file_path.endswith(".ipynb"):
        rename_function(Path(file_path))

