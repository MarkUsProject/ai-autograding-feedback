# tests/test_integration.py
import shutil
import sys
from pathlib import Path
from ai_feedback.__main__ import main
from datetime import datetime
import pytest
from unittest.mock import patch
from subprocess import CompletedProcess

all_prompts = []

@pytest.fixture(scope="session", autouse=True)
def collect_prompts_and_write_individual_files():
    all_prompts.clear()
    yield

    report_dir = Path("prompt_reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    for entry in report_dir.iterdir():
        if entry.is_dir() and entry.suffix == ".txt":
            shutil.rmtree(entry)

    master_index = report_dir / "index.txt"
    with open(master_index, "w", encoding="utf-8") as idx_file:
        idx_file.write(f"Prompt Log Index @ {datetime.now().isoformat()} UTC\n\n")

    for count, (test_name, mock_loc, prompt_text) in enumerate(all_prompts, start=1):
        safe_test = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in test_name)
        safe_loc = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in mock_loc)

        filename = f"{count:03d}_{safe_test}_{safe_loc}.txt"
        file_path = report_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            # (Optional) include a header in each file:
            f.write(f"Test: {test_name}\n")
            f.write(f"Location: {mock_loc}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()} UTC\n")
            f.write("\n")
            f.write(prompt_text)
            f.write("\n")

        with open(master_index, "a", encoding="utf-8") as idx_file:
            idx_file.write(f"{filename}  ←  [{test_name}] at {mock_loc}\n")

    print(f"\nWritten {len(all_prompts)} prompt files into: {report_dir.resolve()}\n")


@pytest.fixture(autouse=True)
def mock_and_capture(request):
    test_name = request.node.name

    def fake_base(self, prompt, *args, **kwargs):
        all_prompts.append((test_name, "Model.generate_response", prompt))
        return prompt, prompt

    patch("ai_feedback.models.Model.Model.generate_response",
          side_effect=fake_base).start()

    def fake_openai_call(prompt):
        all_prompts.append((test_name, "OpenAIModel._call_openai", prompt))
        return prompt

    patch("ai_feedback.models.OpenAIModel._call_openai",
          side_effect=fake_openai_call).start()

    class DummyMsg:
        def __init__(self, text):
            self.content = text

    class DummyReply:
        def __init__(self, text):
            self.message = DummyMsg(text)

    def fake_ip_chat(model, messages, options=None):
        text = messages[-1].content
        all_prompts.append((test_name, "image_processing.chat", text))
        return DummyReply(text)

    patch("ai_feedback.image_processing.chat",
          side_effect=fake_ip_chat).start()

    def fake_ds_chat(model, messages, **kwargs):
        text = messages[-1]["content"]
        all_prompts.append((test_name, "ollama.chat", text))
        return {"message": {"content": text}}

    patch("ollama.chat", side_effect=fake_ds_chat).start()

    yield
    patch.stopall()


def run_cli_and_capture(argv_list, capsys):
    orig_argv = sys.argv
    sys.argv = ["ai_feedback"] + argv_list
    try:
        main()
    finally:
        sys.argv = orig_argv
    captured = capsys.readouterr()
    return captured.out


def run_cli_and_capture(argv_list, capsys):
    """
    Helper to run the CLI with a custom sys.argv and capture stdout/stderr.
    Returns the captured stdout as a string.
    """
    orig_argv = sys.argv
    sys.argv = ["ai_feedback"] + argv_list

    try:
        main()
    finally:
        sys.argv = orig_argv

    captured = capsys.readouterr()
    return captured.out


def test_cnn_example_openai_stdout(capsys):
    """
    Example 1:
    Evaluate cnn_example test using openAI model and print to stdout.
    python -m ai_feedback --prompt code_lines --scope code \
        --submission test_submissions/cnn_example/cnn_submission \
        --solution test_submissions/cnn_example/cnn_solution.py \
        --model openai --output stdout
    """
    args = [
        "--prompt", "code_lines",
        "--scope", "code",
        "--submission", "../test_submissions/cnn_example/cnn_submission.py",
        "--solution",   "../test_submissions/cnn_example/cnn_solution.py",
        "--model", "openai",
    ]
    output = run_cli_and_capture(args, capsys)

    assert "You are a helpful assistant" in output
    assert "Prompt: Compare the student's code and solution code. For each mistake" in output
    assert "(Line 1) import numpy as np" in output
    assert "=== cnn_submission.py ===" in output
    assert "=== cnn_solution.py ===" in output


def test_cnn_example_custom_prompt_stdout(capsys):
    """
    Example 2:
    Evaluate cnn_example test using openAI model and a custom prompt text, printing to stdout.
    python -m ai_feedback --prompt_text "Evaluate the student's code readability." \
        --scope code \
        --submission test_submissions/cnn_example/cnn_submission.py \
        --model openai \
        --output stdout
    """
    args = [
        "--prompt_text", "Evaluate the student's code readability.",
        "--scope", "code",
        "--submission", "../test_submissions/cnn_example/cnn_submission.py",
        "--model", "openai",
        "--output", "stdout",
    ]
    output = run_cli_and_capture(args, capsys)
    assert "You are a helpful assistant that provides detailed feedback on students" in output
    assert "Prompt: Evaluate the student's code readability." in output
    assert "=== cnn_submission.py ===" in output
    assert "(Line 1) import numpy as np" in output

def test_pdf_example_openai_direct(capsys):
    """
    Example 3:
    Evaluate pdf_example test using openAI model and direct output mode.
    python -m ai_feedback --prompt text_pdf_analyze --scope text \
        --submission test_submissions/pdf_example/student_pdf_submission.pdf \
        --model openai --output direct
    """
    args = [
        "--prompt", "text_pdf_analyze",
        "--scope", "text",
        "--submission", "../test_submissions/pdf_example/student_pdf_submission.pdf",
        "--model", "openai",
        "--output", "direct",
    ]

    output = run_cli_and_capture(args, capsys)
    assert "You are a helpful assistant that provides detailed feedback on students" in output
    assert "Prompt: Does the student correctly respond to the question, and meet all the" in output
    assert "student_pdf_submission.pdf" in output
    assert "Normalization allows each feature to have an equal influence on the mode" in output


def test_ggr274_question1_deepseek_markdown(capsys):
    """
    Example 4:
    Evaluate question 1 of test1 of ggr274 homework using DeepSeek model, output as Markdown.
    python -m ai_feedback --prompt code_table --scope code \
        --submission test_submissions/ggr274_homework5/test1/student_submission.ipynb \
        --question 1 --model deepSeek-R1:70B --output markdown
    """
    args = [
        "--prompt", "code_table",
        "--scope", "code",
        "--submission", "../test_submissions/ggr274_homework5/test1/student_submission.ipynb",
        "--question", "1",
        "--model", "deepSeek-R1:70B",
        "--output", "markdown",
    ]
    output = run_cli_and_capture(args, capsys)
    assert "Markdown report saved" in output


def test_ggr274_image_question5b_llama_stdout(capsys):
    """
    Example 5:
    Evaluate the image for question 5b of ggr274 homework with Llama3.2-vision, printing to stdout.
    python3 -m ai_feedback --prompt image_analyze --scope image \
      --solution ./test_submissions/ggr274_homework5/image_test2/student_submission.ipynb \
      --submission_image test_submissions/ggr274_homework5/image_test2/student_submission.png \
      --question "Question 5b" --model llama3.2-vision --output stdout
    """
    args = [
        "--prompt", "image_analyze",
        "--scope", "image",
        "--submission", "../test_submissions/ggr274_homework5/image_test2/student_submission.ipynb",
        "--submission_image", "../test_submissions/ggr274_homework5/image_test2/student_submission.png",
        "--question", "Question 5b",
        "--model", "llama3.2-vision:90b",
        "--output", "stdout",
    ]

    output = run_cli_and_capture(args, capsys)
    assert "Create the same boxplots as in Task 4 f), but use `Age group label`" in output

def test_deepseek_v3_cli_code_scope_stdout(capsys):
    """
    Evaluate a simple code-scope test using DeepSeekV3Model in CLI mode,
    printing to stdout. We mock subprocess.run so that llama-cli never actually runs.

    python3 -m ai_feedback --prompt code_table --scope code \
        --submission test_submissions/ggr274_homework5/test1/student_submission.ipynb \
        --solution test_submissions/ggr274_homework5/test1/Homework_5_solution.ipynb \
        --model deepSeek-v3
    """
    # Prepare fake CompletedProcess: stdout begins with the quoted prompt,
    # then a newline, then some dummy AI content followed by the end marker.
    fake_prompt = "'code_table'"
    fake_ai_output = "\nFAKE AI RESPONSE LINE 1\nFAKE AI RESPONSE LINE 2 [end of text]\n"
    fake_stdout = (fake_prompt + fake_ai_output).encode("utf-8")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = CompletedProcess(
            args=["./llama-cli"], returncode=0, stdout=fake_stdout, stderr=b""
        )

        args = [
            "--prompt", "code_table",
            "--scope", "code",
            "--submission", "../test_submissions/ggr274_homework5/test1/student_submission.ipynb",
            "--solution",   "../test_submissions/ggr274_homework5/test1/Homework_5_solution.ipynb",
            "--model", "deepSeek-v3",
        ]
        output = run_cli_and_capture(args, capsys)

    assert "FAKE AI RESPONSE LINE 1" in output
    assert "FAKE AI RESPONSE LINE 2" in output

def test_deepseek_v3_server_mode_code_scope_stdout(capsys, monkeypatch):
    """
    Evaluate a code‐scope test using DeepSeekV3Model in SERVER mode (llama-server),
    printing to stdout. We mock requests.post so that no real HTTP request is made.

    python3 -m ai_feedback --prompt code_table --scope code \
        --submission test_submissions/ggr274_homework5/test1/student_submission.ipynb \
        --solution test_submissions/ggr274_homework5/test1/Homework_5_solution.ipynb \
        --model deepSeek-v3 --llama_mode server
    """
    # Ensure LLAMA_SERVER_URL is set so generate_response picks the server path:
    monkeypatch.setenv("LLAMA_SERVER_URL", "localhost:8080")

    fake_prompt = "code_table"
    quoted = f"'{fake_prompt}'"
    fake_ai_body = "\nFAKE SERVER RESPONSE LINE 1\nFAKE SERVER RESPONSE LINE 2 [end of text]\n"
    fake_json = {"choices": [{"text": quoted + fake_ai_body}]}

    class DummyResponse:
        def __init__(self, json_data):
            self._json = json_data
            self.status_code = 200

        def raise_for_status(self):
            pass  # no-op, since status_code is 200

        def json(self):
            return self._json

    # Patch requests.post so that _get_response_server never actually sends HTTP:
    with patch("requests.post") as mock_post:
        mock_post.return_value = DummyResponse(fake_json)

        args = [
            "--prompt", fake_prompt,
            "--scope", "code",
            "--submission", "../test_submissions/ggr274_homework5/test1/student_submission.ipynb",
            "--solution",   "../test_submissions/ggr274_homework5/test1/Homework_5_solution.ipynb",
            "--model", "deepSeek-v3",
            "--llama_mode", "server",
            "--output", "stdout",
        ]
        output = run_cli_and_capture(args, capsys)

    assert "FAKE SERVER RESPONSE LINE 1" in output
    assert "FAKE SERVER RESPONSE LINE 2" in output

