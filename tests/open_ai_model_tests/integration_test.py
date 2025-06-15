from pathlib import Path

from tests.test_helper import mock_and_capture, run_cli_and_capture


def test_cnn_example_openai_stdout(capsys, mock_and_capture):
    """
    Example 1:
    Evaluate cnn_example test using openAI model and print to stdout.
    python -m ai_feedback --prompt code_lines --scope code \
        --submission test_submissions/cnn_example/cnn_submission \
        --solution test_submissions/cnn_example/cnn_solution.py \
        --model openai --output stdout
    """
    parent = Path(__file__).parent.parent.parent

    args = [
        "--prompt",
        "code_lines",
        "--scope",
        "code",
        "--submission",
        str(parent / "test_submissions/cnn_example/cnn_submission.py"),
        "--solution",
        str(parent / "test_submissions/cnn_example/cnn_solution.py"),
        "--model",
        "openai",
    ]
    output = run_cli_and_capture(args, capsys)

    assert "Prompt: Compare the student's code and solution code. For each mistake" in output
    assert "(Line 1) import numpy as np" in output
    assert "=== cnn_submission.py ===" in output
    assert "=== cnn_solution.py ===" in output


def test_cnn_example_custom_prompt_stdout(capsys, mock_and_capture):
    """
    Example 2:
    Evaluate cnn_example test using openAI model and a custom prompt text, printing to stdout.
    python -m ai_feedback --prompt_text "Evaluate the student's code readability." \
        --scope code \
        --submission test_submissions/cnn_example/cnn_submission.py \
        --model openai \
        --output stdout
    """
    parent = Path(__file__).parent.parent.parent

    args = [
        "--prompt_text",
        "Evaluate the student's code readability.",
        "--scope",
        "code",
        "--submission",
        str(parent / "test_submissions/cnn_example/cnn_submission.py"),
        "--model",
        "openai",
    ]
    output = run_cli_and_capture(args, capsys)
    assert "Prompt: Evaluate the student's code readability." in output
    assert "=== cnn_submission.py ===" in output
    assert "(Line 1) import numpy as np" in output


def test_pdf_example_openai_direct(capsys, mock_and_capture):
    """
    Example 3:
    Evaluate pdf_example test using openAI model and direct output mode.
    python -m ai_feedback --prompt text_pdf_analyze --scope text \
        --submission test_submissions/pdf_example/student_pdf_submission.pdf \
        --model openai --output direct
    """
    parent = Path(__file__).parent.parent.parent
    args = [
        "--prompt",
        "text_pdf_analyze",
        "--scope",
        "text",
        "--submission",
        str(parent / "test_submissions/pdf_example/student_pdf_submission.pdf"),
        "--model",
        "openai",
    ]

    output = run_cli_and_capture(args, capsys)
    assert "Prompt: Does the student correctly respond to the question, and meet all the" in output
    assert "student_pdf_submission.pdf" in output
    assert "Normalization allows each feature to have an equal influence on the mode" in output
