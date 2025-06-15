from pathlib import Path

from tests.test_helper import mock_and_capture, run_cli_and_capture


def test_ggr274_question1_deepseek_markdown(capsys, mock_and_capture):
    """
    Example 4:
    Evaluate question 1 of test1 of ggr274 homework using DeepSeek model, output as Markdown.
    python -m ai_feedback --prompt code_table --scope code \
        --submission test_submissions/ggr274_homework5/test1/student_submission.ipynb \
        --question 1 --model deepSeek-R1:70B --output markdown
    """
    parent = Path(__file__).parent.parent.parent

    args = [
        "--prompt",
        "code_table",
        "--scope",
        "code",
        "--submission",
        str(parent / "test_submissions/ggr274_homework5/test1/student_submission.ipynb"),
        "--question",
        "1",
        "--model",
        "deepSeek-R1:70B",
    ]
    output = run_cli_and_capture(args, capsys)
    assert "Compare the student's code and solution code" in output
    assert "student_submission.ipynb" in output
    assert "=== student_submission.ipynb ===\n" in output
