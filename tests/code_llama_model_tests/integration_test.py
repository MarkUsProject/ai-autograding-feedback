from pathlib import Path

from tests.test_helper import mock_and_capture, run_cli_and_capture


def test_ggr274_image_question5b_llama_stdout(capsys, mock_and_capture):
    """
    Example 5:
    Evaluate the image for question 5b of ggr274 homework with Llama3.2-vision, printing to stdout.
    python3 -m ai_feedback --prompt image_analyze --scope image \
      --solution ./test_submissions/ggr274_homework5/image_test2/student_submission.ipynb \
      --submission_image test_submissions/ggr274_homework5/image_test2/student_submission.png \
      --question "Question 5b" --model llama3.2-vision --output stdout
    """
    parent = Path(__file__).parent.parent.parent
    args = [
        "--prompt",
        "image_analyze",
        "--scope",
        "image",
        "--submission",
        str(parent / "test_submissions/ggr274_homework5/image_test2/student_submission.ipynb"),
        "--submission_image",
        str(parent / "test_submissions/ggr274_homework5/image_test2/student_submission.png"),
        "--question",
        "Question 5b",
        "--model",
        "llama3.2-vision:90b",
    ]

    output = run_cli_and_capture(args, capsys)
    assert "Create the same boxplots as in Task 4 f), but use `Age group label`" in output
