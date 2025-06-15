from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from tests.test_helper import run_cli_and_capture


def test_deepseek_v3_cli_code_scope_stdout(capsys, monkeypatch):
    """
    Evaluate a simple code-scope test using DeepSeekV3Model in CLI mode,
    printing to stdout. We mock subprocess.run so that llama-cli never actually runs.

    python3 -m ai_feedback --prompt code_table --scope code \
        --submission test_submissions/ggr274_homework5/test1/student_submission.ipynb \
        --solution test_submissions/ggr274_homework5/test1/Homework_5_solution.ipynb \
        --model deepSeek-v3
    """
    monkeypatch.setenv("LLAMA_CLI_PATH", "/path/to/fake/llama-cli")
    monkeypatch.setenv("LLAMA_MODEL_PATH", "/data1/GGUF/DeepSeek-V3-0324-UD-Q2_K_XL/DeepSeek-V3-0324-UD-Q2_K_XL.gguf")
    # Prepare fake CompletedProcess: stdout begins with the quoted prompt,
    # then a newline, then some dummy AI content followed by the end marker.
    fake_prompt = "'code_table'"
    fake_ai_output = "\nFAKE AI RESPONSE LINE 1\nFAKE AI RESPONSE LINE 2 [end of text]\n"
    fake_stdout = (fake_prompt + fake_ai_output).encode("utf-8")
    parent = Path(__file__).parent.parent.parent

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = CompletedProcess(args=["./llama-cli"], returncode=0, stdout=fake_stdout, stderr=b"")

        args = [
            "--prompt",
            "code_table",
            "--scope",
            "code",
            "--submission",
            str(parent / "test_submissions/ggr274_homework5/test1/student_submission.ipynb"),
            "--solution",
            str(parent / "test_submissions/ggr274_homework5/test1/Homework_5_solution.ipynb"),
            "--model",
            "deepSeek-v3",
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
    parent = Path(__file__).parent.parent.parent

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
            "--prompt",
            fake_prompt,
            "--scope",
            "code",
            "--submission",
            str(parent / "test_submissions/ggr274_homework5/test1/student_submission.ipynb"),
            "--solution",
            str(parent / "test_submissions/ggr274_homework5/test1/Homework_5_solution.ipynb"),
            "--model",
            "deepSeek-v3",
            "--llama_mode",
            "server",
        ]
        output = run_cli_and_capture(args, capsys)

    assert "FAKE SERVER RESPONSE LINE 1" in output
    assert "FAKE SERVER RESPONSE LINE 2" in output
