"""Tests for the CLI's handling of a gateway refusal.

The autotester runs this package as a subprocess and reports a failed run from
its stderr, so a refused call must exit non-zero with the reason on stderr.
"""

import sys

import pytest

from ai_feedback import code_processing
from ai_feedback.__main__ import main
from ai_feedback.models import GatewayError, ModelFactory

REASON = "Course budget exhausted for course_id=1: spent CAD 0.01 of CAD 0.01."


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """Run the CLI against a stub model, returning a runner for the given processor."""
    submission = tmp_path / "submission.py"
    submission.write_text("print('hi')\n")
    monkeypatch.setattr(ModelFactory, "create", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ai_feedback",
            "--scope",
            "code",
            "--submission",
            str(submission),
            "--provider",
            "openai-remote",
            "--prompt_text",
            "Review this code.",
        ],
    )
    return main


def test_gateway_refusal_exits_with_the_reason_on_stderr(monkeypatch, capsys, cli):
    def _refuse(*args, **kwargs):
        raise GatewayError(REASON)

    monkeypatch.setattr(code_processing, "process_code", _refuse)

    with pytest.raises(SystemExit) as exited:
        cli()

    assert exited.value.code == 1
    captured = capsys.readouterr()
    assert REASON in captured.err
    assert REASON not in captured.out
