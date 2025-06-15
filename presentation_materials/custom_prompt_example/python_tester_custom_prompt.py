import os
import os.path

import pytest

# Modify depending on name of student's submission file
import student_submission as submission
from llm_helpers import *

# NOTE: add numpy and torch in package requirements section of autotester settings specifically for cnn_submission

llm_feedback = ""


def test_with_feedback(request):
    """Generates LLM Feedback"""
    global llm_feedback
    # change prompt and model here
    llm_feedback = run_llm(
        submission="python",
        prompt_custom=True,
        prompt_text="custom_prompt",
        scope="code",
        model="claude-3.7-sonnet",
        output="stdout",
    )
    request.node.add_marker(pytest.mark.markus_message(llm_feedback))
    request.node.add_marker(pytest.mark.markus_overall_comments(llm_feedback))
