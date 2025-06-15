import os
import os.path

import pytest
from llm_helpers import *


def test_with_markers(request):
    """Generates LLM response"""
    # Run LLM feedback
    llm_feedback = run_llm(
        submission="pdf",
        scope="text",
        output="stdout",
        model="openai",
        prompt="text_pdf_analyze",
    )

    request.node.add_marker(pytest.mark.markus_message(llm_feedback))
    # Display LLM output in the overall comment
    request.node.add_marker(pytest.mark.markus_overall_comments(llm_feedback))
