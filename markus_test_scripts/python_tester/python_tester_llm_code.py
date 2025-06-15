import os
import os.path

import pytest

# Modify depending on name of student's submission file
import student_submission as submission
from llm_helpers import *

# NOTE: add numpy and torch in package requirements section of autotester settings specifically for cnn_submission

llm_feedback = ""


def test_with_feedback(request):
    """Generates LLM Feedback for code scope of assignments"""
    global llm_feedback
    # change prompt and model here
    llm_feedback = run_llm(
        submission="python",
        prompt="code_table",
        scope="code",
        model="claude-3.7-sonnet",
        output="stdout",
    )
    request.node.add_marker(pytest.mark.markus_message(llm_feedback))
    request.node.add_marker(pytest.mark.markus_overall_comments(llm_feedback))


def test_with_annotations(request):
    """Generates LLM Annotations"""
    # feed in previous LLM message to create annotations
    prompt = f"Previous message: {llm_feedback}."
    prompt += ANNOTATION_PROMPT

    # Run LLM feedback
    raw_annotation = run_llm(
        submission="python",
        prompt_text=prompt,
        scope="code",
        model="claude-3.7-sonnet",
        output="direct",
    )  # generate annotations

    annotations_json_list = extract_json(raw_annotation)
    annotations = annotations_json_list[0]["annotations"]

    annotations_with_columns = add_annotation_columns(annotations, submission)

    request.node.add_marker(pytest.mark.markus_message(str(annotations_with_columns)))

    for annotation in annotations_with_columns:
        filename = annotation["filename"]
        content = annotation["content"]
        line_start = annotation["line_start"]
        line_end = annotation["line_end"]
        column_start = annotation["column_start"]
        column_end = annotation["column_end"]

        # Convert filename to relative path
        rel_filename = os.path.relpath(filename, os.getcwd())

        # Add pytest marker for each annotation separately
        request.node.add_marker(
            pytest.mark.markus_annotation(
                filename=rel_filename,
                content=content,
                line_start=line_start,
                line_end=line_end,
                column_start=column_start,
                column_end=column_end,
            )
        )
