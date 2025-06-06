SYSTEM_INSTRUCTIONS = """You are a helpful assistant that provides detailed feedback on students' test errors.
Your goal is to identify mistakes in their assignments, explain why the mistakes occurred,
and offer guidance on how to correct them. Ensure the feedback is clear, supportive, and
encourages improvement."""
TEST_OUTPUTS_DIRECTORY = "test_responses_md"
HELP_MESSAGES = {
    "submission_type": "The format of the submission file (e.g., Jupyter notebook, Python script).",
    "prompt": "The specific prompt to use for evaluating the assignment.",
    "prompt_text": "Additional messages to concatenate to the prompt.",
    "scope": "The section of the assignment the model should analyze (e.g., code or image).",
    "submission": "The file path for the submission file.",
    "solution": "The file path for the solution file.",
    "question": "The specific question number to analyze within the assignment (if applicable).",
    "model": "The name of the LLM model to use for evaluation.",
    "output": "Format to display the output response.",
    "llama_mode": "Specifies how to invoke llama.cpp: either directly via its command‐line interface (CLI) or by sending requests to a running llama-server instance.",
    "test_output": "The output of tests from evaluating the assignment.",
    "submission_image": "The file path for the image file.",
    "solution_image": "The file path to the solution image."
}
