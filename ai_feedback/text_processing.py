import sys
from pathlib import Path
from typing import Tuple

from .helpers.arg_options import model_mapping
from .helpers.template_utils import render_prompt_template


def process_text(args, prompt: str, system_instructions: str) -> Tuple[str, str]:
    """
    Processes text-based assignment files and generates a response using the selected model.

    This function looks for files matching specific suffixes in the provided assignment folder,
    updates the user prompt to reference these files, and invokes a model to generate feedback
    or analysis based on the provided scope and question.

    Args:
        args: Command-line argument namespace containing assignment, model, scope, and question attributes.
        prompt (str): The initial user prompt.
        system_instructions (str): instructions for the model

    Returns:
        Tuple[str, str]: A tuple containing the request and the model's generated response.

    Raises:
        FileNotFoundError: If the specified submission or solution files do not exist.
        SystemExit: If the model name provided in args is not recognized.
    """
    submission_file = Path(args.submission)
    if not submission_file.is_file():
        raise FileNotFoundError(f"Submission file '{submission_file}' not found.")
    solution_file = None
    if args.solution:
        solution_file = Path(args.solution)
        if not solution_file.is_file():
            raise FileNotFoundError(f"Solution file '{solution_file}' not found.")

    assignment_files = [args.submission, args.solution] if solution_file else [args.submission]

    rendered_prompt = render_prompt_template(prompt, assignment_files=assignment_files)

    if args.model in model_mapping:
        model = model_mapping[args.model]()
    else:
        print("Invalid model selected for text scope.")
        sys.exit(1)

    if args.question:
        request, response = model.generate_response(
            prompt=rendered_prompt,
            solution_file=solution_file,
            submission_file=submission_file,
            scope=args.scope,
            question_num=args.question,
            system_instructions=system_instructions,
            llama_mode=args.llama_mode,
        )
    else:
        request, response = model.generate_response(
            prompt=rendered_prompt,
            solution_file=solution_file,
            submission_file=submission_file,
            scope=args.scope,
            system_instructions=system_instructions,
            llama_mode=args.llama_mode,
        )

    return request, response
