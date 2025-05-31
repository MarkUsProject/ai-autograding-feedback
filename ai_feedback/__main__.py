import argparse
import json
import os
import sys
from datetime import datetime
from typing import Tuple

from . import image_processing
from . import code_processing
from . import text_processing
from .helpers import arg_options
from .helpers.constants import TEST_OUTPUTS_DIRECTORY, HELP_MESSAGES


def detect_submission_type(assignment_folder: str) -> str:
    """Automatically detect the submission type based on file extensions in the assignment folder.
    
    Args:
        assignment_folder (str): Path to the assignment directory.
        
    Returns:
        str: The detected submission type ("jupyter", "python", or "pdf").
    """
    for filename in os.listdir(assignment_folder):
        if filename.endswith("_submission.ipynb"):
            return "jupyter"
        elif filename.endswith("_submission.py"):
            return "python"
        elif filename.endswith("_submission.pdf"):
            return "pdf"
    
    print("Error: Could not auto-detect submission type.")
    sys.exit(1)


def load_markdown_template() -> str:
    """
    Loads the markdown template used for formatting output.

    Returns:
        str: The markdown template as a string.

    Raises:
        SystemExit: If the template file is not found, the program will print an error and exit.
    """
    try:
        template_file = os.path.join(
            os.path.dirname(__file__), "data/output/output_template.md"
        )
        with open(template_file, "r") as file:
            return file.read()
    except FileNotFoundError:
        print("Error: Markdown template file 'output_template.md' not found.")
        sys.exit(1)


def main() -> int:
    """
    Parses command-line arguments to determine the type of submission, scope,
    model, and output format. It loads prompts, delegates the processing to specialized
    modules (image, text, or code), and handles output generation as markdown
    or standard output.

    Returns:
        int: Exit status code (0 for success).
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--submission_type",
        type=str,
        choices=arg_options.get_enum_values(arg_options.FileType),
        required=False,
        default=None,
        help=HELP_MESSAGES["submission_type"],
    )
    parser.add_argument(
        "--prompt",
        type=str,
        choices=arg_options.get_enum_values(arg_options.Prompt),
        required=False,
        help=HELP_MESSAGES["prompt"],
    )
    parser.add_argument(
        "--prompt_text", type=str, required=False, help=HELP_MESSAGES["prompt_text"]
    )
    parser.add_argument("--prompt_custom", action="store_true", required=False)
    parser.add_argument(
        "--scope",
        type=str,
        choices=arg_options.get_enum_values(arg_options.Scope),
        required=True,
        help=HELP_MESSAGES["scope"],
    )
    parser.add_argument(
        "--assignment", type=str, required=True, help=HELP_MESSAGES["assignment"]
    )
    parser.add_argument(
        "--question", type=str, required=False, help=HELP_MESSAGES["question"]
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=arg_options.get_enum_values(arg_options.Models),
        required=True,
        help=HELP_MESSAGES["model"],
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=arg_options.get_enum_values(arg_options.OutputType),
        required=False,
        default='stdout',
        help=HELP_MESSAGES["output"],
    )

    args = parser.parse_args()

    # Auto-detect submission type if not provided
    if args.submission_type is None:
        args.submission_type = detect_submission_type(args.assignment)

    prompt_content = ""

    if args.prompt_custom:
        prompt_filename = os.path.join("./", f"{args.prompt_text}.txt")
        print(prompt_filename)
        with open(prompt_filename, "r") as prompt_file:
            prompt_content += prompt_file.read()
    else:
        if args.prompt:
            if not args.prompt.startswith("image") and args.scope == "image":
                print(
                    "Error: The prompt must start with 'image'. Please re-run the command with a valid prompt."
                )
                sys.exit(1)
            if not args.prompt.startswith("code") and args.scope == "code":
                print(
                    "Error: The prompt must start with 'code'. Please re-run the command with a valid prompt."
                )
                sys.exit(1)
            if not args.prompt.startswith("text") and args.scope == "text":
                print(
                    "Error: The prompt must start with 'text'. Please re-run the command with a valid prompt."
                )
                sys.exit(1)

            prompt_filename = os.path.join(
                os.path.dirname(__file__), f"data/prompts/{args.prompt}.json"
            )
            with open(prompt_filename, "r") as prompt_file:
                prompt = json.load(prompt_file)
                prompt_content += prompt["prompt_content"]

        if args.prompt_text:
            prompt_content += args.prompt_text

    if args.scope == "image":
        prompt["prompt_content"] = prompt_content
        request, response = image_processing.process_image(args, prompt)
    elif args.scope == "text":
        request, response = text_processing.process_text(args, prompt_content)
    else:
        request, response = code_processing.process_code(args, prompt_content)

    if args.output == "markdown":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        assignment_directory = f"{TEST_OUTPUTS_DIRECTORY}/{args.model}"
        os.makedirs(assignment_directory, exist_ok=True)

        markdown_filename = f"{assignment_directory}/{args.prompt}_{timestamp}.md"

        markdown_template = load_markdown_template()
        markdown_output = markdown_template.format(
            assignment=args.assignment,
            question=args.question if args.question else "N/A",
            model=args.model,
            request=request,
            response=response,
            timestamp=timestamp,
        )
        with open(markdown_filename, "w") as md_file:
            md_file.write(markdown_output)
        print(f"Markdown report saved to `{markdown_filename}`")

    elif args.output == "stdout":
        print(response)

    elif args.output == "direct":
        print(response)

    return 0


if __name__ == "__main__":
    main()
