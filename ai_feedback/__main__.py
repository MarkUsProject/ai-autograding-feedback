import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from . import code_processing, image_processing, text_processing
from .helpers import arg_options
from .helpers.constants import HELP_MESSAGES, TEST_OUTPUTS_DIRECTORY


def detect_submission_type(filename: str) -> str:
    """Automatically detect the submission type based on file extensions.

    Args:
        filename (str): Path to the file.

    Returns:
        str: The detected submission type ("jupyter", "python", or "pdf").
    """
    if filename.endswith(".ipynb"):
        return "jupyter"
    elif filename.endswith(".py"):
        return "python"
    elif filename.endswith(".pdf"):
        return "pdf"

    print("Error: Could not auto-detect submission type.")
    sys.exit(1)


def load_markdown_template(template: str) -> str:
    """
    Loads the markdown template used for formatting output.

    Args:
        template (str): name of markdown template.

    Returns:
        str: The markdown template as a string.

    Raises:
        SystemExit: If the template file is not found, the program will print an error and exit.
    """
    try:
        template_file = os.path.join(os.path.dirname(__file__), f"data/output/{template}.md")
        with open(template_file, "r") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: Markdown template file '{template}.md' not found.")
        sys.exit(1)


def load_markdown_prompt(prompt_name: str) -> dict:
    """Loads a markdown prompt file.

    Args:
        prompt_name (str): Name of the prompt file (without extension)

    Returns:
        dict: Dictionary containing prompt_content

    Raises:
        SystemExit: If the prompt file is not found
    """
    try:
        prompt_file = os.path.join(os.path.dirname(__file__), f"data/prompts/user/{prompt_name}.md")
        with open(prompt_file, "r") as file:
            prompt_content = file.read()
        return {"prompt_content": prompt_content}
    except FileNotFoundError:
        print(f"Error: Prompt file '{prompt_name}.md' not found in user subfolder.")
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
    parser.add_argument("--prompt_text", type=str, required=False, help=HELP_MESSAGES["prompt_text"])
    parser.add_argument("--prompt_custom", type=str, required=False, help=HELP_MESSAGES["prompt_custom"])
    parser.add_argument(
        "--scope",
        type=str,
        choices=arg_options.get_enum_values(arg_options.Scope),
        required=True,
        help=HELP_MESSAGES["scope"],
    )
    parser.add_argument("--submission", type=str, required=True, help=HELP_MESSAGES["submission"])
    parser.add_argument("--solution", type=str, required=False, default="", help=HELP_MESSAGES["solution"])
    parser.add_argument("--question", type=str, required=False, help=HELP_MESSAGES["question"])
    parser.add_argument(
        "--model",
        type=str,
        choices=arg_options.get_enum_values(arg_options.Models),
        required=True,
        help=HELP_MESSAGES["model"],
    )
    parser.add_argument(
        "--remote_model",
        type=str,
        required=False,
        help=HELP_MESSAGES["remote_model"],
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default='',
        help=HELP_MESSAGES["output"],
    )
    parser.add_argument("--test_output", type=str, required=False, default=None, help=HELP_MESSAGES["test_output"])
    parser.add_argument("--submission_image", type=str, required=False, help=HELP_MESSAGES["submission_image"])
    parser.add_argument("--solution_image", type=str, required=False, help=HELP_MESSAGES["solution_image"])
    parser.add_argument(
        "--output_template",
        required=False,
        type=str,
        choices=arg_options.get_enum_values(arg_options.OutputTemplate),
        default='response_only',
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        required=False,
        choices=arg_options.get_enum_values(arg_options.SystemPrompt),
        help=HELP_MESSAGES["system_prompt"],
        default="student_test_feedback",
    )
    parser.add_argument(
        "--llama_mode",
        type=str,
        choices=arg_options.get_enum_values(arg_options.LlamaMode),
        required=False,
        default="cli",
        help=HELP_MESSAGES["llama_mode"],
    )

    args = parser.parse_args()

    # Auto-detect submission type if not provided
    if args.submission_type is None:
        args.submission_type = detect_submission_type(args.submission)

    prompt_content = ""

    system_prompt_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"data/prompts/system/{args.system_prompt}.md"
    )
    with open(system_prompt_path, "r") as file:
        system_instructions = file.read()

    if args.prompt_custom:
        prompt_filename = os.path.join("./", args.prompt_custom)
        with open(prompt_filename, "r") as prompt_file:
            prompt_content += prompt_file.read()
    else:
        if args.prompt:
            if not args.prompt.startswith("image") and args.scope == "image":
                print("Error: The prompt must start with 'image'. Please re-run the command with a valid prompt.")
                sys.exit(1)
            if not args.prompt.startswith("code") and args.scope == "code":
                print("Error: The prompt must start with 'code'. Please re-run the command with a valid prompt.")
                sys.exit(1)
            if not args.prompt.startswith("text") and args.scope == "text":
                print("Error: The prompt must start with 'text'. Please re-run the command with a valid prompt.")
                sys.exit(1)

            prompt = load_markdown_prompt(args.prompt)
            prompt_content += prompt["prompt_content"]

        if args.prompt_text:
            prompt_content += args.prompt_text

    if args.scope == "image":
        prompt["prompt_content"] = prompt_content
        request, response = image_processing.process_image(args, prompt, system_instructions)
    elif args.scope == "text":
        request, response = text_processing.process_text(args, prompt_content, system_instructions)
    else:
        request, response = code_processing.process_code(args, prompt_content, system_instructions)

    markdown_template = load_markdown_template(args.output_template)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_text = markdown_template.format(
        question=args.question or "N/A",
        model=args.model,
        request=request,
        response=response,
        timestamp=timestamp,
        submission=args.submission,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(output_text)
    else:
        print(output_text)
    return 0


if __name__ == "__main__":
    main()
