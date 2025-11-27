import argparse
import os
import os.path
import sys
from datetime import datetime
from pathlib import Path

from . import code_processing, image_processing, text_processing
from .models import ModelFactory
from .helpers import arg_options
from .helpers.constants import HELP_MESSAGES

_TYPE_BY_EXTENSION = {
    '.c': 'C',
    '.ipynb': 'Jupyter',
    '.java': 'Java',
    '.pdf': 'PDF',
    '.py': 'Python',
    '.r': 'R',
    '.rmd': 'RMarkdown',
    '.sql': 'SQL',
    '.qmd': 'Quarto',
}


def detect_submission_type(filename: str) -> str:
    """Automatically detect the submission type based on file extensions.

    Args:
        filename (str): Path to the file.

    Returns:
        str: The detected submission type.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext in _TYPE_BY_EXTENSION:
        return _TYPE_BY_EXTENSION[ext]
    else:
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


def _load_content_with_fallback(
    content_arg: str, predefined_values: list[str], predefined_subdir: str, content_type: str
) -> str:
    """Generic function to load content by trying pre-defined names first, then treating as file path.

    Args:
        content_arg (str): Either a pre-defined name or a file path
        predefined_values (list[str]): List of valid pre-defined names
        predefined_subdir (str): Subdirectory for pre-defined files (e.g., "user", "system")
        content_type (str): Type of content for error messages (e.g., "prompt", "system prompt")

    Returns:
        str: The content

    Raises:
        SystemExit: If the content cannot be loaded
    """
    # First, check if it's a pre-defined name
    if content_arg in predefined_values:
        try:
            file_path = os.path.join(os.path.dirname(__file__), f"data/prompts/{predefined_subdir}/{content_arg}.md")
            with open(file_path, "r", encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            print(
                f"Error: Pre-defined {content_type} file '{content_arg}.md' not found in {predefined_subdir} subfolder."
            )
            sys.exit(1)
    else:
        # Treat as a file path
        try:
            with open(content_arg, "r", encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            print(f"Error: {content_type.title()} file '{content_arg}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading {content_type} file '{content_arg}': {e}")
            sys.exit(1)


def load_prompt_content(prompt_arg: str) -> str:
    """Loads prompt content by trying pre-defined names first, then treating as file path.

    Args:
        prompt_arg (str): Either a pre-defined prompt name or a file path

    Returns:
        str: The prompt content

    Raises:
        SystemExit: If the prompt cannot be loaded
    """
    return _load_content_with_fallback(prompt_arg, arg_options.get_enum_values(arg_options.Prompt), "user", "prompt")


def load_system_prompt_content(system_prompt_arg: str) -> str:
    """Loads system prompt content by trying pre-defined names first, then treating as file path.

    Args:
        system_prompt_arg (str): Either a pre-defined system prompt name or a file path

    Returns:
        str: The system prompt content

    Raises:
        SystemExit: If the system prompt cannot be loaded
    """
    return _load_content_with_fallback(
        system_prompt_arg, arg_options.get_enum_values(arg_options.SystemPrompt), "system", "system prompt"
    )


def load_marking_instructions_content(marking_instructions_path: str) -> str:
    """Loads marking instructions content from a file path.

    Args:
        marking_instructions_path (str): Path to the marking instructions file

    Returns:
        str: The marking instructions content

    Raises:
        SystemExit: If the file cannot be loaded
    """
    try:
        with open(marking_instructions_path, "r", encoding='utf-8') as file:
            return file.read()
    except Exception:
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
        required=False,
        help=HELP_MESSAGES["prompt"],
    )
    parser.add_argument("--prompt_text", type=str, required=False, help=HELP_MESSAGES["prompt_text"])
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
        "--provider",
        type=str,
        choices=ModelFactory.get_available_providers(),
        required=True,
        help=HELP_MESSAGES["provider"],
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        help=HELP_MESSAGES["model_name"],
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
    parser.add_argument(
        "--json_schema",
        type=str,
        required=False,
        default="",
        help=HELP_MESSAGES["json_schema"],
    )
    parser.add_argument(
        "--marking_instructions",
        type=str,
        required=False,
        help=HELP_MESSAGES["marking_instructions"],
    )
    parser.add_argument(
        "--model_options",
        type=str,
        required=False,
        default="",
        help=HELP_MESSAGES["model_options"],
    )

    args = parser.parse_args()

    if args.model_options:
        args.model_options = dict(pair.split('=') for pair in args.model_options.split(','))
    else:
        args.model_options = {}

    # Auto-detect submission type if not provided
    if args.submission_type is None:
        args.submission_type = detect_submission_type(args.submission)

    prompt_content = ""
    system_instructions = load_system_prompt_content(args.system_prompt)

    marking_instructions = None
    if args.marking_instructions:
        marking_instructions = load_marking_instructions_content(args.marking_instructions)

    if args.prompt:
        # Only validate scope for pre-defined prompts (not for arbitrary file paths)
        predefined_prompts = arg_options.get_enum_values(arg_options.Prompt)
        if args.prompt in predefined_prompts:
            if not args.prompt.startswith("image") and args.scope == "image":
                print("Error: The prompt must start with 'image'. Please re-run the command with a valid prompt.")
                sys.exit(1)
            if not args.prompt.startswith("code") and args.scope == "code":
                print("Error: The prompt must start with 'code'. Please re-run the command with a valid prompt.")
                sys.exit(1)
            if not args.prompt.startswith("text") and args.scope == "text":
                print("Error: The prompt must start with 'text'. Please re-run the command with a valid prompt.")
                sys.exit(1)

        prompt_content += load_prompt_content(args.prompt)

    if args.prompt_text:
        prompt_content += args.prompt_text

    try:
        model_args = {'model_name': args.model_name} if args.model_name else {}
        model = ModelFactory.create(args.provider, **model_args)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.scope == "image":
        prompt = {"prompt_content": prompt_content}
        request, response = image_processing.process_image(
            model, args, prompt, system_instructions, marking_instructions
        )
    elif args.scope == "text":
        request, response = text_processing.process_text(
            model, args, prompt_content, system_instructions, marking_instructions
        )
    else:
        request, response = code_processing.process_code(
            model, args, prompt_content, system_instructions, marking_instructions
        )

    markdown_template = load_markdown_template(args.output_template)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_text = markdown_template.format(
        question=args.question or "N/A",
        model=args.provider,
        request=request,
        response=response,
        timestamp=timestamp,
        submission=args.submission,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_text)
    else:
        print(output_text)
    return 0


if __name__ == "__main__":
    main()
