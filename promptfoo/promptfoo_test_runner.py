import os
import re
import subprocess
from typing import Tuple


def call_api(prompt: str, context: dict, metadata: dict) -> dict:
    """
    Function that promptfoo will call

    Args:
        prompt (str): The prompt to be called
        context (dict): The context of the tests
        metadata (dict): The all metadata that is associated with the test

    Returns:
        dict: The output of the command or error message
    """
    options = metadata['vars']

    try:
        env = os.environ.copy()
        if options.get('submission_type'):
            submission_type = options['submission_type']
        else:
            submission_type = None

        cmd_args = [
            "python3",
            "-m",
            "ai_feedback",
            "--scope",
            options["scope"],
            "--model",
            options["model"],
            "--prompt",
            options['prompt'],
            "--llama_mode",
            "server",
            '--submission_type',
            submission_type,
            "--output_template",
            "response_and_prompt",
        ]

        if options.get('system_prompt'):
            cmd_args.extend(["--system_prompt", options['system_prompt']])

        if options.get('solution_file'):
            cmd_args.extend(["--solution", f"../{options['solution_file']}"])

        if options["scope"] == "image":
            if options['submission_file'].endswith('.png'):
                cmd_args.extend(
                    [
                        "--submission",
                        f"../{options['submission_file']}",
                        "--submission_image",
                        f"../{options['submission_file']}",
                    ]
                )
            else:
                cmd_args.extend(["--submission", f"../{options['submission_file']}"])

                submission_path = options['submission_file']
                submission_basename = os.path.splitext(os.path.basename(submission_path))[0]
                submission_dir = os.path.dirname(submission_path)

                expected_image = f"../{submission_dir}/{submission_basename}.png"

                cmd_args.extend(["--submission_image", expected_image])
        else:
            # For code
            cmd_args.extend(["--submission", f"../{options['submission_file']}"])

        result = subprocess.run(
            cmd_args,
            capture_output=True,
            env=env,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"ai_feedback failed (rc={result.returncode}):\n{result.stderr}")
        else:
            prompt, output = split_prompt_response(result.stdout.strip())
    except Exception as e:
        raise RuntimeError(f"[EXCEPTION] {e}")

    return {"output": output, "prompt": prompt}


def split_prompt_response(md_text: str) -> Tuple[str, str]:
    """
    Split a Markdown string into its prompt and response parts.

    Args:
        md_text: The full Markdown text.

    Returns:
        A tuple (prompt, response), both stripped of leading/trailing whitespace.

    Raises:
        ValueError: If the expected headings are not found.
    """
    pattern = r'^# Prompt\s*(.*?)^# Response\s*(.*)$'
    match = re.search(pattern, md_text, re.DOTALL | re.MULTILINE)
    if not match:
        raise ValueError("Markdown not in expected format with '# Prompt' and '# Response' headers")
    prompt = match.group(1).strip()
    response = match.group(2).strip()
    return prompt, response
