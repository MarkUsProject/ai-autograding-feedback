import sys
from pathlib import Path
from typing import Optional

from ollama import Image, Message
from PIL import Image as PILImage

from .helpers.image_extractor import extract_images, extract_qmd_python_images
from .helpers.image_reader import *
from .helpers.template_utils import render_prompt_template


def process_image(
    model, args, prompt: dict, system_instructions: str, marking_instructions: Optional[str] = None
) -> tuple[str, str]:
    """Generates feedback for an image submission.
    Returns the LLM prompt delivered and the returned response."""
    OUTPUT_DIRECTORY = "output_images"
    submission_notebook = Path(args.submission)
    solution_notebook = None
    if args.solution:
        solution_notebook = Path(args.solution)
    # Extract submission images
    if submission_notebook.suffix == ".qmd":
        images = extract_qmd_python_images(submission_notebook, OUTPUT_DIRECTORY)
    else:
        images = extract_images(submission_notebook, OUTPUT_DIRECTORY, "submission")
    # Optionally extract solution images
    if args.solution and solution_notebook.is_file():
        extract_images(solution_notebook, OUTPUT_DIRECTORY, "solution")

    if args.question:
        questions = [args.question]
    else:
        questions = os.listdir(OUTPUT_DIRECTORY)

    requests: list[str] = []
    responses: list[str] = []

    for question in questions:
        # Start with the raw prompt content
        prompt_content = prompt["prompt_content"]

        # Validate required image arguments based on prompt placeholders
        if "{submission_image}" in prompt_content and not args.submission_image:
            raise SystemExit(f"Prompt requires submission image but --submission-image not provided.")
        if "{solution_image}" in prompt_content and not args.solution_image:
            raise SystemExit(f"Prompt requires solution image but --solution-image not provided.")

        # Always replace {context} when it appears
        if "{context}" in prompt_content:
            context = read_question_context(OUTPUT_DIRECTORY, question)
            prompt_content = prompt_content.replace("{context}", "```\n" + context + "\n```")
        if "{image_size}" in prompt_content:
            submission_image_path = args.submission_image
            # Only consider one image per question
            image = PILImage.open(submission_image_path)
            prompt_content = prompt_content.replace("{image_size}", f"{image.width} by {image.height}")

        rendered_prompt = render_prompt_template(
            prompt_content,
            submission=submission_notebook,
            solution=solution_notebook,
            has_submission_image="{submission_image}" in prompt_content,
            has_solution_image="{solution_image}" in prompt_content and args.solution_image,
            marking_instructions=marking_instructions,
        )

        message = Message(role="user", content=rendered_prompt, images=[])
        if "{submission_image}" in prompt_content:
            # Only consider one image per question
            submission_image_path = args.submission_image
            message.images.append(Image(value=submission_image_path))
        if "{solution_image}" in prompt_content:
            # Only consider one image per question
            solution_image_path = args.solution_image
            message.images.append(Image(value=solution_image_path))

        for image in images:
            message.images.append(Image(value=image))

        # Prompt the LLM
        requests.append(f"{message.content}\n\n{[str(image.value) for image in message.images]}")

        args.rendered_prompt = rendered_prompt
        args.system_instructions = system_instructions
        responses.append(model.process_image(message, args))

    return "\n\n---\n\n".join(requests), "\n\n---\n\n".join(responses)
