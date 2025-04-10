import base64
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from ollama import chat, Message, Image
from openai import OpenAI
from PIL import Image as PILImage
from .helpers.arg_options import Models
from .helpers.image_extractor import extract_images
from .helpers.image_reader import *


def encode_image(image_path: os.PathLike):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def openai_call(message: Message, model: str):
    # Load environment variables from .env file
    load_dotenv()
    client = OpenAI()
    images = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image(image.value)}"}
        } for image in message.images
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": message.content,
                    }
                ] + images,
            }
        ],
        temperature=0.33
    )
    return completion.choices[0].message.content

def anthropic_call(message: Message, model: str):
    load_dotenv()
    client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
    images = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": f"{encode_image(image.value)}"
            }
        } for image in message.images
    ]
    message = client.messages.create(
        max_tokens=2048,
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": message.content,
                    }
                ] + images,
            }
        ],
        temperature=0.33
    )
    return message.content[0].text

def process_image(args, prompt):
    OUTPUT_DIRECTORY = "output_images"
    submission_notebook = Path("./", args.assignment, "student_submission.ipynb")
    solution_notebook = Path("./", args.assignment, "solution.ipynb")

    # Extract submission images
    extract_images(submission_notebook, OUTPUT_DIRECTORY, "submission")
    # Optionally extract solution images
    if solution_notebook.is_file():
        extract_images(solution_notebook, OUTPUT_DIRECTORY, "solution")

    message = Message(
        role="user",
        content=prompt["prompt_content"],
        images=[]
    )

    # Attachments
    if prompt.get("include_question_context", False):
        context = read_question_context(OUTPUT_DIRECTORY, args.question)
        message.content = message.content.replace("{context}", "```\n" + context + "\n```")
    if prompt.get("include_image_size", False):
        submission_image_paths = read_submission_images(OUTPUT_DIRECTORY, args.question)
        submission_image_path = submission_image_paths[0] # Only consider one image per question
        image = PILImage.open(submission_image_path)
        message.content = message.content.replace("{image_size}", f"{image.width} by {image.height}")
    if prompt.get("include_submission_image", False):
        submission_image_paths = read_submission_images(OUTPUT_DIRECTORY, args.question)
        submission_image_path = submission_image_paths[0] # Only consider one image per question
        message.images.append(Image(value=submission_image_path))
    if prompt.get("include_solution_image", False):
        solution_image_paths = read_solution_images(OUTPUT_DIRECTORY, args.question)
        solution_image_path = solution_image_paths[0] # Only consider one image per question
        message.images.append(Image(value=solution_image_path))

    request = f"{message.content}\n\n{[str(image.value) for image in message.images]}"
    if args.model == Models.OPENAI.value:
        response = openai_call(message, model="gpt-4o")
    elif args.model == Models.CLAUDE.value:
        response = anthropic_call(message, model="claude-3-7-sonnet-20250219")
    else:
        response = chat(model=args.model, messages=[message], options={"temperature": 0.33}).message.content
    return request, response