import os
from pathlib import Path
from ollama import Message
from typing import Optional, Tuple

import anthropic
from dotenv import load_dotenv

from ..helpers.image_extractor import encode_image
from ..helpers.model_options_helpers import cast_to_type, claude_option_schema
from .Model import Model

# Load environment variables from .env file
load_dotenv()


class ClaudeModel(Model):
    def __init__(self, model_name: str = None) -> None:
        """
        Initializes the ClaudeModel with the Anthropic client using an API key.
        """
        super().__init__(model_name)
        self.model_name = model_name if model_name else "claude-3-7-sonnet-20250219"
        self.client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

    def generate_response(
        self,
        prompt: str,
        submission_file: Path,
        system_instructions: str,
        model_options: Optional[dict] = None,
        solution_file: Optional[Path] = None,
        scope: Optional[str] = None,
        question: Optional[str] = None,
        test_output: Optional[Path] = None,
        llama_mode: Optional[str] = None,
        json_schema: Optional[str] = None,
    ) -> Optional[Tuple[str, str]]:
        """
        Generates a response from Claude using the provided prompt and assignment file context.

        Args:
            prompt (str): The user's prompt for the model.
            solution_file: path to the solution file.
            submission_file: path to the submission file.
            test_output: path to the file containing the results of tests run on user submission
            scope (Optional[str]): The content scope.
            question (Optional[str]): Specific task to extract from text files.
            system_instructions (str): instructions for the model
            llama_mode (Optional[str]): Optional mode to invoke llama.cpp in.
            json_schema (Optional[str]): Optional json schema to use.
            model_options (Optional[dict]): The optional model options to use for generating the response.

        Returns:
            Optional[Tuple[str, str]]: The original prompt and the model's response, or None if the response is invalid.
        """
        request = ""

        if question:
            request += f" Identify and generate a response for the mistakes **only** in question/task ${question}. "

        request += prompt

        model_options = cast_to_type(claude_option_schema, model_options)

        # Construct request parameters
        request_kwargs = {
            "model": self.model_name,
            "system": system_instructions,
            "messages": [{"role": "user", "content": request}],
            **model_options,
        }

        response = self.client.messages.create(**request_kwargs)

        if not response or not response.content:
            print("Error: Invalid or empty response from Claude.")
            return None

        return prompt, response.content[0].text

    def process_image(self, message: Message, args) -> str:
        """Sends a request to Claude"""
        images = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": f"{encode_image(image.value)}",
                },
            }
            for image in message.images
        ]
        response = self.client.messages.create(
            max_tokens=2048,
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": message.content,
                        }
                    ]
                    + images,
                }
            ],
            temperature=0.33,
        )
        return response.content[0].text
