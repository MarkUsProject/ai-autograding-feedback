import json
import os
from pathlib import Path
from ollama import Message
from typing import Optional, Tuple
from ..helpers.image_extractor import encode_image

import openai
from dotenv import load_dotenv

from ai_feedback.helpers.model_options_helpers import (
    cast_to_type,
    openai_chat_option_schema,
)

from .Model import Model

load_dotenv()


class OpenAIModel(Model):
    def __init__(self, model_name: str = None) -> None:
        """
        Initialize an OpenAIModel instance.

        Loads the OpenAI API key from environment variables and prepares the client.
        """
        super().__init__(model_name)
        self.model_name = model_name if model_name else "gpt-4o"
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_response(
        self,
        prompt: str,
        submission_file: Path,
        system_instructions: str,
        model_options: Optional[dict] = None,
        question: Optional[int] = None,
        solution_file: Optional[Path] = None,
        test_output: Optional[Path] = None,
        scope: Optional[str] = None,
        llama_mode: Optional[str] = None,
        json_schema: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Generate a response based on the given prompt and assignment context.

        Args:
            prompt (str): The user's prompt to feed into the model.
            submission_file (Path): Path to the submission file.
            solution_file (Path): Path to the solution file.
            test_output (Path): Path to the test output file.
            scope (Optional[str]): The content scope.
            question (Optional[str]): Specific question number to focus on.
            system_instructions (str): instructions for the model
            llama_mode (Optional[str]): Optional mode to invoke llama.cpp in.
            json_schema (Optional[str]): Optional json schema to use.
            model_options (Optional[dict]): The optional model_options to use for generating the response.

        Returns:
            Tuple[str, str]: The full prompt and the generated response from OpenAI.
        """
        if json_schema:
            schema_path = Path(json_schema)
            if not schema_path.exists():
                raise FileNotFoundError(f"JSON schema file not found: {schema_path}")
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
        else:
            schema = None

        response = self._call_openai(prompt, system_instructions, model_options, schema)
        return prompt, response

    def process_image(self, message: Message, args) -> str:
        """Sends a request to OpenAI"""

        images = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encode_image(image.value)}"},
            }
            for image in message.images
        ]
        completion = self.client.chat.completions.create(
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
        return completion.choices[0].message.content

    def _call_openai(
        self, prompt: str, system_instructions: str, model_options: Optional[dict] = None, schema: Optional[dict] = None
    ) -> str:
        """
        Send a prompt to OpenAI's chat completion API and retrieve the generated response.

        Args:
            prompt (str): The fully constructed input prompt including file content.
            schema (Optional[dict]): Optional json schema to use.
            model_options (dict): The hyperparameters to use for generating the response.

        Returns:
            str: The model's response text.
        """
        response_format = None
        if schema:
            response_format = {"type": "json_schema", "json_schema": schema}

        model_options = cast_to_type(openai_chat_option_schema, model_options)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": prompt},
            ],
            response_format=response_format,
            **model_options,
        )

        return response.choices[0].message.content
