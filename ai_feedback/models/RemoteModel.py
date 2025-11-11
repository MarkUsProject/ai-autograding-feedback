import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import requests
from dotenv import load_dotenv

from .Model import Model


class RemoteModel(Model):
    """A class representing a remote model for generating responses.

    Currently is tied to ai_server (https://github.com/MarkUsProject/ai-server).
    """

    remote_url: str
    model_name: str

    def __init__(
        self,
        remote_url: str = "https://polymouth.teach.cs.toronto.edu:443/chat",
        model_name: str = "gpt-oss:120b",
    ) -> None:
        """Initializes the remote model with a remote URL and model name."""
        self.remote_url = remote_url
        self.model_name = model_name

    def generate_response(
        self,
        prompt: str,
        submission_file: Path,
        system_instructions: str,
        model_options: Optional[dict] = None,
        solution_file: Optional[Path] = None,
        question: Optional[str] = None,
        test_output: Optional[Path] = None,
        scope: Optional[str] = None,
        llama_mode: Optional[str] = None,
        submission_image: Optional[str] = None,
        json_schema: Optional[str] = None,
    ) -> Optional[Tuple[str, str]]:
        """
        Generate a model response using the prompt and assignment files.

        Args:
            prompt (str): The input prompt provided by the user.
            submission_file (Optional[Path]): The path to the submission file.
            solution_file (Optional[Path]): The path to the solution file.
            question (Optional[str]): An optional question to target specific content.
            test_output (Optional[Path]): The path to the test output file.
            scope (Optional[str]): The scope to use for generating the response.
            system_instructions (str): instructions for the model
            llama_mode (Optional[str]): Optional mode to invoke llama.cpp in.
            submission_image (Optional[str]): An optional path to a submission image file.
            json_schema (Optional[str]): An optional json schema to use.
            model_options (Optional[dict]): The optional model options to use for generating the response.

        Returns:
            Optional[Tuple[str, str]]: A tuple containing the prompt and the model's response,
                                       or None if the response was invalid.
        """
        load_dotenv()

        headers = {"X-API-KEY": os.getenv("REMOTE_API_KEY")}
        data = {
            "content": prompt,
            "model": self.model_name,
            "system_instructions": system_instructions,
            "model_options": model_options,
        }
        if json_schema:
            schema_path = Path(json_schema)
            if not schema_path.exists():
                raise FileNotFoundError(f"JSON schema file not found: {schema_path}")
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)

            data["json_schema"] = json.dumps(schema)

        if llama_mode:
            data["llama_mode"] = llama_mode

        files = {}
        if submission_image:
            filename = os.path.basename(submission_image)
            files[filename] = (filename, open(submission_image, 'rb'))

        response = requests.post(self.remote_url, data=data, headers=headers, files=files)

        return prompt, response.json()
