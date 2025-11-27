import json
from pathlib import Path
from typing import Optional, Tuple

import ollama

from ai_feedback.helpers.model_options_helpers import cast_to_type, ollama_option_schema

from .Model import Model


class CodeLlamaModel(Model):

    def __init__(self, model_name: str = None) -> None:
        """
        Initializes the CodeLlamaModel with configuration for the model and system instructions.
        """
        super().__init__(model_name)
        self.model_name = model_name if model_name else "codellama:latest"

    def generate_response(
        self,
        prompt: str,
        submission_file: Path,
        system_instructions: str,
        model_options: Optional[dict] = None,
        question: Optional[str] = None,
        solution_file: Optional[Path] = None,
        test_output: Optional[Path] = None,
        scope: Optional[str] = None,
        llama_mode: Optional[str] = None,
        json_schema: Optional[str] = None,
    ) -> Optional[Tuple[str, str]]:
        """
        Generates a response from the CodeLlama model using the provided prompt
        and assignment file contents.

        Args:
            prompt (str): The user's prompt to be fed into the model.
            submission_file (Optional[Path]): The path to the submission file.
            solution_file (Optional[Path]): The path to the solution file.
            test_output (Optional[Path]): The path to the test output file.
            scope (Optional[str]): The scope to use for generating the response.
            question (Optional[str]): An optional specific question to extract content for.
            system_instructions (str): instructions for the model
            llama_mode (Optional[str]): Optional mode to invoke llama.cpp in.
            json_schema (Optional[str]): Optional json schema to use.
            model_options (Optional[dict]): The optional model options to use for generating the response.

        Returns:
            Optional[Tuple[str, str]]: A tuple of the request and the model's response,
                                       or None if no valid response is returned.
        """
        if json_schema:
            schema_path = Path(json_schema)
            if not schema_path.exists():
                raise FileNotFoundError(f"JSON schema file not found: {schema_path}")
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
        else:
            schema = None

        model_options = cast_to_type(ollama_option_schema, model_options)

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": prompt},
            ],
            format=schema['schema'] if schema else None,
            options=model_options if model_options else None,
        )

        if not response or "message" not in response or "content" not in response["message"]:
            print("Error: Invalid or empty response from Ollama.")
            return None

        return prompt, response["message"]["content"]
