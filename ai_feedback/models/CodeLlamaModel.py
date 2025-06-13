import ollama
from pathlib import Path
from typing import Optional, Tuple

from .Model import Model


class CodeLlamaModel(Model):

    def __init__(self) -> None:
        """
        Initializes the CodeLlamaModel with configuration for the model and system instructions.
        """
        self.model = {
            "model": "codellama:latest",
        }

    def generate_response(
        self,
        prompt: str,
        submission_file: Path,
        system_instructions: str,
        question_num: Optional[int] = None,
        solution_file: Optional[Path] = None,
        test_output: Optional[Path] = None,
        scope: Optional[str] = None,
        llama_mode: Optional[str] = None,
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
            question_num (Optional[int]): An optional specific question number to extract content for.
            system_instructions (str): instructions for the model
            llama_mode (Optional[str]): Optional mode to invoke llama.cpp in.

        Returns:
            Optional[Tuple[str, str]]: A tuple of the request and the model's response,
                                       or None if no valid response is returned.
        """
        request = f"Prompt: {prompt}"

        response = ollama.chat(
            model=self.model["model"],
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": request},
            ],
        )

        if (
            not response
            or "message" not in response
            or "content" not in response["message"]
        ):
            print("Error: Invalid or empty response from Ollama.")
            return None

        return request, response["message"]["content"]
