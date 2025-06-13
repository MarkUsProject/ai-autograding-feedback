import os
from pathlib import Path
from typing import Optional, Tuple
from dotenv import load_dotenv
import openai
from .Model import Model

load_dotenv()


class OpenAIModel(Model):
    def __init__(self) -> None:
        """
        Initialize an OpenAIModel instance.

        Loads the OpenAI API key from environment variables and prepares the client.
        """
        super().__init__()
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    ) -> Tuple[str, str]:
        """
        Generate a response based on the given prompt and assignment context.

        Args:
            prompt (str): The user's prompt to feed into the model.
            submission_file (Path): Path to the submission file.
            solution_file (Path): Path to the solution file.
            test_output (Path): Path to the test output file.
            scope (Optional[str]): The content scope.
            question_num (Optional[int]): Specific question number to focus on.
            system_instructions (str): instructions for the model
            llama_mode (Optional[str]): Optional mode to invoke llama.cpp in.

        Returns:
            Tuple[str, str]: The full prompt and the generated response from OpenAI.
        """
        request = f"Prompt: {prompt}"
        response = self._call_openai(request, system_instructions)
        return request, response

    def _call_openai(self, prompt: str, system_instructions: str) -> str:
        """
        Send a prompt to OpenAI's chat completion API and retrieve the generated response.

        Args:
            prompt (str): The fully constructed input prompt including file content.

        Returns:
            str: The model's response text.
        """
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            temperature=0.5,
        )
        return response.choices[0].message.content
