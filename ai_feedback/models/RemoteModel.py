import urllib.request
import json
import os
import sys
import re
from pathlib import Path
from typing import List, Optional, Tuple
from .Model import Model
from dotenv import load_dotenv


class RemoteModel(Model):
    """A class representing a remote model for generating responses.

    Currently is tied to ai_server (https://github.com/MarkUsProject/ai-server).
    """
    remote_url: str
    model_name: str

    def __init__(self, remote_url: str = "http://polymouth.teach.cs.toronto.edu:5000/chat", model_name: str = "deepseek-coder-v2:latest") -> None:
        """Initializes the remote model with a remote URL and model name.
        """
        self.remote_url = remote_url
        self.model_name = model_name

    def generate_response(
        self,
        prompt: str,
        submission_file: Path,
        system_instructions: str,
        solution_file: Optional[Path] = None,
        question_num: Optional[int] = None,
        test_output:Optional[Path] = None,
        scope: Optional[str] = None,
        llama_mode: Optional[str] = None,
    ) -> Optional[Tuple[str, str]]:
        """
        Generate a model response using the prompt and assignment files.

        Args:
            prompt (str): The input prompt provided by the user.
            submission_file (Optional[Path]): The path to the submission file.
            solution_file (Optional[Path]): The path to the solution file.
            question_num (Optional[int]): An optional question number to target specific content.
            test_output (Optional[Path]): The path to the test output file.
            scope (Optional[str]): The scope to use for generating the response.
            system_instructions (str): instructions for the model
            llama_mode (Optional[str]): Optional mode to invoke llama.cpp in.

        Returns:
            Optional[Tuple[str, str]]: A tuple containing the prompt and the model's response,
                                       or None if the response was invalid.
        """
        request = f"Prompt: {prompt}"
        load_dotenv()

        headers = {
            "X-API-KEY": os.getenv("REMOTE_API_KEY"),
            "Content-Type": "application/json"
        }
        data = {
            "content": request,
            "model": self.model_name,
            "system_instructions": system_instructions
        }

        # Convert the data to JSON format
        json_data = json.dumps(data).encode("utf-8")

        # Create the request
        request = urllib.request.Request(self.remote_url, data=json_data, headers=headers, method="POST")

        # Send the request and get the response
        with urllib.request.urlopen(request) as response:
            # Print the status code and response data
            print(response.status)
            response = json.loads(response.read().decode())

        return request, response
