import os
from pathlib import Path
from typing import List, Optional

import openai
from dotenv import load_dotenv

from .Model import Model

# Load environment variables from .env file
load_dotenv()


class OpenAIModelVector(Model):
    """
    A model that uses OpenAI's GPT-4-turbo with vector search support for document-augmented responses.
    This class handles uploading files to OpenAI's vector store and cleaning up associated resources
    after processing. Files are not directly uploaded as part of prompt.
    """

    def __init__(self) -> None:
        """
        Creates a new OpenAI vector store and an assistant model that can use file search as a tool.
        """
        super().__init__()

        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.vector_store = self.client.vector_stores.create(name="Markus LLM Vector Store")
        self.model = self.client.beta.assistants.create(
            name="Markus LLM model",
            model="gpt-4-turbo",
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [self.vector_store.id]}},
        )

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
    ) -> tuple[str, str]:
        """
        Generate a response from the OpenAI model using the provided prompt and assignment files.

        Args:
            prompt (str): The user's prompt to feed into the model.
            submission_file (Optional[Path]): The path to a file to store the response to.
            solution_file (Optional[Path]): The path to a file to store the response to.
            test_output (Optional[Path]): The path to a file to store the response to.
            scope (Optional[str]): The path to a file to store the response to.
            question_num (Optional[int]): An optional question number.
            system_instructions (str): instructions for the model
            llama_mode (Optional[str]): Optional mode to invoke llama.cpp in.

        Returns:
            tuple[str, str]: A tuple containing the full system request and the model's text response.
        """
        self.model = self.client.beta.assistants.update(assistant_id=self.model.id, instructions=system_instructions)
        if not self.model:
            raise RuntimeError("Model was not created successfully.")

        request = "Uploaded Files: "
        file_ids: List[str] = []
        assignment_files = [f for f in (submission_file, solution_file, test_output) if f]

        for file_path in assignment_files:
            base, ext = file_path.stem, file_path.suffix
            file_id = self._upload_file(file_path)
            file_ids.append(file_id)
            request += base

        if question_num:
            prompt += f" Identify and generate a response for the mistakes **only** in task ${question_num}. "

        response = self._call_openai(prompt)
        self._cleanup_resources(file_ids)

        request = f"\n{system_instructions}\n{prompt}"
        return request, response

    def _upload_file(self, file_path: Path) -> str:
        """
        Upload a file to OpenAI storage and link it to the vector store.

        Args:
            file_path (str): Local path of the file to be uploaded.

        Returns:
            str: The ID of the uploaded file.
        """
        with open(file_path, "rb") as f:
            response = self.client.files.create(file=f, purpose="assistants")
            self.client.vector_stores.files.create(vector_store_id=self.vector_store.id, file_id=response.id)
        return response.id

    def _call_openai(self, prompt: str) -> str:
        """
        Send the user prompt to OpenAI's assistant model and retrieve the generated response.

        Args:
            prompt (str): The input prompt for the assistant.

        Returns:
            str: The assistant's generated response text.
        """
        thread = self.client.beta.threads.create()

        self.client.beta.threads.messages.create(thread_id=thread.id, role="user", content=prompt)

        run = self.client.beta.threads.runs.create(thread_id=thread.id, assistant_id=self.model.id)

        while run.status not in ["completed", "failed"]:
            run = self.client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

        if run.status == "failed":
            print("Error details:", run.last_error)

        messages = self.client.beta.threads.messages.list(thread_id=thread.id)
        return messages.data[0].content[0].text.value

    def _cleanup_resources(self, file_ids: List[str]) -> None:
        """
        Clean up uploaded files and delete all created assistant models.

        Args:
            file_ids (List[str]): List of uploaded file IDs to delete.
        """
        self._cleanup_files(file_ids)
        self._delete_all_models()

    def _cleanup_files(self, file_ids: List[str]) -> None:
        """
        Delete uploaded files from OpenAI storage and vector store.

        Args:
            file_ids (List[str]): List of uploaded file IDs to remove.
        """
        for file_id in file_ids:
            self.client.files.delete(file_id)
            self.client.vector_stores.files.delete(vector_store_id=self.vector_store.id, file_id=file_id)

    def _delete_all_models(self) -> None:
        """
        Delete all assistant models created under this OpenAI account.
        """
        models = self.client.beta.assistants.list().data
        for model in models:
            self.client.beta.assistants.delete(model.id)

    def _delete_all_files(self) -> None:
        """
        Delete all files from OpenAI storage and the vector store.
        """
        files = self.client.files.list()
        for file in files:
            self.client.files.delete(file.id)
            self.client.vector_stores.files.delete(vector_store_id=self.vector_store.id, file_id=file.id)
