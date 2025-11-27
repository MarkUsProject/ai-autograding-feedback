from typing import Any, Tuple

from ollama import Message, chat

"""
Parent Class for LLMs.

This class serves as an abstract base for defining LLM model interfaces.
Subclasses must implement the `generate_response` method to provide
their own model-specific logic.
"""


class Model:
    def __init__(self, model_name: str = None):
        """
        Initialize the model.
        """
        self.model_name = model_name

    def generate_response(self, prompt: str, **kwargs: Any) -> Tuple[str, str]:
        """
        Generate a response based on the provided prompt.

        This is an abstract method that must be overridden by subclasses to implement
        specific model inference logic.

        Args:
            prompt (str): The input prompt used to generate the response.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[str, str]:
                A tuple containing:
                - The full prompt that was used.
                - The generated response.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError("Subclasses must implement the `generate_response` method.")

    def process_image(self, message: Message, args) -> str | None:
        """
        Process an image-based request and generate a response.

        Args:
            message: The message containing the prompt and images
            args: Command-line arguments containing model configuration

        Returns:
            str | None: The model's response or None if processing fails
        """
        return chat(model=self.model_name, messages=[message], options={"temperature": 0.33}).message.content
