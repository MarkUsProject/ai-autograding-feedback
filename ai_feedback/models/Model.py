from typing import Any, Tuple

"""
Parent Class for LLMs.

This class serves as an abstract base for defining LLM model interfaces.
Subclasses must implement the `generate_response` method to provide
their own model-specific logic.
"""


class Model:
    def __init__(self):
        """
        Initialize the model.
        """
        pass

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
