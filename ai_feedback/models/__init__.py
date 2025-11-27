from typing import Dict, Type

from .ClaudeModel import ClaudeModel
from .CodeLlamaModel import CodeLlamaModel
from .DeepSeekModelModified import DeepSeekModelModified
from .Model import Model
from .OpenAIModel import OpenAIModel
from .OpenAIModelVector import OpenAIModelVector
from .RemoteModel import RemoteModel


class ModelFactory:
    """Factory for creating AI model instances with proper dependency injection."""

    _registry: Dict[str, Type[Model]] = {
        "remote": RemoteModel,
        "claude": ClaudeModel,
        "openai": OpenAIModel,
        "codellama": CodeLlamaModel,
        "deepseek": DeepSeekModelModified,
        "openai-vector": OpenAIModelVector,
    }

    @classmethod
    def create(cls, provider: str, **kwargs) -> Model:
        """
        Create a model instance with dependency injection.

        Args:
            provider: The model provider identifier
            **kwargs: Additional arguments to pass to the model constructor

        Returns:
            Model: An instance of the requested model

        Raises:
            ValueError: If the provider is not registered
        """
        model_class = cls.get_model_class(provider)
        return model_class(**kwargs)

    @classmethod
    def get_model_class(cls, provider: str) -> Type[Model]:
        """
        Get the model class for a provider without instantiating it.

        Args:
            provider: The model provider identifier

        Returns:
            Type[Model]: The model class

        Raises:
            ValueError: If the provider is not registered
        """
        if provider not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown model provider '{provider}'. " f"Available providers: {available}")
        return cls._registry[provider]

    @classmethod
    def register(cls, provider: str, model_class: Type[Model]) -> None:
        """
        Register a new model provider (for extensibility).

        Args:
            provider: The model provider identifier
            model_class: The model class to register
        """
        cls._registry[provider] = model_class

    @classmethod
    def is_registered(cls, provider: str) -> bool:
        """Check if a provider is registered."""
        return provider in cls._registry

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of all available provider names."""
        return sorted(cls._registry.keys())
