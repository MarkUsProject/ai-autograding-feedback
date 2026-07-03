import json
import os
from typing import Optional

import openai

from .OpenAIModel import OpenAIModel


class OpenAIRemoteModel(OpenAIModel):
    """An OpenAI-compatible model served through the MarkUs LiteLLM gateway.

    This is the sibling of :class:`RemoteModel`. ``RemoteModel`` talks to the
    ``markus-ai-server`` proxy ("polymouth") with a custom payload and an
    ``X-API-KEY`` header. ``OpenAIRemoteModel`` instead speaks the standard
    OpenAI chat-completion contract (``Authorization: Bearer`` + OpenAI
    request/response schema), which is what the self-hosted LiteLLM proxy
    expects. The only differences from :class:`OpenAIModel` are the endpoint
    (the LiteLLM gateway rather than ``api.openai.com``) and the per-call
    attribution header.

    Attribution metadata (instance, course_id, assignment_id, group_id,
    batch_id, category) is read from the ``LITELLM_SPEND_METADATA`` environment
    variable and forwarded verbatim as the ``x-litellm-spend-logs-metadata``
    header. The autotester's AI tester sets that variable per invocation. The
    gateway's pre-call hook reads the header to attribute spend to the right
    course and to enforce the gatekeeper budget. See the ai-telemetry-gateway
    project for the receiving side.
    """

    #: Header LiteLLM reads to persist arbitrary metadata on each spend-log row.
    METADATA_HEADER = "x-litellm-spend-logs-metadata"

    #: Reply-size cap sent when the caller does not pick one. The gateway
    #: rejects calls that omit max_tokens, and the autotester exposes no
    #: model-options field, so this default is what keeps a plain config working.
    DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self,
        remote_url: str = "http://localhost:4000/v1",
        model_name: str = "gpt-4o-mini",
    ) -> None:
        """Initialize a client pointed at the LiteLLM gateway.

        Args:
            remote_url: Base URL of the LiteLLM proxy's OpenAI-compatible API
                (the ``/v1`` root). Supplied by the autotester via ``--remote_url``.
            model_name: The model to request, e.g. ``gpt-4o-mini``. Must be one
                of the models the gateway is configured to allow.
        """
        # Bypass OpenAIModel.__init__ on purpose: it builds a client against
        # api.openai.com using OPENAI_API_KEY, which is not how we authenticate
        # to the gateway. We build our own client below.
        super(OpenAIModel, self).__init__(model_name)
        self.client = openai.OpenAI(
            base_url=remote_url,
            api_key=self._require_api_key(),
            default_headers=self._attribution_headers(),
        )

    def _call_openai(
        self, prompt: str, system_instructions: str, model_options: Optional[dict] = None, schema: Optional[dict] = None
    ) -> str:
        """Delegate to OpenAIModel with max_tokens defaulted; the gateway rejects calls without it."""
        model_options = dict(model_options or {})
        model_options.setdefault("max_tokens", self.DEFAULT_MAX_TOKENS)
        return super()._call_openai(prompt, system_instructions, model_options, schema)

    @staticmethod
    def _require_api_key() -> str:
        """The LiteLLM virtual key sent as 'Authorization: Bearer'."""
        api_key = os.getenv("LITELLM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LITELLM_API_KEY is not set. The gateway authenticates callers "
                "with a LiteLLM virtual key sent as 'Authorization: Bearer'."
            )
        return api_key

    @classmethod
    def _attribution_headers(cls) -> dict:
        """The x-litellm-spend-logs-metadata header, or {} when unset.

        Forwarded as-is — the autotester produces the JSON — but we fail loud on
        malformed JSON rather than ship a broken header.
        """
        metadata = os.getenv("LITELLM_SPEND_METADATA")
        if not metadata:
            return {}
        try:
            json.loads(metadata)
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(
                "LITELLM_SPEND_METADATA is not valid JSON; refusing to send a malformed attribution header."
            ) from exc
        return {cls.METADATA_HEADER: metadata}

    @property
    def spend_logs_metadata(self) -> Optional[str]:
        """The attribution header value sent on every call, or None if unset."""
        return self.client.default_headers.get(self.METADATA_HEADER)
