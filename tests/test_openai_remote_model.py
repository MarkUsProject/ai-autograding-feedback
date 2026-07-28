"""Tests for OpenAIRemoteModel — the LiteLLM-gateway-backed OpenAI model.

These tests never touch the network. They patch ``openai.OpenAI`` with a fake
client that records construction kwargs and returns a canned chat completion,
so we can assert the gateway endpoint, auth, and the attribution header without
a running proxy.
"""

import json
import types

import httpx
import openai
import pytest
from ollama import Message

from ai_feedback.models import GatewayError, ModelFactory, OpenAIRemoteModel

METADATA = {
    "instance": "markus.cs.toronto.edu",
    "course_id": 12,
    "assignment_id": 34,
    "group_id": 56,
    "batch_id": None,
    "category": "student",
}


class _FakeCompletions:
    def __init__(self, content):
        self._content = content
        self.calls = []
        self.error = None  # set to make create() fail the way the OpenAI SDK does

    def create(self, **kwargs):
        if self.error:
            raise self.error
        self.calls.append(kwargs)
        message = types.SimpleNamespace(content=self._content)
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])


class _FakeClient:
    """Stand-in for openai.OpenAI that records how it was built."""

    last = None

    def __init__(self, *, base_url=None, api_key=None, default_headers=None, content="feedback"):
        self.base_url = base_url
        self.api_key = api_key
        self.default_headers = default_headers or {}
        self.completions = _FakeCompletions(content)
        self.chat = types.SimpleNamespace(completions=self.completions)
        _FakeClient.last = self


def _status_error(body):
    """A real openai.APIStatusError carrying ``body``, as the SDK would raise it."""
    request = httpx.Request("POST", "http://gateway:4000/v1/chat/completions")
    return openai.APIStatusError("Error code: 400", response=httpx.Response(400, request=request), body=body)


@pytest.fixture
def fake_openai(monkeypatch):
    monkeypatch.setattr(openai, "OpenAI", _FakeClient)
    return _FakeClient


@pytest.fixture(autouse=True)
def gateway_env(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test-virtual-key")
    monkeypatch.delenv("LITELLM_SPEND_METADATA", raising=False)


def test_provider_is_registered():
    assert ModelFactory.is_registered("openai-remote")
    assert ModelFactory.get_model_class("openai-remote") is OpenAIRemoteModel


def test_client_targets_gateway_with_bearer_auth(fake_openai):
    OpenAIRemoteModel(remote_url="http://gateway:4000/v1", model_name="gpt-4o-mini")
    assert fake_openai.last.base_url == "http://gateway:4000/v1"
    assert fake_openai.last.api_key == "sk-test-virtual-key"


def test_attaches_metadata_header_when_set(monkeypatch, fake_openai):
    monkeypatch.setenv("LITELLM_SPEND_METADATA", json.dumps(METADATA))
    model = OpenAIRemoteModel()
    header = fake_openai.last.default_headers[OpenAIRemoteModel.METADATA_HEADER]
    assert json.loads(header) == METADATA
    assert model.spend_logs_metadata == header


def test_no_metadata_header_when_unset(fake_openai):
    model = OpenAIRemoteModel()
    assert OpenAIRemoteModel.METADATA_HEADER not in fake_openai.last.default_headers
    assert model.spend_logs_metadata is None


def test_missing_api_key_fails_loud(monkeypatch, fake_openai):
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LITELLM_API_KEY"):
        OpenAIRemoteModel()


def test_malformed_metadata_fails_loud(monkeypatch, fake_openai):
    monkeypatch.setenv("LITELLM_SPEND_METADATA", "{not valid json")
    with pytest.raises(RuntimeError, match="not valid JSON"):
        OpenAIRemoteModel()


def test_generate_response_speaks_openai_contract(fake_openai):
    model = OpenAIRemoteModel(model_name="gpt-4o-mini")
    prompt, response = model.generate_response(
        prompt="Review this code.",
        submission_file=None,
        system_instructions="You are a TA.",
        model_options={},
    )
    assert response == "feedback"
    assert prompt == "Review this code."
    sent = fake_openai.last.completions.calls[0]
    assert sent["model"] == "gpt-4o-mini"
    roles = [m["role"] for m in sent["messages"]]
    assert roles == ["system", "user"]


def test_max_tokens_defaults_when_caller_omits_it(fake_openai):
    model = OpenAIRemoteModel()
    model.generate_response(
        prompt="Review this code.",
        submission_file=None,
        system_instructions="You are a TA.",
        model_options={},
    )
    sent = fake_openai.last.completions.calls[0]
    assert sent["max_tokens"] == OpenAIRemoteModel.DEFAULT_MAX_TOKENS


def test_caller_max_tokens_wins_over_default(fake_openai):
    model = OpenAIRemoteModel()
    model.generate_response(
        prompt="Review this code.",
        submission_file=None,
        system_instructions="You are a TA.",
        model_options={"max_tokens": 2048},
    )
    sent = fake_openai.last.completions.calls[0]
    assert sent["max_tokens"] == 2048


def _failing_with(sdk_error):
    """Make the gateway raise ``sdk_error``, and return the GatewayError it becomes."""
    model = OpenAIRemoteModel()
    model.client.completions.error = sdk_error
    with pytest.raises(GatewayError) as raised:
        model.generate_response(
            prompt="Review this code.",
            submission_file=None,
            system_instructions="You are a TA.",
            model_options={},
        )
    return raised.value


def test_budget_rejection_surfaces_the_gateway_message(fake_openai):
    """The instructor-facing reason, not a stack trace ending in BadRequestError."""
    reason = "Course budget exhausted for course_id=1: spent CAD 0.01 of CAD 0.01."
    assert str(_failing_with(_status_error({"message": reason}))) == reason


def test_failure_without_our_body_falls_back_to_the_sdk_message(fake_openai):
    """A proxy error page has no 'message' key; we must still say something."""
    assert "Error code: 400" in str(_failing_with(_status_error("<html>502 Bad Gateway</html>")))


def test_unreachable_gateway_is_reported_the_same_way(fake_openai):
    """A gateway restart mid-batch must not print a stack trace either."""
    request = httpx.Request("POST", "http://gateway:4000/v1/chat/completions")
    assert "Connection error" in str(_failing_with(openai.APIConnectionError(request=request)))


def test_image_failure_is_reported_as_a_gateway_error(fake_openai):
    """Image feedback goes through process_image, not _call_openai — same treatment."""
    model = OpenAIRemoteModel()
    model.client.completions.error = _status_error({"message": "Course budget exhausted for course_id=1."})
    with pytest.raises(GatewayError, match="Course budget exhausted"):
        model.process_image(Message(role="user", content="Describe this plot.", images=[]), args=None)


def test_failure_keeps_the_original_error_for_debugging(fake_openai):
    error = _failing_with(_status_error({"message": "Upstream API key is disabled."}))
    assert isinstance(error.__cause__, openai.APIStatusError)
