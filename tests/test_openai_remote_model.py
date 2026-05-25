"""Tests for OpenAIRemoteModel — the LiteLLM-gateway-backed OpenAI model.

These tests never touch the network. They patch ``openai.OpenAI`` with a fake
client that records construction kwargs and returns a canned chat completion,
so we can assert the gateway endpoint, auth, and the attribution header without
a running proxy.
"""

import json
import types

import openai
import pytest

from ai_feedback.models import ModelFactory, OpenAIRemoteModel

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

    def create(self, **kwargs):
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
