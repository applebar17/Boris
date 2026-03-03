# boris/tests/test_provider_adapters.py
from __future__ import annotations

import logging
import types

import pytest

from boris.boriscore.ai_clients.protocols.protocol_chat import ChatResponse, Msg
from boris.boriscore.ai_clients.providers.base import ProviderConfig
from boris.boriscore.ai_clients.providers.openai import (
    azure_openai_adapter as azure_adapter_mod,
)
from boris.boriscore.ai_clients.providers.openai import openai_adapter as openai_mod


def _logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def test_openai_make_client_requires_key():
    adapter = openai_mod.OpenAIAdapter(logger=_logger("test.openai.make_client.err"))
    with pytest.raises(ValueError):
        adapter.make_client(ProviderConfig(provider="openai", openai_api_key=None))


def test_openai_make_client_sets_clients(monkeypatch):
    captured = {}

    class DummyOpenAI:
        def __init__(self, *, api_key, base_url):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr(openai_mod, "OpenAI", DummyOpenAI)

    adapter = openai_mod.OpenAIAdapter(logger=_logger("test.openai.make_client.ok"))
    out = adapter.make_client(
        ProviderConfig(
            provider="openai",
            openai_api_key="sk-test",
            openai_base_url="https://example.local/v1",
        )
    )

    assert out is adapter.client
    assert adapter.openai_embeddings_client is adapter.client
    assert captured["api_key"] == "sk-test"
    assert captured["base_url"] == "https://example.local/v1"


def test_openai_chat_uses_parse_for_structured(monkeypatch):
    class ParseEndpoint:
        called = None

        def parse(self, **kwargs):
            self.called = kwargs
            return "parsed"

    class CreateEndpoint:
        called = None

        def create(self, **kwargs):
            self.called = kwargs
            return "created"

    parse_ep = ParseEndpoint()
    create_ep = CreateEndpoint()
    client = types.SimpleNamespace(
        beta=types.SimpleNamespace(chat=types.SimpleNamespace(completions=parse_ep)),
        chat=types.SimpleNamespace(completions=create_ep),
    )

    payload = {"model": "gpt-test", "response_format": {"type": "json_schema"}}
    expected = ChatResponse(message=Msg(role="assistant", content="ok"))

    monkeypatch.setattr(openai_mod, "_build_openai_payload", lambda _req: payload)
    monkeypatch.setattr(
        openai_mod, "_wants_structured_output", lambda _rf: True
    )
    monkeypatch.setattr(openai_mod, "_from_openai_response", lambda _resp: expected)

    adapter = openai_mod.OpenAIAdapter(logger=_logger("test.openai.chat.parse"))
    adapter.client = client
    out = adapter.chat(req=types.SimpleNamespace())

    assert out is expected
    assert parse_ep.called == payload
    assert create_ep.called is None


def test_openai_chat_uses_create_for_plain(monkeypatch):
    class ParseEndpoint:
        called = None

        def parse(self, **kwargs):
            self.called = kwargs
            return "parsed"

    class CreateEndpoint:
        called = None

        def create(self, **kwargs):
            self.called = kwargs
            return "created"

    parse_ep = ParseEndpoint()
    create_ep = CreateEndpoint()
    client = types.SimpleNamespace(
        beta=types.SimpleNamespace(chat=types.SimpleNamespace(completions=parse_ep)),
        chat=types.SimpleNamespace(completions=create_ep),
    )

    payload = {"model": "gpt-test", "messages": []}
    expected = ChatResponse(message=Msg(role="assistant", content="ok"))

    monkeypatch.setattr(openai_mod, "_build_openai_payload", lambda _req: payload)
    monkeypatch.setattr(
        openai_mod, "_wants_structured_output", lambda _rf: False
    )
    monkeypatch.setattr(openai_mod, "_from_openai_response", lambda _resp: expected)

    adapter = openai_mod.OpenAIAdapter(logger=_logger("test.openai.chat.create"))
    adapter.client = client
    out = adapter.chat(req=types.SimpleNamespace())

    assert out is expected
    assert create_ep.called == payload
    assert parse_ep.called is None


def test_openai_get_embeddings_dimensions_policy():
    class EmbeddingClient:
        def __init__(self):
            self.calls = []
            self.embeddings = types.SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return {"ok": True, "kwargs": kwargs}

    emb_client = EmbeddingClient()
    adapter = openai_mod.OpenAIAdapter(logger=_logger("test.openai.embeddings"))
    adapter.openai_embeddings_client = emb_client

    adapter.embedding_model = "text-embedding-3-small"
    adapter.get_embeddings("hello", dimensions=256)
    assert emb_client.calls[-1]["dimensions"] == 256

    adapter.embedding_model = "text-embedding-ada-002"
    adapter.get_embeddings("hello", dimensions=999)
    assert "dimensions" not in emb_client.calls[-1]


def test_azure_make_client_requires_all_fields():
    adapter = azure_adapter_mod.AzureOpenAIAdapter(logger=_logger("test.azure.missing"))
    with pytest.raises(ValueError):
        adapter.make_client(
            ProviderConfig(
                provider="azure",
                azure_openai_endpoint="https://example.azure.com/",
                azure_openai_api_key="k",
                azure_openai_api_version=None,
            )
        )


def test_azure_make_client_sets_and_returns_client(monkeypatch):
    class DummyAzureOpenAI:
        def __init__(self, *, azure_endpoint, api_key, api_version):
            self.azure_endpoint = azure_endpoint
            self.api_key = api_key
            self.api_version = api_version

    monkeypatch.setattr(azure_adapter_mod, "AzureOpenAI", DummyAzureOpenAI)
    monkeypatch.setattr(azure_adapter_mod, "wrap_openai", None)

    adapter = azure_adapter_mod.AzureOpenAIAdapter(logger=_logger("test.azure.ok"))
    out = adapter.make_client(
        ProviderConfig(
            provider="azure",
            azure_openai_endpoint="https://example.azure.com/",
            azure_openai_api_key="k",
            azure_openai_api_version="2025-01-01-preview",
            tracing_enabled=False,
        )
    )

    assert out is adapter.client
    assert adapter.openai_embeddings_client is adapter.client
    assert out.azure_endpoint == "https://example.azure.com/"


def test_azure_make_client_wraps_when_tracing(monkeypatch):
    class DummyAzureOpenAI:
        def __init__(self, **_kwargs):
            self.kind = "raw"

    wrapped = types.SimpleNamespace(kind="wrapped")
    monkeypatch.setattr(azure_adapter_mod, "AzureOpenAI", DummyAzureOpenAI)
    monkeypatch.setattr(azure_adapter_mod, "wrap_openai", lambda _client: wrapped)

    adapter = azure_adapter_mod.AzureOpenAIAdapter(
        logger=_logger("test.azure.wrap")
    )
    out = adapter.make_client(
        ProviderConfig(
            provider="azure",
            azure_openai_endpoint="https://example.azure.com/",
            azure_openai_api_key="k",
            azure_openai_api_version="2025-01-01-preview",
            tracing_enabled=True,
        )
    )

    assert out is wrapped
    assert adapter.client is wrapped
    assert adapter.openai_embeddings_client is wrapped


def test_anthropic_normalize_payload_passthrough():
    anthropic_mod = pytest.importorskip(
        "boris.boriscore.ai_clients.providers.anthropic.anthropic_adapter"
    )
    adapter = anthropic_mod.AnthropicAdapter(logger=_logger("test.anthropic.pass"))
    adapter.client = types.SimpleNamespace(
        messages=types.SimpleNamespace(create=lambda **kwargs: kwargs)
    )

    payload = {"model": "claude", "messages": []}
    out = adapter._normalize_payload_for_messages_create(dict(payload))
    assert out == payload


def test_anthropic_normalize_payload_to_extra_body():
    anthropic_mod = pytest.importorskip(
        "boris.boriscore.ai_clients.providers.anthropic.anthropic_adapter"
    )

    class Messages:
        def create(self, *, extra_body=None, **kwargs):
            return {"extra_body": extra_body, "kwargs": kwargs}

    adapter = anthropic_mod.AnthropicAdapter(
        logger=_logger("test.anthropic.extra_body")
    )
    adapter.client = types.SimpleNamespace(messages=Messages())
    payload = {
        "model": "claude",
        "messages": [],
        "output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}},
    }

    out = adapter._normalize_payload_for_messages_create(dict(payload))
    assert "output_config" not in out
    assert out["extra_body"]["output_config"]["format"]["type"] == "json_schema"


def test_anthropic_normalize_payload_raises_when_no_supported_field():
    anthropic_mod = pytest.importorskip(
        "boris.boriscore.ai_clients.providers.anthropic.anthropic_adapter"
    )

    class Messages:
        def create(self, *, temperature=None):
            return {"temperature": temperature}

    adapter = anthropic_mod.AnthropicAdapter(
        logger=_logger("test.anthropic.no_support")
    )
    adapter.client = types.SimpleNamespace(messages=Messages())
    payload = {
        "model": "claude",
        "messages": [],
        "output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}},
    }

    with pytest.raises(RuntimeError):
        adapter._normalize_payload_for_messages_create(dict(payload))


def test_anthropic_chat_parses_structured_json(monkeypatch):
    anthropic_mod = pytest.importorskip(
        "boris.boriscore.ai_clients.providers.anthropic.anthropic_adapter"
    )

    class Messages:
        called = None

        def create(self, *, extra_body=None, **kwargs):
            self.called = {"extra_body": extra_body, **kwargs}
            return "raw-response"

    messages = Messages()
    adapter = anthropic_mod.AnthropicAdapter(logger=_logger("test.anthropic.chat"))
    adapter.client = types.SimpleNamespace(messages=messages)

    payload = {
        "model": "claude",
        "max_tokens": 64,
        "messages": [],
        "output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}},
    }
    proto = ChatResponse(message=Msg(role="assistant", content='{"ok": true}'), tool_calls=[])

    class DummyModel:
        pass

    parsed_obj = {"validated": True}

    monkeypatch.setattr(
        anthropic_mod,
        "response_format_to_schema",
        lambda _rf: ({"type": "object"}, DummyModel),
    )
    monkeypatch.setattr(anthropic_mod, "build_anthropic_payload", lambda _req: payload)
    monkeypatch.setattr(anthropic_mod, "from_anthropic_response", lambda _resp: proto)
    monkeypatch.setattr(
        anthropic_mod, "_pydantic_validate", lambda _cls, _data: parsed_obj
    )

    req = types.SimpleNamespace(params={"response_format": DummyModel})
    out = adapter.chat(req)

    assert out.message.content == parsed_obj
    assert messages.called["extra_body"]["output_config"]["format"]["type"] == "json_schema"


def test_anthropic_chat_skips_parse_when_tool_calls(monkeypatch):
    anthropic_mod = pytest.importorskip(
        "boris.boriscore.ai_clients.providers.anthropic.anthropic_adapter"
    )

    class Messages:
        def create(self, *, extra_body=None, **kwargs):
            return "raw-response"

    adapter = anthropic_mod.AnthropicAdapter(
        logger=_logger("test.anthropic.chat.skip")
    )
    adapter.client = types.SimpleNamespace(messages=Messages())

    payload = {
        "model": "claude",
        "max_tokens": 64,
        "messages": [],
        "output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}},
    }
    proto = ChatResponse(
        message=Msg(role="assistant", content="not-json"),
        tool_calls=[types.SimpleNamespace(function=types.SimpleNamespace(name="tool"))],
    )

    monkeypatch.setattr(
        anthropic_mod,
        "response_format_to_schema",
        lambda _rf: ({"type": "object"}, object),
    )
    monkeypatch.setattr(anthropic_mod, "build_anthropic_payload", lambda _req: payload)
    monkeypatch.setattr(anthropic_mod, "from_anthropic_response", lambda _resp: proto)
    monkeypatch.setattr(
        anthropic_mod,
        "_pydantic_validate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("validate should not run when tool_calls are present")
        ),
    )

    req = types.SimpleNamespace(params={"response_format": object})
    out = adapter.chat(req)
    assert out.message.content == "not-json"


def test_anthropic_chat_does_not_log_payload_body(monkeypatch):
    anthropic_mod = pytest.importorskip(
        "boris.boriscore.ai_clients.providers.anthropic.anthropic_adapter"
    )

    class Messages:
        def create(self, **_kwargs):
            return "raw-response"

    adapter = anthropic_mod.AnthropicAdapter(
        logger=_logger("test.anthropic.chat.logging")
    )
    adapter.client = types.SimpleNamespace(messages=Messages())

    logs = []

    def _capture_log(msg: str, log_type: str = "info"):
        logs.append((msg, log_type))

    payload = {
        "model": "claude",
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "top-secret prompt"}],
            }
        ],
    }
    proto = ChatResponse(message=Msg(role="assistant", content="ok"), tool_calls=[])

    monkeypatch.setattr(adapter, "_log", _capture_log)
    monkeypatch.setattr(anthropic_mod, "build_anthropic_payload", lambda _req: payload)
    monkeypatch.setattr(anthropic_mod, "from_anthropic_response", lambda _resp: proto)

    req = types.SimpleNamespace(params={})
    out = adapter.chat(req)

    assert out is proto
    log_text = "\n".join(msg for msg, _ in logs)
    assert "Payload body logging disabled." in log_text
    assert "top-secret prompt" not in log_text
    assert '"messages"' not in log_text
