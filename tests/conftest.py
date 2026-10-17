import os
import socket
import tempfile

# Isolate logs from the repository's logs/ directory (set before project imports).
os.environ.setdefault("DEEPDIVE_LOG_DIR", tempfile.mkdtemp(prefix="deepdive-test-logs-"))

import pytest  # noqa: E402

from config.settings import load_settings, set_settings_override  # noqa: E402
from tests.fakes import FakeEmbeddingsClient, ScriptedChat  # noqa: E402

TEST_ENV = {
    "MESH_API_KEY": "test-key-not-real",
    "MESH_MODEL_SUPERVISOR": "test/supervisor",
    "MESH_MODEL_GENERATOR": "test/generator",
    "MESH_MODEL_VERIFIER": "test/verifier",
    "TAVILY_API_KEY": "test-tavily-not-real",
    "MILVUS_ENDPOINT": "https://milvus.invalid",
    "MILVUS_API_KEY": "test-milvus-not-real",
}


class NetworkBlocked(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def block_network(request, monkeypatch):
    """Offline tests must never touch the network."""
    if request.node.get_closest_marker("live"):
        return

    def guard(*args, **kwargs):
        raise NetworkBlocked("network access is blocked in offline tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    monkeypatch.setattr(socket, "getaddrinfo", guard)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    import gateway.mesh
    import retriever.web_search

    monkeypatch.setattr(gateway.mesh, "_sleep", lambda s: None)
    monkeypatch.setattr(retriever.web_search, "_sleep", lambda s: None)


def make_settings(**overrides):
    env = dict(TEST_ENV)
    env.update({k: str(v) for k, v in overrides.items()})
    return load_settings(environ=env)


@pytest.fixture
def settings():
    s = make_settings()
    set_settings_override(s)
    yield s
    set_settings_override(None)


@pytest.fixture
def use_settings():
    """Factory: install settings with overrides for the duration of a test."""

    def _use(**overrides):
        s = make_settings(**overrides)
        set_settings_override(s)
        return s

    yield _use
    set_settings_override(None)


@pytest.fixture
def fake_mesh(monkeypatch, settings):
    """Route the gateway's client factories to scripted fakes (one per role model)."""
    import gateway.mesh

    clients = {
        settings.mesh_model_supervisor: ScriptedChat(settings.mesh_model_supervisor),
        settings.mesh_model_generator: ScriptedChat(settings.mesh_model_generator),
        settings.mesh_model_verifier: ScriptedChat(settings.mesh_model_verifier),
    }
    emb = FakeEmbeddingsClient()

    def make_chat(model, s, temperature=None):
        return clients.setdefault(model, ScriptedChat(model))

    monkeypatch.setattr(gateway.mesh, "_make_chat_client", make_chat)
    monkeypatch.setattr(gateway.mesh, "_make_embeddings_client", lambda s: emb)

    class Handle:
        supervisor = clients[settings.mesh_model_supervisor]
        generator = clients[settings.mesh_model_generator]
        verifier = clients[settings.mesh_model_verifier]
        embeddings = emb
        all = clients

        def total_chat_calls(self):
            return sum(c.calls for c in clients.values())

    return Handle()
