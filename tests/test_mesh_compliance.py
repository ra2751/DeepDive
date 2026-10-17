"""T01–T03: every AI call goes through the Mesh gateway (TD-20)."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", "venv", ".git", "tests", "gateway", "__pycache__", "docs"}

FORBIDDEN_MODULES = {
    "openai", "langchain_openai", "langchain_google_genai", "langchain_anthropic", "langchain_groq",
    "langchain_huggingface", "langchain_mistralai", "langchain_cohere", "sentence_transformers",
    "google.generativeai", "google.genai", "anthropic", "groq", "cohere", "mistralai",
}
FORBIDDEN_NAMES = {"init_chat_model", "ChatOpenAI", "OpenAIEmbeddings", "AzureChatOpenAI",
                   "ChatGoogleGenerativeAI", "ChatAnthropic"}
FORBIDDEN_PACKAGES = {
    "openai", "langchain-google-genai", "langchain-anthropic", "langchain-groq", "langchain-huggingface",
    "langchain_huggingface", "langchain-mistralai", "langchain-cohere", "sentence-transformers",
    "google-generativeai", "google-genai", "anthropic", "groq", "cohere", "mistralai",
}


def _project_files():
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if rel.parts and rel.parts[0] in SKIP_DIRS:
            continue
        yield rel, path


def test_no_direct_provider_imports_or_clients_outside_gateway():
    violations = []
    for rel, path in _project_files():
        tree = ast.parse(path.read_text(), filename=str(rel))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
                for a in node.names:
                    if a.name in FORBIDDEN_NAMES:
                        violations.append(f"{rel}:{node.lineno} imports {a.name}")
            else:
                if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                    violations.append(f"{rel}:{node.lineno} uses {node.id}")
                if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
                    violations.append(f"{rel}:{node.lineno} uses .{node.attr}")
                continue
            for m in mods:
                if any(m == f or m.startswith(f + ".") for f in FORBIDDEN_MODULES):
                    violations.append(f"{rel}:{node.lineno} imports {m}")
    assert not violations, "Direct inference outside gateway/:\n" + "\n".join(violations)


def test_requirements_contain_no_other_inference_sdks():
    names = set()
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(line.split("==")[0].split("[")[0].strip().lower())
    assert not (names & FORBIDDEN_PACKAGES), names & FORBIDDEN_PACKAGES
    assert "langchain-openai" in names  # used only by gateway/ as the Mesh client


def test_gateway_clients_target_mesh(settings):
    from gateway.mesh import _make_chat_client, _make_embeddings_client

    chat = _make_chat_client(settings.mesh_model_generator, settings)
    assert chat.openai_api_base == settings.mesh_base_url == "https://api.meshapi.ai/v1"
    assert chat.model_name == settings.mesh_model_generator
    assert chat.max_retries == 0  # every retry is counted by the gateway loop instead
    emb = _make_embeddings_client(settings)
    assert emb.openai_api_base == "https://api.meshapi.ai/v1"
    assert emb.model == "openai/text-embedding-3-small"
    assert emb.check_embedding_ctx_length is False
    assert emb.chunk_size == settings.embedding_batch_size
