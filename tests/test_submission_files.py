"""Submission readiness: official workflow present and unmodified; project files in place; no secrets."""

import hashlib
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_official_workflow_is_present_and_unmodified():
    wf = ROOT / ".github/workflows/agent-arena-2026-checks.yml"
    recorded = (ROOT / "docs/submission/official-workflow.sha256").read_text().split()[0]
    assert hashlib.sha256(wf.read_bytes()).hexdigest() == recorded


def test_required_submission_files_exist():
    for f in ("README.md", "requirements.txt", ".gitignore", ".env.example", ".github/workflows/ci.yml"):
        assert (ROOT / f).exists(), f


def test_env_is_ignored_and_example_has_no_values_for_secrets():
    assert ".env" in (ROOT / ".gitignore").read_text()
    example = (ROOT / ".env.example").read_text()
    for name in ("MESH_API_KEY", "TAVILY_API_KEY", "MILVUS_API_KEY"):
        assert re.search(rf"^{name}=\s*$", example, re.M), name


def test_no_secret_like_strings_in_tracked_or_new_files():
    git = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True)
    if git.returncode == 0:
        files = git.stdout.split()
    else:   # no git metadata (e.g. an exported copy): scan everything except the git-ignored locations
        skip_dirs = {".git", ".venv", "venv", "data", "logs", "uploads", "__pycache__", ".pytest_cache", "outputs"}
        files = [str(f.relative_to(ROOT)) for f in ROOT.rglob("*")
                 if f.is_file() and not (set(f.relative_to(ROOT).parts) & skip_dirs)
                 and (not f.name.startswith(".env") or f.name == ".env.example")]
    pattern = re.compile(r"(rsk_[A-Za-z0-9]{16,}|tvly-[A-Za-z0-9]{16,}|sk-[A-Za-z0-9]{20,})")
    hits = []
    for f in files:
        p = ROOT / f
        if p.suffix in {".py", ".md", ".txt", ".yml", ".yaml", ".json", ".jsonl", ".ini", ".example", ""} and p.is_file():
            if pattern.search(p.read_text(errors="ignore")):
                hits.append(f)
    assert not hits, hits
