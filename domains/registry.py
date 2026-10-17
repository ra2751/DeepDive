"""Loads and validates research-domain configurations from domains/configs/*.yaml."""

from functools import lru_cache
from pathlib import Path

import yaml

from schemas.domain import DomainConfig

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_DOMAIN = "general"


@lru_cache(maxsize=None)
def load_domains() -> dict:
    domains = {}
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg = DomainConfig(**yaml.safe_load(path.read_text()))
        if cfg.id in domains:
            raise ValueError(f"duplicate domain id {cfg.id!r} in {path.name}")
        domains[cfg.id] = cfg
    return domains


def get_domain(domain_id: str = DEFAULT_DOMAIN) -> DomainConfig:
    domains = load_domains()
    if domain_id not in domains:
        raise KeyError(f"Unknown domain {domain_id!r}; available: {sorted(domains)}")
    return domains[domain_id]
