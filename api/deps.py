# api/deps.py
from __future__ import annotations

from functools import lru_cache

from internalpy.config import Cfg, build_cfg


@lru_cache(maxsize=1)
def _load_cfg() -> Cfg:
    return build_cfg()


def get_cfg() -> Cfg:
    """
    FastAPI dependency to access the global, immutable configuration.
    """
    return _load_cfg()
