"""Adapter registry. Built-ins are imported lazily so optional dependencies stay optional."""

from __future__ import annotations

import importlib
from typing import Any, Callable

from ..errors import ConfigError
from .base import BaseAdapter

_REGISTRY: dict[str, type[BaseAdapter]] = {}
_BUILTIN = {
    "yolov8": "gccvt.adapters.yolov8:YoloV8Adapter",
    "insightface": "gccvt.adapters.insightface_det:InsightFaceAdapter",
}


def register(name: str) -> Callable[[type[BaseAdapter]], type[BaseAdapter]]:
    def deco(cls: type[BaseAdapter]) -> type[BaseAdapter]:
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


def _import(path: str) -> type[BaseAdapter]:
    module, _, attr = path.partition(":")
    if not attr:
        raise ConfigError(f"adapter path must look like 'package.module:ClassName', got {path!r}")
    try:
        cls = getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError) as e:
        raise ConfigError(f"cannot import adapter {path!r}: {e}") from e
    return cls


def create_adapter(name: str, params: dict[str, Any] | None = None) -> BaseAdapter:
    """Instantiate and load an adapter by registered name or 'module:Class' path."""
    if name in _REGISTRY:
        cls = _REGISTRY[name]
    elif name in _BUILTIN:
        cls = _import(_BUILTIN[name])
    elif ":" in name:
        cls = _import(name)
    else:
        known = sorted(set(_REGISTRY) | set(_BUILTIN))
        raise ConfigError(f"unknown adapter {name!r}; known: {known}, or use 'package.module:ClassName'")
    adapter = cls()
    adapter.load(**(params or {}))
    return adapter
