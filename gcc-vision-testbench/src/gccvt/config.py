"""Loads config/*.yaml and .env. All paths and URLs come from here."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


class Settings:
    """Project settings: resolved paths, taxonomy, network and video options."""

    def __init__(self, config_dir: Path | None = None, root: Path | None = None):
        self.root = (root or PROJECT_ROOT).resolve()
        load_dotenv(self.root / ".env")
        env_cfg = os.environ.get("GCCVT_CONFIG_DIR")
        self.config_dir = Path(config_dir or env_cfg or self.root / "config")
        if not self.config_dir.is_absolute():
            self.config_dir = (self.root / self.config_dir).resolve()
        self.data = load_yaml(self.config_dir / "settings.yaml")
        self._paths = self._resolve_paths(self.data.get("paths") or {})

    # ---- paths -------------------------------------------------------------
    def _resolve_paths(self, raw: dict[str, str]) -> dict[str, Path]:
        data_root = os.environ.get("GCCVT_DATA_ROOT") or raw.get("data_root") or "data"
        out: dict[str, Path] = {"data_root": self._abs(data_root)}
        for key, val in raw.items():
            if key == "data_root":
                continue
            out[key] = self._abs(str(val).replace("{data_root}", str(out["data_root"])))
        return out

    def _abs(self, p: str) -> Path:
        path = Path(p).expanduser()
        return path if path.is_absolute() else (self.root / path)

    def path(self, key: str) -> Path:
        if key not in self._paths:
            raise ConfigError(f"paths.{key} missing from {self.config_dir / 'settings.yaml'}")
        return self._paths[key]

    def rel(self, p: Path) -> str:
        """Path as stored in the manifest: relative to the project root when possible."""
        p = Path(p).resolve()
        try:
            return p.relative_to(self.root).as_posix()
        except ValueError:
            return p.as_posix()

    # ---- sections ----------------------------------------------------------
    def section(self, name: str) -> dict[str, Any]:
        sec = self.data.get(name)
        if not isinstance(sec, dict):
            raise ConfigError(f"Section '{name}:' missing from {self.config_dir / 'settings.yaml'}")
        return sec

    def load(self, filename: str) -> dict[str, Any]:
        return load_yaml(self.config_dir / filename)

    # ---- taxonomy ----------------------------------------------------------
    @property
    def countries(self) -> list[str]:
        return list(self.section("taxonomy").get("countries") or [])

    @property
    def conditions(self) -> list[str]:
        return list(self.section("taxonomy").get("conditions") or [])

    def normalize_country(self, value: Any, *, where: str = "") -> str:
        v = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
        aliases = self.section("taxonomy").get("country_aliases") or {}
        v = aliases.get(v, v)
        if v not in self.countries:
            raise ConfigError(
                f"{where}unknown country {value!r}; expected one of {self.countries} "
                f"(or an alias from taxonomy.country_aliases)"
            )
        return v

    def normalize_condition(self, value: Any, *, where: str = "") -> str:
        v = str(value or "").strip().lower()
        if v not in self.conditions:
            raise ConfigError(f"{where}condition must be one of {self.conditions}, got {value!r}")
        return v

    def guess_country(self, parts: list[str]) -> str | None:
        for part in parts:
            try:
                return self.normalize_country(part)
            except ConfigError:
                continue
        return None

    def guess_condition(self, parts: list[str]) -> str | None:
        for part in parts:
            if part.lower() in self.conditions:
                return part.lower()
        return None


def env_required(name: str, hint: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise ConfigError(f"Environment variable {name} is not set. {hint}")
    return val
