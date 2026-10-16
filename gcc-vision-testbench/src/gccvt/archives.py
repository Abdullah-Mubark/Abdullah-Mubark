"""Safe archive extraction (no path traversal)."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

from .errors import MediaError

ARCHIVE_SUFFIXES = (".tar.xz", ".tar.gz", ".tgz", ".tar.bz2", ".tar", ".zip")


def is_archive(path: Path) -> bool:
    return path.name.lower().endswith(ARCHIVE_SUFFIXES)


def extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    name = archive.name.lower()
    try:
        if name.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                for member in zf.namelist():
                    if not (root / member).resolve().is_relative_to(root):
                        raise MediaError(f"{archive.name}: unsafe path in archive: {member}")
                zf.extractall(root)
        else:
            with tarfile.open(archive) as tf:
                if hasattr(tarfile, "data_filter"):
                    tf.extractall(root, filter="data")
                else:  # Python < 3.11.4
                    for m in tf.getmembers():
                        if m.issym() or m.islnk() or not (root / m.name).resolve().is_relative_to(root):
                            raise MediaError(f"{archive.name}: unsafe member in archive: {m.name}")
                    tf.extractall(root)
    except (tarfile.TarError, zipfile.BadZipFile, EOFError) as e:
        raise MediaError(f"Could not extract {archive.name} (corrupt or incomplete?): {e}") from e
