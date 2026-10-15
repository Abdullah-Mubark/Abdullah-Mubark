"""HTTP with retries/backoff and resumable, checksum-verified downloads."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import requests

from .errors import ChecksumError, DownloadError

log = logging.getLogger(__name__)
T = TypeVar("T")

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
STATUS_HINTS = {
    401: "authentication failed - check your API key",
    403: "access denied - check your API key / permissions for this resource",
    404: "not found - check the URL, record id or dataset slug in config",
}


_SECRET_RX = re.compile(r"(api_key|key|token)=[^&\s'\"]+", re.I)


def redact(text: object) -> str:
    """Hide API keys that may appear in URLs inside error messages."""
    return _SECRET_RX.sub(r"\1=***", str(text))


class TransientError(Exception):
    """Raised internally for failures worth retrying."""


TRANSIENT = (
    TransientError,
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def with_retries(
    fn: Callable[[], T],
    *,
    what: str,
    attempts: int,
    backoff_s: float,
    backoff_max_s: float,
    retry_on: tuple[type[BaseException], ...] = TRANSIENT,
) -> T:
    attempts = max(1, int(attempts))
    for i in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as e:
            if i == attempts:
                raise DownloadError(f"{what} failed after {attempts} attempts: {redact(e)}") from None
            delay = min(backoff_max_s, backoff_s * 2 ** (i - 1))
            log.warning("%s failed (attempt %d/%d): %s - retrying in %.0fs", what, i, attempts, redact(e), delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def check_status(resp: requests.Response, what: str) -> None:
    if resp.status_code in RETRYABLE_STATUS:
        raise TransientError(f"HTTP {resp.status_code}")
    if resp.status_code >= 400:
        hint = STATUS_HINTS.get(resp.status_code, "")
        body = resp.text[:300].strip().replace("\n", " ")
        raise DownloadError(f"{what}: HTTP {resp.status_code} {hint}. Response: {body}")


def file_hash(path: Path, algo: str = "md5") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_checksum(value: str | None) -> tuple[str, str] | None:
    """'md5:abc..' -> ('md5', 'abc..'). Bare hex is treated as md5."""
    if not value:
        return None
    if ":" in value:
        algo, digest = value.split(":", 1)
        return algo.lower(), digest.lower()
    return "md5", value.lower()


def _marker(dest: Path) -> Path:
    return dest.with_name(dest.name + ".verified")


def is_verified(dest: Path, size: int | None, checksum: str | None) -> bool:
    """True if dest was already downloaded and verified against the same size/checksum."""
    marker = _marker(dest)
    if not (dest.is_file() and marker.is_file()):
        return False
    try:
        info = json.loads(marker.read_text())
    except (OSError, ValueError):
        return False
    if size is not None and dest.stat().st_size != size:
        return False
    return info.get("checksum") == checksum and info.get("size") == dest.stat().st_size


class HttpClient:
    def __init__(self, net_cfg: dict[str, Any]):
        self.timeout = float(net_cfg.get("timeout_s", 60))
        self.attempts = int(net_cfg.get("retries", 5))
        self.backoff_s = float(net_cfg.get("backoff_s", 2))
        self.backoff_max_s = float(net_cfg.get("backoff_max_s", 60))
        self.chunk = int(float(net_cfg.get("chunk_mib", 4)) * 1024 * 1024)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = net_cfg.get("user_agent", "gcc-vision-testbench")

    def _retry(self, fn: Callable[[], T], what: str) -> T:
        return with_retries(
            fn, what=what, attempts=self.attempts, backoff_s=self.backoff_s, backoff_max_s=self.backoff_max_s
        )

    def get_json(self, url: str, *, params: dict | None = None, what: str | None = None) -> Any:
        what = what or f"GET {url}"

        def call() -> Any:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            check_status(resp, what)
            try:
                return resp.json()
            except ValueError as e:
                raise DownloadError(f"{what}: expected JSON, got: {resp.text[:200]!r}") from e

        return self._retry(call, what)

    def download(
        self,
        url: str,
        dest: Path,
        *,
        size: int | None = None,
        checksum: str | None = None,
        params: dict | None = None,
        what: str | None = None,
    ) -> Path:
        """Download url -> dest, resuming a partial .part file. Verifies size and checksum.

        Skips the download entirely if dest was already verified with the same checksum.
        """
        what = what or f"download {dest.name}"
        if is_verified(dest, size, checksum):
            log.info("Already downloaded and verified: %s", dest.name)
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        if dest.is_file() and not part.exists():
            # Downloaded before but not verified (e.g. marker deleted): verify in place.
            dest.rename(part)

        for full_attempt in (1, 2):
            self._retry(lambda: self._fetch_into(url, part, size, params, what), what)
            try:
                self._verify(part, size, checksum)
                break
            except ChecksumError:
                part.unlink(missing_ok=True)
                if full_attempt == 2:
                    raise
                log.warning("%s: checksum mismatch - deleting and downloading again from scratch", what)

        part.replace(dest)
        _marker(dest).write_text(json.dumps({"checksum": checksum, "size": dest.stat().st_size}))
        log.info("Downloaded %s (%.1f MiB)", dest.name, dest.stat().st_size / 2**20)
        return dest

    def _fetch_into(self, url: str, part: Path, size: int | None, params: dict | None, what: str) -> None:
        have = part.stat().st_size if part.exists() else 0
        if size is not None and have > size:
            log.warning("%s: partial file larger than expected - restarting", what)
            part.unlink()
            have = 0
        if size is not None and have == size:
            return
        headers = {"Range": f"bytes={have}-"} if have else {}
        with self.session.get(url, params=params, headers=headers, stream=True, timeout=self.timeout) as resp:
            if resp.status_code == 416 and have:
                if size is None:
                    return  # server says we already have everything
                part.unlink()
                raise TransientError("server rejected resume range; restarting")
            check_status(resp, what)
            mode = "ab"
            if have and resp.status_code != 206:
                log.info("%s: server does not support resume - restarting from 0", what)
                have, mode = 0, "wb"
            elif have:
                log.info("%s: resuming at %.1f MiB", what, have / 2**20)
            total = size
            if total is None and resp.headers.get("Content-Length"):
                total = have + int(resp.headers["Content-Length"])
            done, last_log = have, time.monotonic()
            with open(part, mode) as f:
                for chunk in resp.iter_content(self.chunk):
                    f.write(chunk)
                    done += len(chunk)
                    if time.monotonic() - last_log > 15:
                        pct = f" ({100 * done / total:.0f}%)" if total else ""
                        log.info("%s: %.1f MiB%s", what, done / 2**20, pct)
                        last_log = time.monotonic()
        got = part.stat().st_size
        if size is not None and got < size:
            raise TransientError(f"connection ended early ({got}/{size} bytes)")

    @staticmethod
    def _verify(part: Path, size: int | None, checksum: str | None) -> None:
        got = part.stat().st_size
        if size is not None and got != size:
            raise ChecksumError(f"{part.name}: size {got} != expected {size}")
        parsed = parse_checksum(checksum)
        if parsed:
            algo, want = parsed
            try:
                actual = file_hash(part, algo)
            except ValueError as e:
                raise ChecksumError(f"unsupported checksum algorithm {algo!r}") from e
            if actual != want:
                raise ChecksumError(f"{part.name}: {algo} {actual} != expected {want}")
