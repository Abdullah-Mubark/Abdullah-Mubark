import json
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg_dir(tmp_path, monkeypatch):
    """A copy of config/ pointing data and logs into tmp_path."""
    d = tmp_path / "config"
    shutil.copytree(ROOT / "config", d)
    s = yaml.safe_load((d / "settings.yaml").read_text())
    s["paths"]["data_root"] = str(tmp_path / "data")
    s["paths"]["logs"] = str(tmp_path / "logs")
    s["network"].update({"retries": 3, "backoff_s": 0, "backoff_max_s": 0, "timeout_s": 5})
    (d / "settings.yaml").write_text(yaml.safe_dump(s))
    monkeypatch.delenv("GCCVT_DATA_ROOT", raising=False)
    monkeypatch.delenv("GCCVT_CONFIG_DIR", raising=False)
    return d


def edit_yaml(path: Path, fn):
    data = yaml.safe_load(path.read_text()) or {}
    fn(data)
    path.write_text(yaml.safe_dump(data, allow_unicode=True))


class Server:
    """Tiny HTTP server: routes map path -> bytes | dict (JSON) | callable(handler) -> (status, bytes|dict)."""

    def __init__(self):
        self.routes = {}
        self.hits = {}
        self.drop_after = {}  # path -> bytes to send before dropping the connection (once)
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                outer.hits[path] = outer.hits.get(path, 0) + 1
                route = outer.routes.get(path)
                if route is None:
                    self.send_response(404); self.end_headers(); self.wfile.write(b"nope"); return
                status = 200
                if callable(route):
                    status, route = route(self)
                if isinstance(route, (dict, list)):
                    body = json.dumps(route).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers(); self.wfile.write(body); return
                body, start = route, 0
                rng = self.headers.get("Range")
                if rng:
                    start = int(rng.split("=")[1].split("-")[0])
                    if start >= len(body):
                        self.send_response(416); self.end_headers(); return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
                else:
                    self.send_response(status)
                chunk = body[start:]
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                cut = outer.drop_after.pop(path, None)
                if cut is not None:
                    self.wfile.write(chunk[:cut]); self.wfile.flush()
                    self.connection.shutdown(2); return
                self.wfile.write(chunk)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


@pytest.fixture
def server():
    s = Server()
    yield s
    s.httpd.shutdown()


def make_video(path: Path, seconds=4, size="640x360", fps=25):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc=duration={seconds}:size={size}:rate={fps}", "-pix_fmt", "yuv420p", str(path)],
                   check=True)
    return path


def make_jpeg(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=800x600",
                    "-frames:v", "1", str(path)], check=True)
    return path
