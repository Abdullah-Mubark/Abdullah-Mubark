import hashlib

import pytest
import requests

from conftest import make_video
from gccvt.config import Settings
from gccvt.errors import ChecksumError, ConfigError, DownloadError
from gccvt.manifest import Manifest, next_clip_path
from gccvt.media import parse_timestamp, probe, transcode
from gccvt.net import HttpClient, redact

NET = {"retries": 3, "backoff_s": 0, "backoff_max_s": 0, "timeout_s": 5, "chunk_mib": 0.01}


@pytest.mark.parametrize("v,want", [("01:02:03.5", 3723.5), ("02:03", 123), ("90", 90), (12.5, 12.5), (0, 0)])
def test_parse_timestamp(v, want):
    assert parse_timestamp(v) == want


@pytest.mark.parametrize("v", ["abc", "1:2:3:4", "-5"])
def test_parse_timestamp_bad(v):
    with pytest.raises(ValueError):
        parse_timestamp(v)


def test_settings_paths_and_country(cfg_dir, tmp_path, monkeypatch):
    s = Settings(config_dir=cfg_dir)
    assert s.path("plates") == tmp_path / "data" / "plates"
    assert s.normalize_country("Saudi") == "ksa"
    assert s.normalize_country("AE") == "uae"
    with pytest.raises(ConfigError, match="unknown country"):
        s.normalize_country("france")
    monkeypatch.setenv("GCCVT_DATA_ROOT", str(tmp_path / "elsewhere"))
    assert Settings(config_dir=cfg_dir).path("manifest") == tmp_path / "elsewhere" / "manifest.csv"


def test_manifest_roundtrip(tmp_path):
    m = Manifest(tmp_path / "m.csv")
    m.upsert({"clip_id": "b", "task": "plate", "duration_s": 1.5, "source_key": "k1"})
    m.upsert({"clip_id": "a", "task": "face"})
    m.save()
    m2 = Manifest(tmp_path / "m.csv")
    assert list(m2.rows) == ["a", "b"]
    assert m2.get("b")["duration_s"] == "1.5"
    assert m2.find_by_source_key("k1")["clip_id"] == "b"
    added = m2.get("b")["added_at"]
    m2.upsert({"clip_id": "b", "notes": "x"})
    assert m2.get("b")["added_at"] == added and m2.get("b")["task"] == "plate"
    with pytest.raises(ValueError):
        m2.upsert({"clip_id": "c", "bogus": 1})


def test_next_clip_path(tmp_path):
    assert next_clip_path(tmp_path / "none", "clip_{n:04d}", ".mp4")[0].name == "clip_0001.mp4"
    (tmp_path / "clip_0001.mp4").touch()
    (tmp_path / "clip_0007.json").touch()
    assert next_clip_path(tmp_path, "clip_{n:04d}", ".mp4")[0].name == "clip_0008.mp4"
    assert next_clip_path(tmp_path, "clip_{n:04d}", ".mp4", ["clip_0012"])[0].name == "clip_0013.mp4"


def test_manifest_clip_path_keeps_listed_ids(tmp_path):
    m = Manifest(tmp_path / "manifest.csv")
    m.upsert({"clip_id": "plate-ksa-day-clip_0003", "source_key": "youtube:a:0-60",
              "path": "/other/pc/data/plates/ksa/day/clip_0003.mp4"})
    m.upsert({"clip_id": "plate-uae-day-clip_0009", "source_key": "youtube:b:0-60"})
    folder = tmp_path / "plates/ksa/day"                    # empty: clips not collected here yet
    args = (folder, "clip_{n:04d}", ".mp4", "plate-ksa-day")
    assert m.clip_path(*args, "youtube:a:0-60") == folder / "clip_0003.mp4"   # listed: same id
    assert m.clip_path(*args, "youtube:new:0-60").name == "clip_0004.mp4"     # new: skips listed ids
    assert m.clip_path(*args, "youtube:b:0-60").name == "clip_0004.mp4"       # other folder's row


def test_download_resume_and_skip(server, tmp_path):
    data = bytes(range(256)) * 4000
    server.routes["/f.bin"] = data
    server.drop_after["/f.bin"] = 300_000
    md5 = "md5:" + hashlib.md5(data).hexdigest()
    c = HttpClient(NET)
    out = c.download(server.url + "/f.bin", tmp_path / "f.bin", size=len(data), checksum=md5)
    assert out.read_bytes() == data
    assert server.hits["/f.bin"] == 2  # dropped once, resumed once
    c.download(server.url + "/f.bin", tmp_path / "f.bin", size=len(data), checksum=md5)
    assert server.hits["/f.bin"] == 2  # verified -> skipped


def test_download_checksum_mismatch(server, tmp_path):
    server.routes["/f.bin"] = b"hello"
    with pytest.raises(ChecksumError):
        HttpClient(NET).download(server.url + "/f.bin", tmp_path / "f.bin", checksum="md5:" + "0" * 32)
    assert server.hits["/f.bin"] == 2
    assert not (tmp_path / "f.bin").exists() and not (tmp_path / "f.bin.part").exists()


def test_http_404_no_retry_with_hint(server):
    with pytest.raises(DownloadError, match="not found"):
        HttpClient(NET).get_json(server.url + "/missing")
    assert server.hits["/missing"] == 1


def test_http_503_retried(server):
    calls = []
    server.routes["/flaky"] = lambda h: (calls.append(1) or (503 if len(calls) < 3 else 200), {"ok": True})
    assert HttpClient(NET).get_json(server.url + "/flaky") == {"ok": True}
    assert len(calls) == 3


def test_redact():
    assert redact("GET https://x/y?api_key=SECRET123&a=1") == "GET https://x/y?api_key=***&a=1"
    err = requests.ConnectionError("Max retries exceeded with url: /a?api_key=SECRET")
    assert "SECRET" not in redact(err)


def test_transcode_trim_and_scale(tmp_path):
    src = make_video(tmp_path / "src.mp4", seconds=5, size="640x360")
    out = transcode(src, tmp_path / "a.mp4", {"target_height": 1080, "allow_upscale": False, "preset": "ultrafast"},
                    start=1, duration=2)
    assert out["height"] == 360 and abs(out["duration_s"] - 2) < 0.2
    out = transcode(src, tmp_path / "b.mp4", {"target_height": 1080, "allow_upscale": True, "preset": "ultrafast"})
    assert out["height"] == 1080 and out["width"] == 1920
    assert probe(tmp_path / "b.mp4")["fps"] == 25
