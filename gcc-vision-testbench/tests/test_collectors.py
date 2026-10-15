import hashlib
import io
import json
import tarfile
import zipfile

import pytest

from conftest import edit_yaml, make_jpeg, make_video
from gccvt.collect import chokepoint, local, roboflow, youtube
from gccvt.errors import ConfigError
from gccvt.manifest import Manifest


def _manifest(tmp_path):
    return Manifest(tmp_path / "data" / "manifest.csv")


# ---------------------------------------------------------------- ChokePoint
def test_chokepoint_end_to_end(server, cfg_dir, tmp_path):
    src = tmp_path / "build"
    for i in range(3):
        make_jpeg(src / "P1E_S1" / "P1E_S1_C1" / f"{i:08d}.jpg")
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:xz") as tf:
        tf.add(src / "P1E_S1", arcname="P1E_S1")
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("groundtruth/P1E_S1_C1.xml", "<dataset/>")
    blobs = {"P1E_S1.tar.xz": tar_buf.getvalue(), "groundtruth.zip": zip_buf.getvalue(), "P2L_S1.tar.xz": b"x"}
    files = []
    for name, data in blobs.items():
        server.routes[f"/files/{name}"] = data
        files.append({"key": name, "size": len(data), "checksum": "md5:" + hashlib.md5(data).hexdigest(),
                      "links": {"self": f"{server.url}/files/{name}"}})
    server.routes["/api/records/815657"] = {"files": files}
    edit_yaml(cfg_dir / "chokepoint.yaml", lambda c: (
        c["zenodo"].update(api_url=server.url + "/api/records/{record_id}"),
        c["files"].update(exclude=["P2L*"])))

    assert chokepoint.main(["--config-dir", str(cfg_dir)]) == 0
    rows = _manifest(tmp_path).rows
    row = rows["face-chokepoint-P1E_S1__P1E_S1_C1"]
    assert row["frame_count"] == "3" and row["width"] == "800" and row["fps"] == "30"
    assert row["gt_path"].endswith("groundtruth/P1E_S1_C1.xml")
    assert "/files/P2L_S1.tar.xz" not in server.hits

    hits = dict(server.hits)
    assert chokepoint.main(["--config-dir", str(cfg_dir)]) == 0
    assert server.hits["/files/P1E_S1.tar.xz"] == hits["/files/P1E_S1.tar.xz"]  # nothing re-downloaded


# ---------------------------------------------------------------- Roboflow
def test_roboflow_end_to_end(server, cfg_dir, tmp_path, monkeypatch):
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        for split in ("train", "valid"):
            zf.writestr(f"{split}/images/a.jpg", b"jpg")
            zf.writestr(f"{split}/labels/a.txt", "0 0.5 0.5 0.1 0.1\n")
        zf.writestr("data.yaml", "names: [plate]\n")
    server.routes["/dl/export.zip"] = zbuf.getvalue()
    calls = []

    def export(h):
        assert "api_key=KEY123" in h.path
        calls.append(1)
        return 200, ({"progress": 0.5} if len(calls) == 1 else {"export": {"link": server.url + "/dl/export.zip"}})

    server.routes["/ws/ksa-plates/2/yolov8"] = export
    server.routes["/ws/ksa-plates"] = {"project": {"license": "CC BY 4.0"}}
    edit_yaml(cfg_dir / "roboflow_datasets.yaml", lambda c: c.update(
        api_url=server.url, export_poll_s=0,
        datasets=[{"url": "https://universe.roboflow.com/ws/ksa-plates/dataset/2", "country": "Saudi"}]))
    monkeypatch.setenv("ROBOFLOW_API_KEY", "KEY123")

    assert roboflow.main(["--config-dir", str(cfg_dir)]) == 0
    row = _manifest(tmp_path).rows["plateimg-ksa-ws__ksa-plates__v2"]
    assert row["frame_count"] == "2" and row["license"] == "CC BY 4.0" and row["country"] == "ksa"
    assert (tmp_path / "data/plates_images/ksa/ws__ksa-plates__v2/train/labels/a.txt").is_file()
    assert roboflow.main(["--config-dir", str(cfg_dir)]) == 0
    assert len(calls) == 2  # second run skipped


def test_roboflow_missing_key(cfg_dir, monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "")
    edit_yaml(cfg_dir / "roboflow_datasets.yaml", lambda c: c.update(
        datasets=[{"slug": "a/b/1", "country": "qatar"}]))
    with pytest.raises(ConfigError, match="ROBOFLOW_API_KEY"):
        roboflow.main(["--config-dir", str(cfg_dir)])


def test_roboflow_bad_slug(cfg_dir):
    edit_yaml(cfg_dir / "roboflow_datasets.yaml", lambda c: c.update(datasets=[{"slug": "a/b", "country": "oman"}]))
    with pytest.raises(ConfigError, match="workspace/project/version"):
        roboflow.main(["--config-dir", str(cfg_dir)])


# ---------------------------------------------------------------- YouTube
def test_youtube_job_validation(cfg_dir):
    from gccvt.config import Settings
    s = Settings(config_dir=cfg_dir)
    jobs = youtube.build_jobs({"sources": [{"url": "https://youtu.be/abcdefghijk", "country": "kw",
                                            "segments": [{"start": "1:00", "end": "1:30", "condition": "night"},
                                                         {"start": 5, "end": 9, "condition": "day"}]}]}, s)
    assert [(j.country, j.condition, j.start, j.end) for j in jobs] == [
        ("kuwait", "night", 60, 90), ("kuwait", "day", 5, 9)]
    assert jobs[0].source_key == "youtube:abcdefghijk:60.000-90.000"
    with pytest.raises(ConfigError, match="must be after"):
        youtube.build_jobs({"sources": [{"url": "u", "country": "ksa", "condition": "day",
                                         "start": 10, "end": 5}]}, s)
    with pytest.raises(ConfigError, match="condition"):
        youtube.build_jobs({"sources": [{"url": "u", "country": "ksa", "condition": "dusk",
                                         "start": 1, "end": 5}]}, s)


def test_youtube_end_to_end(cfg_dir, tmp_path, monkeypatch):
    downloads = []

    def fake_info(self, url):
        return {"id": "abcdefghijk", "title": "Riyadh drive", "uploader": "someone", "height": 360}

    def fake_download(self, url, template, section):
        downloads.append(section)
        out = template.parent / template.name.replace("%(ext)s", "mkv")
        return make_video(out, seconds=(section[1] - section[0]) if section else 12)

    monkeypatch.setattr(youtube.YtDlp, "info", fake_info)
    monkeypatch.setattr(youtube.YtDlp, "download", fake_download)
    edit_yaml(cfg_dir / "settings.yaml", lambda c: c["video"].update(preset="ultrafast"))
    edit_yaml(cfg_dir / "youtube_sources.yaml", lambda c: c.update(sources=[
        {"url": "https://www.youtube.com/watch?v=abcdefghijk", "country": "ksa", "condition": "night",
         "start": "00:02", "end": "00:05", "notes": "test"},
        {"url": "https://www.youtube.com/watch?v=abcdefghijk", "country": "uae", "condition": "day",
         "start": 1, "end": 3},
    ]))

    assert youtube.main(["--config-dir", str(cfg_dir)]) == 0
    clip = tmp_path / "data/plates/ksa/night/clip_0001.mp4"
    meta = json.loads(clip.with_suffix(".json").read_text())
    assert meta["country"] == "ksa" and meta["source_url"].endswith("abcdefghijk")
    assert abs(meta["duration_s"] - 3) < 0.2 and meta["height"] == 360
    assert (tmp_path / "data/plates/uae/day/clip_0001.mp4").is_file()
    rows = _manifest(tmp_path).rows
    assert rows["plate-ksa-night-clip_0001"]["source"] == "youtube"
    assert not list((tmp_path / "data/raw/youtube/tmp").glob("*"))  # temp cleaned up

    assert youtube.main(["--config-dir", str(cfg_dir)]) == 0
    assert len(downloads) == 2  # already collected -> skipped


def test_youtube_full_mode_reuses_cache(cfg_dir, tmp_path, monkeypatch):
    downloads = []
    monkeypatch.setattr(youtube.YtDlp, "info", lambda self, url: {"id": "zzzzzzzzzzz"})

    def fake_download(self, url, template, section):
        downloads.append(section)
        return make_video(template.parent / template.name.replace("%(ext)s", "mp4"), seconds=10)

    monkeypatch.setattr(youtube.YtDlp, "download", fake_download)
    edit_yaml(cfg_dir / "settings.yaml", lambda c: c["video"].update(preset="ultrafast"))
    edit_yaml(cfg_dir / "youtube_sources.yaml", lambda c: (c["defaults"].update(download_mode="full"), c.update(
        sources=[{"url": "https://youtu.be/zzzzzzzzzzz", "country": "bahrain", "condition": "day",
                  "segments": [{"start": 1, "end": 3}, {"start": 5, "end": 8}]}])))
    assert youtube.main(["--config-dir", str(cfg_dir)]) == 0
    assert downloads == [None]
    clips = sorted((tmp_path / "data/plates/bahrain/day").glob("*.mp4"))
    assert [c.name for c in clips] == ["clip_0001.mp4", "clip_0002.mp4"]


# ---------------------------------------------------------------- Local
def test_local_import(cfg_dir, tmp_path):
    footage = tmp_path / "footage"
    make_video(footage / "qatar" / "night" / "a.mp4", seconds=2)
    make_video(footage / "qatar" / "day" / "b.mov", seconds=2, size="320x240")
    (footage / "qatar" / "day" / "notes.txt").write_text("x")
    args = ["--config-dir", str(cfg_dir), "--src", str(footage)]

    with pytest.raises(ConfigError, match="PDPL"):
        local.main(args)
    assert local.main(args + ["--consent-ack"]) == 0
    assert (tmp_path / "data/plates/qatar/night/clip_0001.mp4").is_file()
    assert (tmp_path / "data/plates/qatar/day/clip_0001.mov").is_file()
    meta = json.loads((tmp_path / "data/plates/qatar/night/clip_0001.json").read_text())
    assert meta["original_path"].endswith("a.mp4") and meta["consent_ack"] is True

    assert local.main(args + ["--consent-ack"]) == 0  # dedupe by hash
    assert len(_manifest(tmp_path).rows) == 2


def test_local_needs_country(cfg_dir, tmp_path):
    make_video(tmp_path / "misc" / "x.mp4", seconds=1)
    assert local.main(["--config-dir", str(cfg_dir), "--src", str(tmp_path / "misc"), "--consent-ack"]) == 1


def test_local_face_transcode(cfg_dir, tmp_path):
    make_video(tmp_path / "door" / "x.avi", seconds=2)
    edit_yaml(cfg_dir / "settings.yaml", lambda c: c["video"].update(preset="ultrafast"))
    assert local.main(["--config-dir", str(cfg_dir), "--src", str(tmp_path / "door"), "--task", "face",
                       "--dataset", "office", "--mode", "transcode", "--consent-ack"]) == 0
    row = _manifest(tmp_path).rows["face-office-clip_0001"]
    assert row["path"].endswith("faces/office/clip_0001.mp4")
