import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from conftest import edit_yaml, make_jpeg, make_video
from gccvt.adapters import BaseAdapter, Box, create_adapter, register
from gccvt.errors import ConfigError
from gccvt.manifest import Manifest
from gccvt.prep import dedupe, extract, preannotate
from gccvt.prep.common import read_index


def static_then_noise(path, static_s=3, noise_s=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"color=c=red:s=320x240:d={static_s}:r=25",
        "-f", "lavfi", "-i", f"nullsrc=s=320x240:d={noise_s}:r=25,geq=lum='random(1)*255':cb=128:cr=128",
        "-filter_complex", "[0:v]format=yuv420p[a];[1:v]format=yuv420p[b];[a][b]concat=n=2:v=1[v]",
        "-map", "[v]", str(path)], check=True)
    return path


@pytest.fixture
def clips(cfg_dir, tmp_path):
    """Manifest with one plate video and one ChokePoint-style image sequence."""
    data = tmp_path / "data"
    video = static_then_noise(data / "plates/ksa/night/clip_0001.mp4")
    seq = data / "faces/chokepoint/P1E_S1/P1E_S1_C1"
    for i in range(30):
        make_jpeg(seq / f"{i:08d}.jpg")
    m = Manifest(data / "manifest.csv")
    m.upsert({"clip_id": "plate-ksa-night-clip_0001", "task": "plate", "source": "youtube",
              "country": "ksa", "condition": "night", "path": str(video), "fps": 25})
    m.upsert({"clip_id": "face-chokepoint-P1E_S1__P1E_S1_C1", "task": "face", "source": "chokepoint",
              "path": str(seq), "fps": 30})
    m.upsert({"clip_id": "plateimg-ksa-x", "task": "plate", "source": "roboflow", "path": str(data)})
    m.save()
    return data


def test_hashes():
    rng = np.random.default_rng(0)
    a = Image.fromarray(rng.integers(0, 255, (120, 160), dtype=np.uint8))
    b = Image.fromarray(255 - np.asarray(a))
    noisy = Image.fromarray(np.clip(np.asarray(a, dtype=int) + rng.integers(-3, 3, (120, 160)), 0, 255).astype(np.uint8))
    for fn in (dedupe.phash, dedupe.dhash):
        assert dedupe.hamming(fn(a), fn(a)) == 0
        assert dedupe.hamming(fn(a), fn(noisy)) <= 6
        assert dedupe.hamming(fn(a), fn(b)) > 30


def test_extract_and_dedupe(cfg_dir, clips):
    args = ["--config-dir", str(cfg_dir)]
    assert extract.main(args) == 0
    vdir = clips / "frames/plate-ksa-night-clip_0001"
    sdir = clips / "frames/face-chokepoint-P1E_S1__P1E_S1_C1"
    rows = read_index(vdir)
    assert len(rows) == 10 and rows[3]["t_s"] == "1.500"
    srows = read_index(sdir)
    assert [r["frame"] for r in srows] == ["00000000.jpg", "00000015.jpg"]  # 30 fps -> 2 fps, names kept
    assert not (clips / "frames/plateimg-ksa-x").exists()

    mtime = (vdir / "000000.jpg").stat().st_mtime_ns
    assert extract.main(args) == 0
    assert (vdir / "000000.jpg").stat().st_mtime_ns == mtime  # up to date -> untouched

    assert dedupe.main(args) == 0
    rows = read_index(vdir)
    status = [r["status"] for r in rows]
    assert status[0] == "kept" and status[1] == "duplicate" and rows[1]["dup_of"] == "000000.jpg"
    assert status[-3:] == ["kept"] * 3  # noise frames are all different
    n_dup = status.count("duplicate")
    assert 4 <= n_dup <= 6
    assert len(list((vdir / "_duplicates").iterdir())) == n_dup
    assert not (sdir / ".dedupe.json").exists()  # chokepoint override: dedupe disabled

    # Different threshold: duplicates restored and re-checked
    assert dedupe.main(args + ["--threshold", "0", "--clip", "plate-ksa-night-clip_0001"]) == 0
    assert len(list(vdir.glob("*.jpg"))) + len(list((vdir / "_duplicates").glob("*.jpg"))) == 10

    assert dedupe.main(args + ["--undo"]) == 0
    assert len(list(vdir.glob("*.jpg"))) == 10 and not (vdir / "_duplicates").exists()
    assert {r["status"] for r in read_index(vdir)} == {"kept"}

    # Changing fps re-extracts from scratch
    assert extract.main(args + ["--fps", "1", "--force"]) == 0
    assert len(read_index(vdir)) == 5


def test_extract_missing_source(cfg_dir, clips):
    (clips / "plates/ksa/night/clip_0001.mp4").unlink()
    assert extract.main(["--config-dir", str(cfg_dir), "--task", "plate"]) == 1


@register("fake_plate")
class FakePlate(BaseAdapter):
    tasks = {"plate_det"}
    loaded = {}

    def load(self, **params):
        FakePlate.loaded = params

    def detect(self, frame):
        h, w = frame.shape[:2]
        return [Box(np.int64(10), np.int64(20), 110, 60, np.float32(0.9), "whatever"),  # numpy scalars are fine
                Box(w - 20, h - 10, w + 50, h + 50, 0.5, "x"),     # partly outside -> clipped
                Box(w + 5, h + 5, w + 50, h + 50, 0.4, "x")]       # fully outside -> dropped


def test_preannotate(cfg_dir, clips, tmp_path):
    args = ["--config-dir", str(cfg_dir)]
    assert preannotate.main(args) == 0  # nothing enabled -> no-op
    extract.main(args)
    dedupe.main(args)
    edit_yaml(cfg_dir / "prep.yaml", lambda c: c["preannotate"]["detectors"].update(
        plate={"enabled": True, "adapter": "fake_plate", "tasks": ["plate"], "label": "plate",
               "params": {"conf": 0.3}}, face={"enabled": False}))
    assert preannotate.main(args) == 0
    assert FakePlate.loaded == {"label": "plate", "conf": 0.3}

    out = clips / "annotations/plate-ksa-night-clip_0001"
    kept = [r["frame"] for r in read_index(clips / "frames/plate-ksa-night-clip_0001") if r["status"] == "kept"]
    labels = sorted(p.stem for p in (out / "yolo/labels").glob("*.txt"))
    assert labels == sorted(k.split(".")[0] for k in kept)
    lines = (out / "yolo/labels/000000.txt").read_text().split("\n")
    assert lines[0] == "0 0.187500 0.166667 0.312500 0.166667"  # (10,20,110,60) in 320x240
    assert len([ln for ln in lines if ln]) == 2
    assert (out / "yolo/classes.txt").read_text() == "plate\nface\n"

    root = ET.parse(out / "cvat/annotations.xml").getroot()
    images = root.findall("image")
    assert len(images) == len(kept) and images[0].get("width") == "320"
    boxes = images[0].findall("box")
    assert boxes[0].get("label") == "plate" and boxes[0].get("source") == "auto"
    assert float(boxes[1].get("xbr")) == 320.0  # clipped to the image
    assert not (clips / "annotations/face-chokepoint-P1E_S1__P1E_S1_C1").exists()  # plate detector only

    mtime = (out / "cvat/annotations.xml").stat().st_mtime_ns
    assert preannotate.main(args) == 0
    assert (out / "cvat/annotations.xml").stat().st_mtime_ns == mtime  # up to date


def test_preannotate_bad_label(cfg_dir, clips):
    edit_yaml(cfg_dir / "prep.yaml", lambda c: c["preannotate"]["detectors"].update(
        plate={"enabled": True, "adapter": "fake_plate", "label": "car"}))
    with pytest.raises(ConfigError, match="not in preannotate.classes"):
        preannotate.main(["--config-dir", str(cfg_dir)])


def test_registry():
    with pytest.raises(ConfigError, match="unknown adapter"):
        create_adapter("nope")
    with pytest.raises(ConfigError, match="cannot import"):
        create_adapter("no.such.module:Thing")
    a = create_adapter("test_prep:FakePlate", {"x": 1})
    assert isinstance(a, FakePlate) and FakePlate.loaded == {"x": 1}
    with pytest.raises(NotImplementedError):
        a.read_plate(np.zeros((2, 2, 3), np.uint8))
