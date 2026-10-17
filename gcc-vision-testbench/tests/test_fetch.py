import shutil
import tarfile

import pytest

from gccvt import fetch
from gccvt.collect import chokepoint, roboflow, summary, youtube
from gccvt.manifest import Manifest
from gccvt.prep import sequence_clips


def touch(p, text="x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


@pytest.fixture
def calls(monkeypatch):
    """Record which collectors ran; YouTube fails (e.g. a video went private), the rest succeed."""
    ran = []
    for mod, code in ((youtube, 1), (chokepoint, 0), (roboflow, 0), (sequence_clips, 0)):
        monkeypatch.setattr(mod, "main", lambda argv, m=mod, c=code: ran.append(m.__name__.split(".")[-1]) or c)
    monkeypatch.setattr(summary, "main", lambda argv: 0)
    return ran


def test_pack_then_fetch_on_new_machine(cfg_dir, tmp_path, calls, monkeypatch):
    data = tmp_path / "data"
    m = Manifest(data / "manifest.csv")
    m.upsert({"clip_id": "plate-ksa-day-clip_0003", "source_key": "youtube:abc:60.000-120.000"})
    m.save()
    touch(data / "annotations/plate-ksa-day-clip_0003/gt/annotations.xml", "<labels/>")
    touch(data / "annotations/plate-ksa-day-clip_0003/yolo/labels/000000.txt")       # draft: not packed
    touch(data / "results/run1/metrics.json", "{}")
    touch(data / "results/run1/clips/plate-ksa-day-clip_0003.mp4")                 # video: not packed
    touch(data / "plates/ksa/day/clip_0003.mp4")                                    # video: not packed
    touch(tmp_path / "models/plate_yolov8.pt", "weights")                           # models/: packed
    args = ["--config-dir", str(cfg_dir)]
    monkeypatch.setattr("gccvt.config.PROJECT_ROOT", tmp_path)                     # bundle goes in tmp_path

    assert fetch.main([*args, "--pack"]) == 0
    bundle = tmp_path / fetch.DEFAULT_BUNDLE
    with tarfile.open(bundle) as tf:
        assert sorted(tf.getnames()) == ["annotations/plate-ksa-day-clip_0003/gt/annotations.xml",
                                         "manifest.csv", "models/plate_yolov8.pt", "results/run1/metrics.json"]

    shutil.rmtree(data)                                                             # "the other PC"
    shutil.rmtree(tmp_path / "models")
    monkeypatch.setenv("ROBOFLOW_API_KEY", "")
    assert fetch.main(args) == 1                                                    # YouTube step failed
    assert calls == ["youtube", "chokepoint", "sequence_clips"]                     # no key: Roboflow skipped
    assert (data / "annotations/plate-ksa-day-clip_0003/gt/annotations.xml").read_text() == "<labels/>"
    assert "plate-ksa-day-clip_0003" in Manifest(data / "manifest.csv").rows        # ids pinned before collecting
    assert (tmp_path / "models/plate_yolov8.pt").read_text() == "weights"           # models/ routed back
    assert not (data / "models").exists() and not [p for p in data.iterdir() if p.name.startswith("tmp")]
    assert not bundle.exists() and bundle.with_name(bundle.name + ".unpacked").is_file()

    calls.clear()
    monkeypatch.setenv("ROBOFLOW_API_KEY", "k")
    fetch.main(args)                                                                # re-run: bundle not re-applied
    assert calls == ["youtube", "chokepoint", "roboflow", "sequence_clips"]


def test_fetch_missing_bundle(cfg_dir):
    from gccvt.errors import GccvtError
    with pytest.raises(GccvtError, match="bundle not found"):
        fetch.main(["--config-dir", str(cfg_dir), "--from", "nope.tgz"])
