import pytest

from gccvt import clean
from gccvt.manifest import Manifest


def touch(p, data=b"x" * 10):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


@pytest.fixture
def data(cfg_dir, tmp_path):
    """Two KSA YouTube clips (one labelled), one UAE clip, one Roboflow dataset, plus derived folders."""
    d = tmp_path / "data"
    m = Manifest(d / "manifest.csv")
    for cid, country, n in [("plate-ksa-day-clip_0001", "ksa", 1), ("plate-ksa-day-clip_0002", "ksa", 2),
                            ("plate-uae-day-clip_0001", "uae", 1)]:
        clip = touch(d / f"plates/{country}/day/clip_000{n}.mp4")
        meta = touch(clip.with_suffix(".json"))
        m.upsert({"clip_id": cid, "task": "plate", "source": "youtube", "country": country, "condition": "day",
                  "path": str(clip), "meta_path": str(meta)})
        touch(d / f"frames/{cid}/000000.jpg")
        touch(d / f"annotations/{cid}/yolo/labels/000000.txt")
    touch(d / "annotations/plate-ksa-day-clip_0001/gt/annotations.xml")      # labelled
    touch(d / "plates/ksa/night/clip_0002.tmp.mp4")                           # collector leftover
    rf = d / "plates_images/uae/ws__proj__v1"
    touch(rf / "test/images/a.jpg"); touch(rf / ".complete")
    m.upsert({"clip_id": "plateimg-uae-ws__proj__v1", "task": "plate", "source": "roboflow", "country": "uae",
              "path": str(rf)})
    m.save()
    touch(d / "raw/roboflow/ws__proj__v1__yolov8.zip")
    touch(d / "results/run1/metrics.json")
    return d


def run(cfg_dir, *argv):
    return clean.main(["--config-dir", str(cfg_dir), *argv])


def rows(d):
    return set(Manifest(d / "manifest.csv").rows)


def test_overview_and_preview_delete_nothing(cfg_dir, data, capsys):
    before = sorted(p for p in data.rglob("*"))
    assert run(cfg_dir) == 0
    out = capsys.readouterr().out
    assert "clips" in out and "never in 'all'" in out
    assert run(cfg_dir, "all") == 0
    assert "Nothing deleted" in capsys.readouterr().out
    assert sorted(p for p in data.rglob("*")) == before


def test_clips_keeps_labelled_clip_and_drops_rows(cfg_dir, data, capsys):
    assert run(cfg_dir, "clips", "--yes") == 0
    assert "kept (labelled" in capsys.readouterr().out
    assert rows(data) == {"plate-ksa-day-clip_0001", "plateimg-uae-ws__proj__v1"}
    assert (data / "plates/ksa/day/clip_0001.mp4").is_file()
    assert (data / "frames/plate-ksa-day-clip_0001").is_dir()
    assert not (data / "plates/ksa/day/clip_0002.mp4").exists()
    assert not (data / "plates/uae").exists()                        # emptied folders are pruned
    assert not (data / "frames/plate-ksa-day-clip_0002").exists()
    assert not (data / "annotations/plate-ksa-day-clip_0002").exists()
    assert (data / "plates_images/uae/ws__proj__v1/.complete").is_file()   # other categories untouched


def test_filters(cfg_dir, data):
    assert run(cfg_dir, "clips", "frames", "--country", "uae", "--yes") == 0
    assert not (data / "plates/uae").exists()
    assert (data / "plates/ksa/day/clip_0002.mp4").is_file()
    assert (data / "frames/plate-ksa-day-clip_0002").is_dir()
    assert (data / "plates/ksa/night/clip_0002.tmp.mp4").is_file()      # no sweep when filtered


def test_include_labelled_sweeps_folder_but_never_gt(cfg_dir, data):
    assert run(cfg_dir, "all", "--include-labelled", "--yes") == 0
    assert rows(data) == set()
    assert list((data / "plates").iterdir()) == []                      # leftovers swept too
    for sub in ("frames", "plates_images", "raw", "results"):
        assert not any((data / sub).iterdir())
    assert (data / "annotations/plate-ksa-day-clip_0001/gt/annotations.xml").is_file()
    assert not (data / "annotations/plate-ksa-day-clip_0001/yolo").exists()


def test_gt_only_when_named(cfg_dir, data):
    assert run(cfg_dir, "gt", "--yes") == 0
    assert not (data / "annotations/plate-ksa-day-clip_0001/gt").exists()
    assert (data / "annotations/plate-ksa-day-clip_0001/yolo").is_dir()
    assert len(rows(data)) == 4
