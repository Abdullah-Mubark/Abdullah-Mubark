import csv
import shutil

from conftest import make_jpeg
from gccvt import clean
from gccvt.manifest import Manifest
from gccvt.media import probe
from gccvt.prep import sequence_clips


def test_sequence_to_mp4(cfg_dir, tmp_path):
    seq = tmp_path / "data/faces/chokepoint/P1E_S1/P1E_S1_C1"
    first = make_jpeg(seq / "00000000.jpg")
    for n in range(1, 6):
        shutil.copy(first, seq / f"{n:08d}.jpg")
    m = Manifest(tmp_path / "data/manifest.csv")
    cid = "face-chokepoint-P1E_S1__P1E_S1_C1"
    m.upsert({"clip_id": cid, "task": "face", "source": "chokepoint", "path": str(seq), "fps": 10})
    m.save()
    args = ["--config-dir", str(cfg_dir)]

    assert sequence_clips.main(args) == 0
    mp4 = tmp_path / f"data/faces/clips/{cid}.mp4"
    info = probe(mp4)
    assert info["frame_count"] == 6 and abs(info["fps"] - 10) < 0.01 and info["height"] == 600
    rows = list(csv.DictReader(open(mp4.with_suffix(".csv"))))
    assert [r["image"] for r in rows] == [f"{n:08d}.jpg" for n in range(6)] and rows[3]["t_s"] == "0.300"
    assert not list(mp4.parent.glob("*.tmp.*")) and not list(mp4.parent.glob("*.ffconcat"))

    mtime = mp4.stat().st_mtime
    assert sequence_clips.main(args) == 0
    assert mp4.stat().st_mtime == mtime                                   # up to date -> skipped

    assert clean.main([*args, "faces", "--yes"]) == 0
    assert not mp4.exists() and not mp4.with_suffix(".csv").exists()
