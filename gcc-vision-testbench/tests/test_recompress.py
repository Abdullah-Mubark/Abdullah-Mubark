import json
import subprocess

from conftest import edit_yaml, make_video
from gccvt import recompress
from gccvt.manifest import Manifest
from gccvt.media import probe


def test_recompress_h264_60fps_to_hevc_30fps(cfg_dir, tmp_path):
    data = tmp_path / "data"
    clip = data / "plates/ksa/day/clip_0001.mp4"
    make_video(clip, seconds=2, fps=60)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(clip), "-c:v", "libx264",
                    str(clip.with_name("h.mp4"))], check=True)
    clip.with_name("h.mp4").replace(clip)                       # an H.264 60 fps clip, like before
    meta = clip.with_suffix(".json")
    meta.write_text(json.dumps({"fps": 60, "video_id": "abc"}))
    m = Manifest(data / "manifest.csv")
    cid = "plate-ksa-day-clip_0001"
    m.upsert({"clip_id": cid, "task": "plate", "source": "youtube", "country": "ksa", "condition": "day",
              "path": str(clip), "meta_path": str(meta), "fps": 60, "frame_count": 120})
    m.upsert({"clip_id": "plate-ksa-day-clip_0002", "task": "plate", "source": "local", "path": str(clip)})
    m.save()
    edit_yaml(cfg_dir / "settings.yaml", lambda c: c["video"].update(preset="ultrafast", target_height=360))
    args = ["--config-dir", str(cfg_dir)]

    assert recompress.main(args) == 0
    info = probe(clip)
    assert info["codec"] == "hevc" and info["fps"] == 30 and info["height"] == 360
    row = Manifest(data / "manifest.csv").rows[cid]
    assert row["fps"] == "30" and row["frame_count"] == "60" and row["source"] == "youtube"
    assert json.loads(meta.read_text()) | {"fps": 30, "codec": "hevc", "video_id": "abc"} == json.loads(meta.read_text())
    assert not list(clip.parent.glob("*.tmp.*"))

    mtime = clip.stat().st_mtime
    assert recompress.main(args) == 0                           # already HEVC <=30 fps: skipped
    assert clip.stat().st_mtime == mtime
