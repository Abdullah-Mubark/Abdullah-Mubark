import json
import math

import numpy as np
import pytest
from PIL import Image

from conftest import edit_yaml
from gccvt.adapters import BaseAdapter, Box, PlateRead, register
from gccvt.config import Settings
from gccvt.eval import gt as G
from gccvt.eval import metrics as M
from gccvt.eval import runner
from gccvt.manifest import Manifest
from gccvt.text_norm import cer, normalize_plate


# ---------------------------------------------------------------- text normalisation
@pytest.mark.parametrize("raw,country,want", [
    ("١٢٣٤ ا ب ح", "ksa", "1234ABJ"),        # Eastern Arabic digits + Arabic letters -> Latin pair
    ("ABJ-1234", "ksa", "1234ABJ"),          # order-insensitive between digit and letter groups
    ("۱۲۳۴ أبح", "ksa", "1234ABJ"),          # Persian digits, hamza alef folded
    ("ﺏ", "ksa", "B"),                       # presentation form -> base letter -> Latin
    ("A 12345", "uae", "12345A"),
    ("١٢٣٤٥ ـهـ", "uae", "12345ه"),          # tatweel removed; no KSA mapping outside KSA
    ("  ", "qatar", ""),
])
def test_normalize_plate(raw, country, want):
    assert normalize_plate(raw, country=country) == want


def test_normalize_options():
    assert normalize_plate("ABJ 1234", canonical_order=False) == "ABJ1234"
    assert normalize_plate("ا ب ح", country="ksa", ksa_arabic_to_latin=False) == "ابح"
    assert cer("1234ABJ", "1234ABJ") == 0 and cer("1234ABJ", "1284AB") == pytest.approx(2 / 7)


# ---------------------------------------------------------------- metrics
def test_match_and_ap():
    gts = [{"xyxy": [0, 0, 10, 10], "label": "plate"}, {"xyxy": [20, 20, 30, 30], "label": "plate"}]
    preds = [{"xyxy": [0, 0, 10, 10], "score": 0.9, "label": "plate"},
             {"xyxy": [1, 1, 10, 10], "score": 0.95, "label": "plate"},     # duplicate of GT 0, higher score
             {"xyxy": [50, 50, 60, 60], "score": 0.3, "label": "plate"}]
    p2g, g2p = M.match(preds, gts, 0.5)
    assert p2g == [-1, 0, -1] and g2p == [1, -1]
    assert M.average_precision([0.9, 0.8], [True, True], 2) == pytest.approx(1.0)
    assert M.average_precision([0.9, 0.8], [False, True], 1) == pytest.approx(0.5)
    assert math.isnan(M.average_precision([], [], 0))
    m = M.detection_metrics([{"preds": preds, "gt": gts}], "plate", 0.5, 0.5, [0.5, 0.75])
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["ap"] == pytest.approx(0.5, abs=0.01)


def test_ocr_and_recognition_metrics():
    o = M.ocr_metrics([("1234ABJ", "1234ABJ"), ("5678ABJ", "5678AB"), ("99", "")])
    assert o["exact_match"] == pytest.approx(1 / 3)
    assert o["cer"] == pytest.approx((0 + 1 + 2) / 16)
    assert o["digits_exact_match"] == pytest.approx(2 / 3)
    g = {"a": np.array([1.0, 0, 0]), "b": np.array([0, 1.0, 0])}
    r = M.identification(g, [("a", np.array([0.9, 0.1, 0])), ("b", np.array([0.8, 0.2, 0])), ("c", np.ones(3))])
    assert r["rank1"] == 0.5 and r["probes"] == 2 and r["probes_without_gallery"] == 1
    assert 0 <= r["eer"] <= 1


def test_eyes_to_box():
    box = G.eyes_to_box(40, 40, 60, 40, {"width": 2.0, "height": 2.4, "eye_line": 0.4})
    assert box == pytest.approx([30, 20.8, 70, 68.8])


# ---------------------------------------------------------------- end to end
@register("bright_box")
class BrightBox(BaseAdapter):
    """Finds bright rectangles separated by dark columns."""
    def detect(self, frame):
        mask = frame.max(axis=2) > 60
        cols = np.where(mask.any(0))[0]
        boxes = []
        if not len(cols):
            return boxes
        runs = np.split(cols, np.where(np.diff(cols) > 1)[0] + 1)
        for run in runs:
            ys = np.where(mask[:, run].any(1))[0]
            boxes.append(Box(run[0], ys[0], run[-1] + 1, ys[-1] + 1, 0.9, "x"))
        return boxes


@register("fixed_reader")
class FixedReader(BaseAdapter):
    def read_plate(self, crop):
        return PlateRead("١٢٣٤ ا ب ح", 0.8)


@register("color_embed")
class ColorEmbed(BaseAdapter):
    def embed_face(self, crop):
        v = crop.reshape(-1, 3).mean(0)
        if v.max() < 10:
            raise ValueError("no face")
        return v.astype(np.float32)


def _img(path, rects, size=(200, 120)):
    a = np.zeros((size[1], size[0], 3), np.uint8)
    for (x1, y1, x2, y2), rgb in rects:
        a[int(y1):int(y2), int(x1):int(x2)] = rgb
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)


def _cvat(path, images):
    xml = ['<?xml version="1.0"?><annotations><version>1.1</version>']
    for name, boxes in images:
        xml.append(f'<image id="0" name="{name}" width="200" height="120">')
        for (x1, y1, x2, y2), text in boxes:
            attr = f'<attribute name="text">{text}</attribute>' if text else ""
            xml.append(f'<box label="plate" xtl="{x1}" ytl="{y1}" xbr="{x2}" ybr="{y2}">{attr}</box>')
        xml.append("</image>")
    xml.append("</annotations>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(xml), encoding="utf-8")


def _chokepoint_seq(data, name, people):
    """people: [(person_id, (lx, ly, rx, ry), rgb)] -> 2 frames with eye-derived boxes drawn."""
    folder = data / "frames" / f"face-chokepoint-{name}"
    xml = ["<dataset>"]
    for fi in range(2):
        rects = []
        xml.append(f'<frame number="{fi:08d}">')
        for pid, (lx, ly, rx, ry), rgb in people:
            rects.append((G.eyes_to_box(lx, ly, rx, ry, {}), rgb))
            xml.append(f'<person id="{pid}"><leftEye x="{lx}" y="{ly}"/><rightEye x="{rx}" y="{ry}"/></person>')
        xml.append("</frame>")
        _img(folder / f"{fi:08d}.png", rects)
    xml.append("</dataset>")
    gt = data / "faces/chokepoint/groundtruth" / f"{name}.xml"
    gt.parent.mkdir(parents=True, exist_ok=True)
    gt.write_text("".join(xml))
    return {"clip_id": f"face-chokepoint-{name}", "task": "face", "source": "chokepoint", "dataset": "chokepoint",
            "path": str(data / "faces/chokepoint" / name), "gt_path": str(gt), "fps": 30}


@pytest.fixture
def eval_setup(cfg_dir, tmp_path):
    data = tmp_path / "data"
    m = Manifest(data / "manifest.csv")
    # Plate clip: 0 exact read, 1 misread, 2 missed (too dark), 3 false positive (no GT)
    pf = data / "frames/plate-ksa-day-clip_0001"
    plate = (20, 20, 100, 50)
    _img(pf / "000000.png", [(plate, (255, 255, 255))])
    _img(pf / "000001.png", [(plate, (255, 255, 255))])
    _img(pf / "000002.png", [(plate, (30, 30, 30))])
    _img(pf / "000003.png", [(plate, (255, 255, 255))])
    _cvat(data / "annotations/plate-ksa-day-clip_0001/gt/annotations.xml", [
        ("000000.png", [(plate, "1234 ABJ")]), ("000001.png", [(plate, "5678 ABJ")]),
        ("000002.png", [(plate, "1234 ABJ")]), ("000003.png", [])])
    m.upsert({"clip_id": "plate-ksa-day-clip_0001", "task": "plate", "source": "youtube", "country": "ksa",
              "condition": "day", "path": str(data / "plates/ksa/day/clip_0001.mp4")})
    # Unlabelled clip -> skipped
    _img(data / "frames/plate-uae-night-clip_0001/000000.png", [])
    m.upsert({"clip_id": "plate-uae-night-clip_0001", "task": "plate", "source": "youtube", "country": "uae",
              "condition": "night", "path": "x.mp4"})
    # ChokePoint: gallery + probe, two people with distinct colours (positions swapped in probe)
    a, b = ("0001", (255, 40, 40)), ("0002", (40, 255, 40))
    m.upsert(_chokepoint_seq(data, "P1E_S1_C1", [(a[0], (40, 40, 60, 40), a[1]), (b[0], (130, 40, 150, 40), b[1])]))
    m.upsert(_chokepoint_seq(data, "P1E_S2_C1", [(b[0], (40, 40, 60, 40), b[1]), (a[0], (130, 40, 150, 40), a[1])]))
    m.save()
    edit_yaml(cfg_dir / "models.yaml", lambda c: c.update(default="fake", models={"fake": {
        "plate_detector": {"adapter": "bright_box"}, "plate_reader": {"adapter": "fixed_reader"},
        "face_detector": {"adapter": "bright_box"}, "face_recognizer": {"adapter": "color_embed"}}}))
    return data


def test_run_eval_end_to_end(cfg_dir, eval_setup):
    data = eval_setup
    assert runner.main(["--config-dir", str(cfg_dir), "--run-id", "r1"]) == 0
    run = data / "results/r1"
    m = json.loads((run / "metrics.json").read_text())
    assert m["unlabeled_clips"] == ["plate-uae-night-clip_0001"]

    pd = m["detection"]["plate"]["overall"]
    assert (pd["tp"], pd["fp"], pd["fn"]) == (2, 1, 1)
    assert pd["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert set(m["detection"]["plate"]["by"]["country"]) == {"ksa"}

    fd = m["detection"]["face"]["overall"]
    assert (fd["tp"], fd["fp"], fd["fn"]) == (8, 0, 0) and fd["ap"] == pytest.approx(1.0)

    e2e = m["plate_ocr"]["end_to_end"]["overall"]
    rec = m["plate_ocr"]["recognition_only"]["overall"]
    assert e2e["plates"] == 3 and e2e["exact_match"] == pytest.approx(1 / 3, abs=1e-4)
    assert rec["exact_match"] == pytest.approx(2 / 3, abs=1e-4)

    fr = m["face_recognition"]
    assert fr["gallery_identities"] == 2
    assert fr["frame_level"]["rank1"] == 1.0 and fr["track_level"]["probes"] == 2

    f = json.loads((run / "failures.json").read_text())
    assert f["counts"] == {"missed_detections": 1, "false_positives": 1, "plate_misreads": 2}
    worst = f["plate_misreads"][0]
    assert worst["raw_ref"] == "1234 ABJ" and worst["detected"] is False and worst["cer"] == 1.0
    assert all((run / x["crop"]).is_file() for k in ("missed_detections", "false_positives", "plate_misreads")
               for x in f[k])

    # Rescore with a stricter threshold, no inference
    edit_yaml(cfg_dir / "eval.yaml", lambda c: c["detection"].update(score_threshold=0.95))
    assert runner.main(["--config-dir", str(cfg_dir), "--rescore", "r1"]) == 0
    m2 = json.loads((run / "metrics.json").read_text())
    assert m2["detection"]["plate"]["overall"]["tp"] == 0
    assert m2["detection"]["plate"]["overall"]["ap"] == pd["ap"]


def test_run_eval_unknown_model(cfg_dir, eval_setup):
    from gccvt.errors import ConfigError
    with pytest.raises(ConfigError, match="not in config/models.yaml"):
        runner.main(["--config-dir", str(cfg_dir), "--model", "nope"])


def test_roboflow_gt(cfg_dir, tmp_path):
    root = tmp_path / "data/plates_images/ksa/ds"
    _img(root / "valid/images/a.png", [])
    (root / "valid/labels").mkdir(parents=True)
    (root / "valid/labels/a.txt").write_text("0 0.5 0.5 0.2 0.5\n1 0.1 0.1 0.1 0.1\n")
    (root / "data.yaml").write_text("names: [license-plate, car]\n")
    s = Settings(config_dir=cfg_dir)
    row = {"clip_id": "x", "source": "roboflow", "path": str(root)}
    gt = G.load_gt(s, row, {"roboflow": {"splits": ["test", "valid"], "class_map": {"car": "", "*": "plate"}}})
    boxes = gt[root / "valid/images/a.png"]
    assert len(boxes) == 1 and boxes[0]["label"] == "plate"
    assert boxes[0]["xyxy"] == pytest.approx([80, 30, 120, 90])


# ---------------------------------------------------------------- report
def test_report_with_clips(cfg_dir, eval_setup):
    from gccvt.media import probe
    from gccvt.report import build
    data = eval_setup
    assert runner.main(["--config-dir", str(cfg_dir), "--run-id", "r2"]) == 0
    assert build.main(["--config-dir", str(cfg_dir)]) == 0  # latest run
    run = data / "results/r2"
    html = (run / "report.html").read_text()
    for needle in ("Plate detection AP", "Plate exact match by country", "Worst failures", "<video",
                   "clips/plate-ksa-day-clip_0001.mp4", "١٢٣٤ ا ب ح", "Face recognition"):
        assert needle in html, needle
    clip = probe(run / "clips/plate-ksa-day-clip_0001.mp4")
    assert clip["frame_count"] == 4 and clip["height"] == 120
    assert probe(run / "clips/face-chokepoint-P1E_S2_C1.mp4")["frame_count"] == 2
    assert (run / "clips/plate-ksa-day-clip_0001.jpg").is_file()  # poster
    assert 'poster="clips/plate-ksa-day-clip_0001.jpg"' in html
    fails = sorted(p.name for p in (run / "clips/failures").glob("*.mp4"))
    assert fails == ["false_positives_000.mp4", "missed_detections_000.mp4",
                     "plate_misreads_000.mp4", "plate_misreads_001.mp4"]
    assert 'src="clips/failures/missed_detections_000.mp4"' in html

    assert build.main(["--config-dir", str(cfg_dir), "--run", "r2", "--no-clips"]) == 0
    assert "<video" not in (run / "report.html").read_text()


def test_report_no_runs(cfg_dir):
    from gccvt.errors import GccvtError
    from gccvt.report import build
    with pytest.raises(GccvtError, match="make eval"):
        build.main(["--config-dir", str(cfg_dir)])
