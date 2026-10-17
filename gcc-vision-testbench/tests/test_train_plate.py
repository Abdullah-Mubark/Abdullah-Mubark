import yaml

from gccvt import train_plate
from gccvt.config import Settings
from gccvt.manifest import Manifest


def make_dataset(root, names, splits):
    """Roboflow-style YOLO export: {split}/images/*.jpg + {split}/labels/*.txt + data.yaml."""
    (root / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": names}))
    for split, items in splits.items():
        for i, lines in enumerate(items):
            (root / split / "images").mkdir(parents=True, exist_ok=True)
            (root / split / "labels").mkdir(parents=True, exist_ok=True)
            (root / split / "images" / f"{split}{i}.jpg").write_bytes(b"jpg")
            (root / split / "labels" / f"{split}{i}.txt").write_text("\n".join(lines) + "\n")


def test_plate_lines_maps_classes_and_polygons(tmp_path):
    lbl = tmp_path / "a.txt"
    lbl.write_text("50 0.5 0.5 0.2 0.1\n3 0.1 0.1 0.01 0.02\n50 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n")
    names = [str(i) for i in range(50)] + ["plate"]                  # UAE-style: digits + 'plate'
    out = train_plate.plate_lines(lbl, names, {"plate": "plate", "*": ""})
    assert out == ["0 0.500000 0.500000 0.200000 0.100000", "0 0.200000 0.300000 0.200000 0.200000"]


def test_build_dataset_holds_out_eval_split_and_caps(cfg_dir, tmp_path):
    data = tmp_path / "data"
    ksa = data / "plates_images/ksa/ws__ksa__v1"
    uae = data / "plates_images/uae/ws__uae__v2"
    box = ["0 0.5 0.5 0.2 0.1"]
    make_dataset(ksa, ["LP"], {"train": [box] * 5, "valid": [box] * 2, "test": [box] * 3})
    make_dataset(uae, ["7", "plate"], {"train": [["0 0.1 0.1 0.01 0.01", "1 0.5 0.5 0.2 0.1"]] * 2,
                                       "valid": [box]})      # no test split: eval scores on valid
    m = Manifest(data / "manifest.csv")
    for name, root, country in (("ws__ksa__v1", ksa, "ksa"), ("ws__uae__v2", uae, "uae")):
        m.upsert({"clip_id": f"plateimg-{country}-{name}", "task": "plate", "source": "roboflow",
                  "dataset": name, "country": country, "path": str(root)})
    m.save()
    settings = Settings(cfg_dir)
    out = data / "train/plate"
    counts = train_plate.build_dataset(settings, {"max_per_dataset": 4, "max_val_per_dataset": 1},
                                       settings.load("eval.yaml"), out, log=_Log())

    assert counts["ws__ksa__v1"] == {"held_out_for_eval": "test", "train": 4, "val": 1, "plates": 5}
    assert counts["ws__uae__v2"] == {"held_out_for_eval": "valid", "train": 2, "val": 0, "plates": 2}
    assert not any("test" in p.name for p in (out / "images").rglob("*"))           # eval split never used
    uae_lbl = next((out / "labels/train").glob("ws__uae__v2__*.txt")).read_text()
    assert uae_lbl == "0 0.500000 0.500000 0.200000 0.100000\n"                        # digit box dropped
    assert all(p.is_symlink() for p in (out / "images").rglob("*.jpg"))
    assert yaml.safe_load((out / "data.yaml").read_text())["names"] == {0: "plate"}


class _Log:
    def info(self, *a):
        pass
