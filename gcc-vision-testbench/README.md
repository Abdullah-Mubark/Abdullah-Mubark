# gcc-vision-testbench

Collects and organises test video for evaluating **face detection/recognition** and **licence-plate
recognition** for Saudi and GCC plates (KSA, UAE, Qatar, Kuwait, Oman, Bahrain).

**Status:** collection and preprocessing (frame extraction, de-duplication, pre-annotation) work
and are tested. Evaluation and the HTML report are next (see [Roadmap](#roadmap)).

> Read [DATA_NOTICE.md](DATA_NOTICE.md) before collecting anything.

## Setup

Requirements: Python 3.11+, `ffmpeg`/`ffprobe` on PATH.

```bash
cd gcc-vision-testbench
make install            # creates .venv, installs requirements, copies .env.example -> .env
# edit .env: set ROBOFLOW_API_KEY (only needed for Roboflow)
make test
make install-models   # optional: detectors for pre-annotation
```

## Collecting data

| Command | Source | Config | Output |
|---|---|---|---|
| `make collect-faces` | ChokePoint (Zenodo) | `config/chokepoint.yaml` | `data/faces/chokepoint/...` |
| `make collect-roboflow` | Roboflow Universe | `config/roboflow_datasets.yaml` | `data/plates_images/{country}/{dataset}/` |
| `make collect-youtube` | YouTube | `config/youtube_sources.yaml` | `data/plates/{country}/{day\|night}/clip_NNNN.mp4` |
| `make collect-local` | your footage | `config/local_sources.yaml` | `data/plates/...` or `data/faces/{dataset}/` |
| `make collect` | all of the above, then a manifest summary | | |
| `make manifest` | prints counts/minutes by task, country and condition | | |

Every script accepts `--dry-run`, `--config-dir` and `--log-level`; pass them through make with
`ARGS`, e.g. `make collect-youtube ARGS="--dry-run --country ksa"`. Run any script with `--help`
for its own options. Logs go to the console and to `logs/<script>.log`.

All collectors are **safe to re-run**: finished items are skipped and interrupted downloads resume.
Failures are retried with exponential backoff; a failed item is reported and the rest still run.

### ChokePoint

```bash
python scripts/collect/download_chokepoint.py --list            # see files and sizes first
python scripts/collect/download_chokepoint.py --only "P1E*" "groundtruth*"
```

Files are fetched one at a time with HTTP Range resume, checked against Zenodo's md5 checksums,
and extracted. Each camera sequence (a folder of JPEG frames) becomes one row in the manifest, linked
to its ground-truth XML. The full dataset is large, so restrict `files.include` if disk is tight.
`delete_archives_after_extract: true` saves space, and the archives won't be downloaded again.

### Roboflow Universe

Add datasets to `config/roboflow_datasets.yaml` using any of these forms:

```yaml
datasets:
  - url: "https://universe.roboflow.com/<workspace>/<project>/dataset/<version>"
    country: ksa
  - slug: "<workspace>/<project>/<version>"
    country: uae
```

Exports are in YOLOv8 format (`format: yolov8`). If Roboflow still has to generate an export, the
script polls until it is ready. Image datasets are listed in the manifest as one row each
(`frame_count` = number of images).

### YouTube

```yaml
sources:
  - url: "https://www.youtube.com/watch?v=XXXXXXXXXXX"
    country: ksa            # or alias: saudi, sa
    condition: night        # day | night
    start: "00:01:30"       # HH:MM:SS, MM:SS or seconds
    end: "00:02:15"
    notes: "Riyadh, dashcam"
  - url: "https://youtu.be/YYYYYYYYYYY"
    country: qatar
    segments:               # several clips from one video
      - {start: "10:00", end: "10:45", condition: day}
      - {start: 1520, end: 1580, condition: night}
```

- **`download_mode: section`** (default) asks yt-dlp for just the segment.
- **`download_mode: full`** downloads the whole video once to `data/raw/youtube/full/` and reuses
  it for every segment. Use it when you cut many clips from one video.
- **Encoding.** ffmpeg re-encodes each clip to H.264 MP4 at 1080p.
  - With `video.allow_upscale: false`, lower-resolution sources keep their resolution, and the
    script logs a warning.
  - Audio is dropped unless `video.keep_audio: true`.
- **Metadata.** A JSON file is saved beside each clip: source URL, video id and title, country,
  condition, start/end, duration, resolution and fps.
- **Sign-in prompts.** If YouTube asks you to sign in, set `defaults.cookies_from_browser: chrome`
  (or your browser).

### Your own footage

```bash
# folders named by country/condition are detected automatically: footage/ksa/night/*.mp4
python scripts/collect/local_import.py --src ~/footage --consent-ack
# or be explicit
python scripts/collect/local_import.py --src ~/riyadh_run --country ksa --condition day --mode transcode --consent-ack
python scripts/collect/local_import.py --src ~/door_cam --task face --dataset office_door --consent-ack
```

Modes: `copy` (default, keeps the original format), `link` (hard link or symlink, no extra disk
space), `transcode` (re-encode to 1080p MP4). Files are de-duplicated by SHA-256. You can also list
sources in `config/local_sources.yaml` and run `make collect-local`.

## Preprocessing

```bash
make prep                 # = make extract dedupe preannotate
make extract ARGS="--fps 5 --country ksa"
make dedupe ARGS="--threshold 4"
make dedupe ARGS="--undo"  # put all duplicates back
```

All settings are in `config/prep.yaml`. Every step accepts `--clip`, `--task`, `--country`,
`--condition`, `--force` and `--dry-run`. Steps are skipped when nothing has changed, and
re-run automatically when their settings or input change.

### Frame extraction

Takes every clip in the manifest at `extract.fps` (default 2, never more than the source fps).
Output goes to `data/frames/{clip_id}/000000.jpg`. Each clip folder gets an `index.csv` with the
frame name, timestamp and source frame number. ChokePoint image sequences are subsampled and
hard-linked rather than re-encoded. They keep their original file names so the ground truth still
matches. Roboflow image datasets are skipped because they're already images.

### De-duplication

Each frame gets a perceptual hash (`phash` or `dhash`). A frame counts as a duplicate when its
hash is within `threshold` bits of one of the last `window` kept frames. Duplicates are recorded
in `index.csv` (`status`, `dup_of`), and `action` decides what happens to them:

- `move` (default): moved to `_duplicates/`. If you change the settings and re-run, they are
  restored and checked again.
- `delete`: removed for good.
- `mark`: left in place, recorded in `index.csv` only.

Dedupe is **off for ChokePoint** by default (`dedupe.overrides`). With a fixed camera and small
faces, a whole-frame hash barely changes when a person walks past, so it would drop real frames.
Turn it on or off per source or per task in the same place.

### Pre-annotation (optional)

Runs detectors over the kept frames and writes draft labels for correction in CVAT to
`data/annotations/{clip_id}/`:

- `yolo/labels/*.txt`, `yolo/classes.txt`, `yolo/images.txt`: YOLO format. Class ids follow
  `preannotate.classes`.
- `cvat/annotations.xml`: CVAT for images 1.1. Import it into a CVAT task built from the same
  frames; boxes are marked `source="auto"`.
- `predictions.jsonl`: every box with its confidence score.

To turn it on:

1. Run `make install-models`.
2. Set `enabled: true` on a detector in `config/prep.yaml`.
3. For plates, point `weights:` at a YOLOv8 plate model, for example one trained on your
   Roboflow datasets. No official GCC plate weights exist.
4. For faces, use `adapter: insightface` (RetinaFace family) or `adapter: yolov8` with
   YOLOv8-face weights.

## Data layout
## Data layout

```
data/
  raw/                       downloads as received (archives, full YouTube videos)
  faces/chokepoint/...       ChokePoint frames + ground truth
  faces/{dataset}/clip_NNNN.*
  plates/{country}/{day|night}/clip_NNNN.mp4 (+ clip_NNNN.json)
  plates_images/{country}/{workspace}__{project}__v{N}/   YOLOv8 export
  frames/{clip_id}/          extracted frames + index.csv (+ _duplicates/)
  annotations/{clip_id}/     draft labels: yolo/, cvat/annotations.xml, predictions.jsonl
  manifest.csv
```

`manifest.csv` columns: `clip_id, task, source, dataset, country, condition, path, duration_s, fps,
width, height, frame_count, source_url, source_key, license, gt_path, meta_path, sha256, added_at,
notes`. Paths are relative to the project root. Set `GCCVT_DATA_ROOT` in `.env` to keep data on
another disk.

## Configuration

Everything lives in `config/`. The code has no hard-coded URLs or paths.

- `settings.yaml`: paths, country list and aliases, retry/timeout settings, video encoding.
- `chokepoint.yaml`, `roboflow_datasets.yaml`, `youtube_sources.yaml`, `local_sources.yaml`: sources.
- `.env`: secrets (`ROBOFLOW_API_KEY`) and optional overrides (`GCCVT_DATA_ROOT`,
  `GCCVT_CONFIG_DIR`, `LOG_LEVEL`).

## Adding a new kind of source

Add a module in `src/gccvt/collect/` with a `main(argv)` function, and a wrapper in
`scripts/collect/`. Reuse the shared pieces:

- `HttpClient` (`net.py`): retries and resumable, verified downloads.
- `media.transcode` / `media.probe`: encoding and probing.
- `next_clip_path` and `Manifest.upsert` (`manifest.py`): clip naming and manifest rows.

Give each item a stable `source_key` so re-runs skip it.

## Plugging in a model

Pre-annotation (and later evaluation) uses one small interface in `src/gccvt/adapters/base.py`.
Frames are numpy arrays, HxWx3 uint8, in **BGR** order (OpenCV convention).

```python
from gccvt.adapters import BaseAdapter, Box, PlateRead, register

@register("my_plate_model")
class MyPlateModel(BaseAdapter):
    tasks = {"plate_det", "plate_ocr"}            # face_det, face_rec, plate_det, plate_ocr

    def load(self, weights, conf=0.3, **kwargs):  # kwargs = `params:` from the config
        self.model = ...

    def detect(self, frame) -> list[Box]:         # Box(x1, y1, x2, y2, score, label, meta)
        return [Box(x1, y1, x2, y2, score, "plate") for ...]

    def read_plate(self, crop) -> PlateRead:      # used by evaluation
        return PlateRead(text="أ ب ج 1234", score=0.9)

    def embed_face(self, crop):                   # optional, for face recognition
        ...
```

You only implement the methods your model supports; the rest raise `NotImplementedError`.
To use it, reference it in config as `adapter: "mypackage.my_module:MyPlateModel"`, or add it to
`_BUILTIN` in `adapters/registry.py`. There are two built-in adapters:

- `yolov8`: any Ultralytics model. AGPL-3.0.
- `insightface`: RetinaFace-family face detection plus ArcFace embeddings. The pretrained packs
  are for non-commercial use only.

## Roadmap

- [x] Collection: ChokePoint, Roboflow, YouTube, local footage, manifest
- [x] Prep: frame extraction, perceptual-hash de-duplication, YOLO/CVAT pre-annotation
- [ ] `scripts/eval/`: adapter runner; face P/R/mAP and ChokePoint recognition accuracy; plate mAP,
      exact match and CER with Arabic/Latin normalisation (٠-٩ → 0-9); breakdowns by country,
      day/night and clip
- [ ] HTML report with tables, charts and worst-failure examples
