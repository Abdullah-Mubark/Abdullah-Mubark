# gcc-vision-testbench

Collects and organises test video for evaluating **face detection/recognition** and **licence-plate
recognition** for Saudi and GCC plates (KSA, UAE, Qatar, Kuwait, Oman, Bahrain).

**Status:** all four stages work and are tested: collection, preprocessing, evaluation, and
an HTML report with annotated video clips.

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

Add datasets to `config/roboflow_datasets.yaml` using any of these forms. Leave the version out to
get the latest one:

```yaml
datasets:
  - url: "https://universe.roboflow.com/<workspace>/<project>"            # latest version
    country: ksa
  - url: "https://universe.roboflow.com/<workspace>/<project>/dataset/3"  # pinned version
    country: uae
  - slug: "<workspace>/<project>[/<version>]"
    country: qatar
```

The shipped config already lists public Saudi and UAE plate datasets found on Universe. No usable
Qatar, Kuwait, Oman or Bahrain datasets were found; the config names the workspaces to look in.

Exports are in YOLOv8 format (`format: yolov8`). If Roboflow still has to generate an export, the
script polls until it is ready. Image datasets are listed in the manifest as one row each
(`frame_count` = number of images).

### YouTube

The shipped config lists 15 real day and night driving videos, covering all six countries. Their
segment times are placeholders (02:00–03:00): preview each video and move the segment to a stretch
where plates are readable.

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

## Evaluation

```bash
make eval                                   # default model from config/models.yaml
make eval ARGS="--model my_model --country ksa --limit-frames 50"
make eval ARGS="--rescore baseline_20261015-120000"   # recompute metrics, no inference
```

### Ground truth

Only frames that have ground truth are evaluated. Clips without it are listed as
`unlabeled_clips` in `metrics.json`.

| Source | Where the ground truth comes from |
|---|---|
| YouTube / local clips | `data/annotations/{clip_id}/gt/`, either `annotations.xml` (CVAT for images 1.1) or `labels/*.txt` + `classes.txt` (YOLO) |
| ChokePoint | its ground-truth XML (eye positions), converted to face boxes |
| Roboflow | YOLO labels from the export's `test` split (else `valid`), using `roboflow.class_map` |

Labelling workflow for a clip:

1. `make prep` with a detector enabled, which writes draft labels.
2. Create a CVAT task from `data/frames/{clip_id}/` and upload `cvat/annotations.xml` from the
   draft folder.
3. Correct the boxes. On every plate box, add a text attribute named `text` with the plate as you
   read it, e.g. `1234 ABJ` or `١٢٣٤ ا ب ح`. On face boxes, add `identity` if you want
   recognition results.
4. Export as "CVAT for images 1.1" and save it as `data/annotations/{clip_id}/gt/annotations.xml`.

YOLO ground truth gives detection metrics only, because it has nowhere to store plate text.

### Metrics

- **Detection** (plates and faces):
  - Precision and recall at `detection.score_threshold`.
  - AP at the task IoU (0.5) and COCO mAP@[.5:.95], using greedy matching by score and COCO
    101-point interpolation.
  - ChokePoint face boxes are built from the eye positions (`faces.chokepoint.box_from_eyes`), so
    they are approximate and matched at a looser IoU (`faces.chokepoint.iou`, default 0.4).
- **Plate OCR**, in two modes:
  - *end-to-end*: the reader runs on detected plates, so a missed plate counts as an empty read.
  - *recognition-only*: the reader runs on the ground-truth crops, which isolates OCR from
    detection.

  Both report exact-match accuracy, character error rate (CER: total edits ÷ total reference
  characters), mean per-plate CER, and digits-only exact match.
- **Plate text normalisation** (`plates.normalize`; affects scoring only):
  - Eastern Arabic and Persian digits become 0-9.
  - NFKC folds Arabic presentation forms to base letters.
  - Spaces, separators, tatweel, diacritics and bidi marks are removed; Latin is uppercased.
  - Alef forms, ya/alef maqsura and ta marbuta are folded together.
  - For KSA, Arabic letters are mapped to their official Latin pair (ا→A, ب→B, ح→J, … ى→V), so
    reading either row of the plate counts.
  - Digits come first, then letters, so `ABJ 1234`, `1234 ABJ` and `١٢٣٤ ا ب ح` are all equal.
- **Face recognition** (ChokePoint):
  - Each person's gallery template is the mean embedding over frames from the `gallery`
    sequences.
  - Faces in the `probe` sequences are identified against those templates. The script reports
    rank-1 accuracy per frame and per track (mean embedding per sequence and person), EER, and
    TAR at a 1% false-accept rate.
  - Crops come from the ground-truth boxes, so recognition is measured separately from detection.
  - The default gallery/probe split is only a starting point. Set it to the split you report
    against, such as the G1/G2 groups in the ChokePoint paper.
- **Breakdowns** of every metric by country, day/night, source and clip (`breakdown:` in
  `config/eval.yaml`).

### Output: `data/results/{run_id}/`

- `run.json`: model and settings snapshot.
- `predictions.jsonl`: per frame, the ground truth, detections and plate reads, with raw model
  text.
- `embeddings.npz`: face embeddings for the recognition protocol.
- `metrics.json`: overall results plus breakdowns.
- `failures.json` + `failures/*.jpg`: the worst missed detections (largest first), the
  highest-scoring false positives, and plate misreads (highest CER first), each with a crop.
  Ground-truth boxes are outlined in red, false positives in orange.

Models are defined in `config/models.yaml`. Each one has up to four roles: `plate_detector`,
`plate_reader`, `face_detector`, `face_recognizer`. A role is an adapter plus its params, and
roles with identical settings share one loaded instance. Keep the detector confidence low there
(e.g. 0.05), because AP needs the low-scoring detections too.

## Report and annotated clips

```bash
make report                          # latest eval run
make report ARGS="--run baseline_20261015-120000"
make report ARGS="--no-clips"        # tables and charts only (fast)
```

This writes `data/results/{run_id}/report.html` next to the clips. Open it from that folder,
or copy the whole folder to share it.

**Annotated clips** (`clips/{clip_id}.mp4`): every evaluated frame of a clip, encoded as H.264 MP4
with the following drawn on:

| Colour | Tag | Meaning |
|---|---|---|
| Green | the GT text | ground truth that was detected |
| Blue | score, plate read, ✓/✗ | detection matched to ground truth |
| Red | `MISSED` | ground truth with no detection |
| Orange | `FALSE +` | detection with no ground truth |

Plate text is drawn with a font that has Arabic glyphs (`report.fonts`). With Pillow's raqm
support it is also shaped correctly.

**Failure clips** (`clips/failures/*.mp4`): a few frames before and after each of the worst missed
detections, false positives and misreads, played at half speed. The failure is outlined in yellow.

**The report page** contains:
- headline numbers;
- day-vs-night charts per country (detection AP, recall, plate exact match, CER), each with a
  table view;
- every clip's video and a per-clip results table, worst clip first;
- the failure gallery;
- full breakdown tables;
- the settings used.

It supports light and dark mode and works on a phone. The playback speed, size and frame cap
for clips are under `report:` in `config/eval.yaml`.

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
  annotations/{clip_id}/gt/  corrected ground truth (CVAT XML or YOLO) used by evaluation
  results/{run_id}/          evaluation runs: metrics, predictions, report.html, clips/*.mp4
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

Pre-annotation and evaluation use one small interface in `src/gccvt/adapters/base.py`.
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

    def read_plate(self, crop) -> PlateRead:      # plate text, raw (normalised only for scoring)
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
- [x] Eval: adapter runner; face/plate P/R/AP/mAP; plate exact match, CER and digit accuracy with
      Arabic/Latin normalisation; ChokePoint rank-1/EER; breakdowns; worst-failure crops
- [x] HTML report: tables, day/night charts, annotated clips, failure clips
