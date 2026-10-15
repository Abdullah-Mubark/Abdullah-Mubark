# gcc-vision-testbench

Collects and organises test video for evaluating **face detection/recognition** and **licence-plate
recognition** for Saudi and GCC plates (KSA, UAE, Qatar, Kuwait, Oman, Bahrain).

**Status:** the data-collection stage works and is tested. Frame extraction, de-duplication,
pre-annotation, evaluation and the HTML report are the next steps (see [Roadmap](#roadmap)).

> Read [DATA_NOTICE.md](DATA_NOTICE.md) before collecting anything.

## Setup

Requirements: Python 3.11+, `ffmpeg`/`ffprobe` on PATH.

```bash
cd gcc-vision-testbench
make install            # creates .venv, installs requirements, copies .env.example -> .env
# edit .env: set ROBOFLOW_API_KEY (only needed for Roboflow)
make test
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

## Data layout

```
data/
  raw/                       downloads as received (archives, full YouTube videos)
  faces/chokepoint/...       ChokePoint frames + ground truth
  faces/{dataset}/clip_NNNN.*
  plates/{country}/{day|night}/clip_NNNN.mp4 (+ clip_NNNN.json)
  plates_images/{country}/{workspace}__{project}__v{N}/   YOLOv8 export
  frames/                    (prep stage)
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

## Plugging in a model (planned interface)

Evaluation is the next stage. Models will implement a small adapter:

```python
class ModelAdapter(Protocol):
    name: str
    tasks: set[str]                                   # face_det, face_rec, plate_det, plate_ocr
    def load(self, **kwargs) -> None: ...
    def detect(self, frame: np.ndarray) -> list[Box]: ...      # Box: xyxy, score, label
    def read_plate(self, crop: np.ndarray) -> PlateRead: ...   # PlateRead: text, score
    def embed_face(self, crop: np.ndarray) -> np.ndarray: ...  # optional, for recognition
```

## Roadmap

- [x] Collection: ChokePoint, Roboflow, YouTube, local footage, manifest
- [ ] `scripts/prep/`: frame extraction (default 2 fps), perceptual-hash de-duplication,
      optional YOLOv8 / RetinaFace pre-annotation to YOLO/CVAT format
- [ ] `scripts/eval/`: adapter runner; face P/R/mAP and ChokePoint recognition accuracy; plate mAP,
      exact match and CER with Arabic/Latin normalisation (٠-٩ → 0-9); breakdowns by country,
      day/night and clip
- [ ] HTML report with tables, charts and worst-failure examples
