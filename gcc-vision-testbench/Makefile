# gcc-vision-testbench
# Override the interpreter with: make collect PY=.venv/bin/python
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
ARGS ?=

.PHONY: help venv install collect collect-faces collect-roboflow collect-youtube collect-local \
        manifest fetch pack recompress prep extract dedupe preannotate face-clips train-plate install-models eval report compare test clean-data clean-logs

help:
	@echo "make install          create .venv and install requirements"
	@echo "make collect          run all collectors (faces, roboflow, youtube, local)"
	@echo "make collect-faces    ChokePoint from Zenodo"
	@echo "make collect-roboflow Roboflow Universe plate datasets"
	@echo "make collect-youtube  YouTube segments -> MP4 clips (4K by default)"
	@echo "make collect-local    your own footage (config/local_sources.yaml)"
	@echo "make manifest         summary of data/manifest.csv"
	@echo "make fetch            new machine: all data in one go (unpacks gccvt-transfer.tgz first if present)"
	@echo "make pack             gccvt-transfer.tgz: manifest, gt labels, results (no videos) for make fetch"
	@echo "make recompress       re-encode existing YouTube clips with the current video settings (smaller)"
	@echo "make prep             extract frames, dedupe, pre-annotate (if detectors enabled)"
	@echo "make extract / dedupe / preannotate   run one prep step"
	@echo "make face-clips       encode image-sequence clips (ChokePoint) as MP4 in data/faces/clips/"
	@echo "make install-models   optional detector dependencies (ultralytics, insightface)"
	@echo "make train-plate      train a YOLOv8 plate detector on the Roboflow datasets -> models/plate_yolov8.pt"
	@echo "make eval             run the default model (config/models.yaml) on labelled clips"
	@echo "make report           HTML report + annotated video clips for the latest eval run"
	@echo "make compare ARGS='RUN_A RUN_B'   what differed between two runs, and the metric changes"
	@echo "make test             run unit tests"
	@echo "make clean-data       show disk use under data/; ARGS=\"raw frames --yes\" deletes (see --help)"
	@echo "Pass extra flags with ARGS, e.g. make collect-youtube ARGS='--dry-run --country ksa'"

venv:
	python3.11 -m venv .venv || python3 -m venv .venv

install: venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements.txt
	@test -f .env || (cp .env.example .env && echo "Created .env - add your ROBOFLOW_API_KEY")

install-models: venv
	.venv/bin/pip install -r requirements-models.txt

collect: collect-faces collect-roboflow collect-youtube collect-local manifest

collect-faces:
	$(PY) scripts/collect/download_chokepoint.py $(ARGS)

collect-roboflow:
	$(PY) scripts/collect/download_roboflow.py $(ARGS)

collect-youtube:
	$(PY) scripts/collect/download_youtube.py $(ARGS)

collect-local:
	$(PY) scripts/collect/local_import.py $(ARGS)

manifest:
	$(PY) scripts/collect/manifest_summary.py

fetch:
	$(PY) scripts/fetch_data.py $(ARGS)

pack:
	$(PY) scripts/fetch_data.py --pack $(ARGS)

recompress:
	$(PY) scripts/recompress.py $(ARGS)

prep: extract dedupe preannotate

extract:
	$(PY) scripts/prep/extract_frames.py $(ARGS)

dedupe:
	$(PY) scripts/prep/dedupe.py $(ARGS)

preannotate:
	$(PY) scripts/prep/preannotate.py $(ARGS)

face-clips:
	$(PY) scripts/prep/sequence_clips.py $(ARGS)

train-plate:
	$(PY) scripts/train_plate.py $(ARGS)

eval:
	$(PY) scripts/eval/run_eval.py $(ARGS)

report:
	$(PY) scripts/eval/build_report.py $(ARGS)

compare:
	$(PY) scripts/eval/compare_runs.py $(ARGS)

test:
	$(PY) -m pytest -q

clean-data:
	$(PY) scripts/clean_data.py $(ARGS)

clean-logs:
	rm -rf logs
