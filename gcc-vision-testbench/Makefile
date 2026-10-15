# gcc-vision-testbench
# Override the interpreter with: make collect PY=.venv/bin/python
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
ARGS ?=

.PHONY: help venv install collect collect-faces collect-roboflow collect-youtube collect-local \
        manifest prep eval report test clean-logs

help:
	@echo "make install          create .venv and install requirements"
	@echo "make collect          run all collectors (faces, roboflow, youtube, local)"
	@echo "make collect-faces    ChokePoint from Zenodo"
	@echo "make collect-roboflow Roboflow Universe plate datasets"
	@echo "make collect-youtube  YouTube segments -> 1080p clips"
	@echo "make collect-local    your own footage (config/local_sources.yaml)"
	@echo "make manifest         summary of data/manifest.csv"
	@echo "make test             run unit tests"
	@echo "Pass extra flags with ARGS, e.g. make collect-youtube ARGS='--dry-run --country ksa'"

venv:
	python3.11 -m venv .venv || python3 -m venv .venv

install: venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements.txt
	@test -f .env || (cp .env.example .env && echo "Created .env - add your ROBOFLOW_API_KEY")

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

prep eval report:
	@echo "'make $@' is not implemented yet - collection is done first (see README roadmap)." >&2
	@exit 1

test:
	$(PY) -m pytest -q

clean-logs:
	rm -rf logs
