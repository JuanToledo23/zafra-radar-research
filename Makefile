# Uses uv if installed, otherwise falls back to python3.11+ venv + pip.
# setup installs the whole pipeline (core + download + ee + dev); add llm / sam by hand:
#   .venv/bin/pip install -e ".[llm]"   (or uv pip install ...)
PYTHON ?= $(shell command -v python3.12 || command -v python3.11 || command -v python3)
VENV   := .venv
BIN    := $(VENV)/bin
UV     := $(shell command -v uv)

.PHONY: setup lint test download-nisar download-firms

setup:
ifdef UV
	uv venv --python 3.12 $(VENV)
	uv pip install --python $(BIN)/python -e ".[download,ee,dev]"
else
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[download,ee,dev]"
endif
	$(BIN)/python -c "import h5py, rasterio, asf_search, ee; print('imports OK')"

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

test:
	$(BIN)/pytest -q || [ $$? -eq 5 ]  # exit 5 = no tests collected yet

download-nisar:
	$(BIN)/python scripts/00_download_nisar.py $(ARGS)

download-firms:
	$(BIN)/python scripts/00_download_firms.py $(ARGS)
