# Table Oracle
#
# Windows note: this repo was developed on Windows, where `make` is usually
# absent. Every target below is a one-line command you can also run directly;
# the equivalent is printed by `make help`.

PY ?= .venv/Scripts/python.exe
ifeq ($(OS),)
PY = .venv/bin/python
endif

Q ?= Can I cast a spell and disengage in the same turn?

.PHONY: help venv install ingest search ask dev status test eval eval-retrieval charts clean

help:
	@echo "make install  - create .venv and install dependencies"
	@echo "make ingest   - build the search index from corpus/ (no API key needed)"
	@echo "make status   - show what is currently indexed"
	@echo "make search Q=\"...\"  - retrieval only, no model call"
	@echo "make ask    Q=\"...\"  - streamed, cited answer (needs ANTHROPIC_API_KEY)"
	@echo "make dev      - run the API at http://127.0.0.1:8000"
	@echo "make test     - run the unit tests (no API keys required)"
	@echo "make eval     - full eval: retrieval + answers + judge (costs money)"
	@echo "make eval-retrieval - retrieval scoring only (free, no API key)"
	@echo "make charts   - regenerate the README charts from evals/results/"

venv:
	python -m venv .venv

install: venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

ingest:
	$(PY) -m tableoracle.cli ingest

status:
	$(PY) -m tableoracle.cli status

search:
	$(PY) -m tableoracle.cli search "$(Q)" -v

ask:
	$(PY) -m tableoracle.cli ask "$(Q)"

dev:
	$(PY) -m uvicorn tableoracle.api.app:app --reload --port 8000

test:
	$(PY) -m pytest -q

eval:
	$(PY) -m tableoracle.evals.run

eval-retrieval:
	$(PY) -m tableoracle.evals.run --retrieval-only

charts:
	$(PY) scripts/make_charts.py

clean:
	rm -rf data/*.db data/*.sqlite data/usage.jsonl
