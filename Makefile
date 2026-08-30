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

.PHONY: help venv install ingest search ask dev status test eval clean

help:
	@echo "make install  - create .venv and install dependencies"
	@echo "make ingest   - build the search index from corpus/ (needs OPENAI_API_KEY)"
	@echo "make status   - show what is currently indexed"
	@echo "make search Q=\"...\"  - retrieval only, no model call"
	@echo "make ask    Q=\"...\"  - streamed, cited answer (needs both keys)"
	@echo "make dev      - run the API at http://127.0.0.1:8000"
	@echo "make test     - run the unit tests (no API keys required)"
	@echo "make eval     - M3; not implemented yet"

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
	@echo "The eval harness lands in M3. evals/questions.yaml defines the format."
	@exit 1

clean:
	rm -rf data/*.db data/*.sqlite data/usage.jsonl
