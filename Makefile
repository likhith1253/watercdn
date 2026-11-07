PY ?= .venv/bin/python

env:
	python3 -m venv .venv && $(PY) -m pip install -r requirements-lock.txt && $(PY) -m pip install -e . --no-deps

test:
	$(PY) -m pytest

reproduce:
	$(PY) scripts/reproduce.py

e1 e2 e3 e4 report:
	$(PY) scripts/reproduce.py --only $@

.PHONY: env test reproduce e1 e2 e3 e4 report
