# Convenience wrapper around reproduce.py. Every target is one command you can
# also run directly; nothing here is required to reproduce the study.
#
# Quick start on a clean machine:
#   make setup && make reproduce

PYTHON ?= python

.DEFAULT_GOAL := help
.PHONY: help setup test lint reproduce tune train data all report clean

help:  ## Show this list of targets
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "} {printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Create .venv and install the exact recorded environment
	python3.12 -m venv .venv
	.venv/bin/python -m pip install -r requirements-lock.txt
	@echo "Activate it with: source .venv/bin/activate"

test:  ## Run the unit tests
	$(PYTHON) -m unittest -v

lint:  ## Optional style check (installs nothing: uses uvx)
	uvx ruff check .

reproduce:  ## Rebuild every table and figure and revalidate all 45 saved models
	$(PYTHON) reproduce.py

data:  ## Download 20.48 GB of raw data and rebuild the aggregates (needs DATAVERSE_EMAIL)
	$(PYTHON) reproduce.py --stages data

tune:  ## Replay the four validation rounds on the tuning area
	$(PYTHON) reproduce.py --stages tune

train:  ## Fit the frozen final configuration on every area and seed
	$(PYTHON) reproduce.py --stages train

all:  ## Everything, from raw download to verification
	$(PYTHON) reproduce.py --stages all

report:  ## Rebuild the PDF from output/report.md
	$(PYTHON) render_report.py

clean:  ## Remove caches only. Never touches results/ or data/
	find . -name __pycache__ -type d -not -path './.venv/*' -exec rm -rf {} +
