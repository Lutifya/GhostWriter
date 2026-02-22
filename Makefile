# ──────────────────────────────────────────────────────────────────────────────
# GhostWriter – Makefile
# ──────────────────────────────────────────────────────────────────────────────
# Usage:
#   make setup          Create virtualenv and install all dependencies
#   make devices        List available audio devices
#   make phase1         Run Phase 1 – audio level monitor (Ctrl+C to stop)
#   make phase2         Run Phase 2 – transcribe a 5-second recorded clip
#   make phase2-wav     Run Phase 2 – transcribe WAV=$(WAV) (e.g. make phase2-wav WAV=file.wav)
#   make phase3         Run Phase 3 – live real-time transcription (Ctrl+C to stop)
#   make run            Run the full app with overlay UI
#   make clean          Remove virtualenv and __pycache__ directories
# ──────────────────────────────────────────────────────────────────────────────

VENV       := .venv
PYTHON     := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip
ACTIVATE   := . $(VENV)/bin/activate

# Configurable at the command line, e.g.:  make phase2 MODEL=distil-medium.en
MODEL      ?= distil-large-v3
LANG       ?=
WAV        ?=

# Pass --language only when LANG is set
ifdef LANG
LANG_FLAG := --language $(LANG)
else
LANG_FLAG :=
endif

.DEFAULT_GOAL := help

# ──────────────────────────────────────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo ""
	@echo "  GhostWriter – available targets"
	@echo "  ────────────────────────────────────────────────────────────"
	@echo "  make setup           Create venv + install all dependencies"
	@echo "  make devices         List audio devices and exit"
	@echo "  make phase1          Audio level monitor (Ctrl+C to stop)"
	@echo "  make phase2          Transcribe a 5-second recorded clip"
	@echo "  make phase2-wav WAV=path/to/file.wav"
	@echo "                       Transcribe a WAV file"
	@echo "  make phase3          Live real-time transcription (Ctrl+C to stop)"
	@echo "  make run             Full app with overlay UI"
	@echo "  make clean           Remove venv and caches"
	@echo ""
	@echo "  Options (override with make <target> OPTION=value):"
	@echo "    MODEL   Whisper model  (default: $(MODEL))"
	@echo "    LANG    Language code  (default: auto-detect)"
	@echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: setup
setup: $(VENV)/bin/activate

$(VENV)/bin/activate:
	@echo "→ Creating virtual environment in $(VENV)/ ..."
	python3 -m venv $(VENV)
	@echo "→ Upgrading pip ..."
	$(PIP) install --upgrade pip
	@echo "→ Installing dependencies from requirements.txt ..."
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "✓ Setup complete. Virtual environment: $(VENV)/"
	@echo "  To activate manually: source $(VENV)/bin/activate"
	@echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Phases
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: devices
devices: $(VENV)/bin/activate
	$(PYTHON) main_phase1.py --list-devices

.PHONY: phase1
phase1: $(VENV)/bin/activate
	$(PYTHON) main_phase1.py

.PHONY: phase2
phase2: $(VENV)/bin/activate
	$(PYTHON) main_phase2.py --record 5 --model $(MODEL) $(LANG_FLAG)

.PHONY: phase2-wav
phase2-wav: $(VENV)/bin/activate
ifndef WAV
	$(error WAV is not set. Usage: make phase2-wav WAV=path/to/file.wav)
endif
	$(PYTHON) main_phase2.py --wav $(WAV) --model $(MODEL) $(LANG_FLAG)

.PHONY: phase3
phase3: $(VENV)/bin/activate
	$(PYTHON) main_phase3.py --model $(MODEL) $(LANG_FLAG)

.PHONY: run
run: $(VENV)/bin/activate
	$(PYTHON) main.py --model $(MODEL) $(LANG_FLAG)

# ──────────────────────────────────────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	@echo "→ Removing $(VENV)/ ..."
	rm -rf $(VENV)
	@echo "→ Removing __pycache__ directories ..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Clean."
