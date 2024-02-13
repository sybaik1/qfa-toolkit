VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
TARGET := $(VENV)/bin/

.PHONY: default
default:

.PHONY: install
install: $(VENV)
	$(PIP) install .

.PHONY: dev
dev: $(VENV)
	$(PIP) install -e .

.PHONY: test
test: $(VENV)
	$(PYTHON) -m unittest

.PHONY: clean
clean:
	rm -rf $(VENV)

.PHONY: vim
vim: .git
	vim $$(git ls-files)

.git:
	git init

$(VENV):
	python3 -m venv $(VENV)
