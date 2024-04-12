VENV := .venv

.DEFAULT:

.PHONY: install
install:
	pip install .

$(VENV): pyproject.toml
	python3 -m venv $(VENV)
	touch $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e .[dev]

.PHONY: check
check: $(VENV)
	$(VENV)/bin/python -m unittest

.PHONY: mypy
mypy: $(VENV)
	$(VENV)/bin/mypy src tests --explicit-package-bases

.PHONY: vim
vim: $(VENV) | .git
	. $(VENV)/bin/activate; vim $$(git ls-files)

.PHONY: jupyter
jupyter: $(VENV)
	$(VENV)/bin/jupyter notebook \
		--ip=0.0.0.0 --port=50000 --no-browser

document.md: | $(VENV)
	$(VENV)/bin/pdoc3 --pdf qfa_toolkit > $@

document.pdf: document.md | $(VENV)
	$(VENV)/bin/mdpdf -o $@ $<

.PHONY: clean clean-venv clean-build clean-pyc
clean: clean-venv clean-build clean-pyc

clean-venv:
	rm -rf $(VENV)

clean-build:
	rm -rf build/
	find . -name '*.egg-info' -exec rm -rf {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +
