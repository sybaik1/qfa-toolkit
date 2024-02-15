VENV := .venv

.DEFAULT:

.PHONY: install
install:
	pip install .

$(VENV):
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e .

.PHONY: check
check: $(VENV)
	$(VENV)/bin/python -m unittest

.PHONY: vim
vim: .git $(VENV)
	. $(VENV)/bin/activate; vim $$(git ls-files)

.git:
	git init


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
