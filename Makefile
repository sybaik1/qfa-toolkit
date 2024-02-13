.PHONY: install
install:
	pip install .

.PHONY: dev
dev:
	pip install -e .

.PHONY: test
test:
	python -m unittest

.PHONY: clean
clean:
	pip uninstall qfa_toolkit -y
