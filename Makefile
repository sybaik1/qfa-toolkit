.PHONY: install
install:
	pip install .

.PHONY: dev
dev:
	pip install -e .

.PHONY: test
test:
	python -m unittest test/strategy_test.py
