.PHONY: install
install:
	pip install -e .

.PHONY: test
test:
	python -m unittest test/strategy_test.py
