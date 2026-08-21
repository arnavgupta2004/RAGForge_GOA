.PHONY: install download-data build-index test test-unit test-integration evaluate benchmark serve

install:
	pip install -r requirements.txt

download-data:
	python scripts/download_dataset.py

build-index:
	python scripts/build_index.py

test-unit:
	pytest tests/unit -q

test-integration:
	pytest tests/integration -q

test: test-unit test-integration

evaluate:
	python scripts/evaluate.py --sample-size 150

evaluate-full:
	python scripts/evaluate.py --sample-size 150 --with-generation

benchmark:
	python scripts/benchmark.py --n 100

serve:
	uvicorn api.main:app --reload --port 8420
