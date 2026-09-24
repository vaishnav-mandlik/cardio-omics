SMK := uv run snakemake -s workflow/Snakefile

.PHONY: setup lint test test-e2e fetch run run-hpc clean docker

setup:
	uv sync --frozen --all-extras

lint:
	uv run ruff check .

test:
	uv run pytest -m "not slow"

test-e2e:
	uv run pytest -m slow

fetch:
	uv run cardioomics fetch --config config/config.yaml

run:
	$(SMK) --profile profiles/local

run-hpc:
	$(SMK) --profile profiles/slurm-puma

docker:
	docker build -f containers/Dockerfile -t cardio-omics:dev .

clean:
	rm -rf results .test_work
