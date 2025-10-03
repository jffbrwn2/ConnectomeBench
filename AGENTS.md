# Repository Guidelines

## Project Structure & Module Organization
The project is a Python 3.12 workspace managed by `pyproject.toml` and `uv`. Operational scripts live in `scripts/`, with `connectome_visualizer.py` for 3D neuron rendering, `split_merge_resolution.py` and `segmentation_classification.py` for benchmarking workflows, plus utility helpers in `util.py`. Notebooks (`*.ipynb`) illustrate analysis flows; keep them lightweight and export code into scripts. Place any large datasets or prompt variants under `scripts/training_data/` (git-ignored) and document data provenance in PRs.

## Setup, Build & Run Commands
Install dependencies once via `uv sync` (preferred) or `pip install -r requirements.txt`. Use `uv run python scripts/get_data.py --help` to inspect data download options. Benchmark tasks run with `uv run python scripts/split_merge_resolution.py …` and `uv run python scripts/segmentation_classification.py …`; record CLI arguments in your PR. When iterating on visualization, run `uv run python -m scripts.connectome_visualizer` to validate imports.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation and descriptive snake_case for modules, functions, and filenames (`split_merge_resolution.py`). Public classes such as `ConnectomeVisualizer` use CapWords. Prefer type hints and structured logging (see `scripts/util.py`) over print statements. Keep long-running API helpers in `scripts/util.py`; new providers should slot in as cohesive classes or functions within that module.

## Testing Guidelines
Automated tests are not yet seeded—add them under `tests/` mirroring the module structure (`tests/test_split_merge_resolution.py`). Use `pytest` and target critical LLM-selection logic, parsing helpers, and data pre/post-processing. Capture external API calls with fixtures or VCR-style cassettes, and document any credentials required. Run `uv run pytest -q` before opening a PR.

## Commit & Pull Request Guidelines
Recent history shows terse commits; aim for imperative, specific subjects (`Add FlyWire merge comparison pipeline`). Reference issues in the body when applicable. PRs should include: summary of changes, sample command invocations, validation evidence (test logs or before/after metrics), and notes on data sources or credential expectations. Request review from a domain owner before merging.

## Data Access & Credentials
CAVEClient and vendor APIs require tokens; load them via environment variables (`export ANTHROPIC_API_KEY=...`) or a `.env` file consumed by your runner. Never commit secrets. If sharing sample notebooks, redact IDs or export anonymized subsets.
