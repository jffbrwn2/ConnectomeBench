# NEURD Native Runbook

## Overview
This repository vendors the NEURD framework as a git submodule at `external/NEURD` and exposes it through the `neurd` optional extra in `pyproject.toml`. The goal is to bootstrap a repeatable, docker-free environment so we can benchmark NEURD inside ConnectomeBench.

## Pixi Environment Setup
1. Install [pixi](https://pixi.sh/latest/) in your user space (no sudo required).
2. Point pixi at a storage location with sufficient quota, otherwise it defaults to `~/.pixi`:
   ```bash
   export PIXI_HOME="/orcd/data/edboyden/pixi"
   ```
3. From the repo root, create or refresh the environment:
   ```bash
   pixi install
   ```
   This pulls conda-forge packages (CGAL, Embree, Meshlab, OpenGL shims, toolchain) declared in `pixi.toml` for Linux x86_64.

## Python Dependencies via uv
Once the pixi environment is active:
```bash
pixi run uv-sync-neurd
```
This invokes `uv --project envs/neurd sync --python 3.8`, installing NEURD from the submodule (`external/NEURD`) inside a Python 3.8 virtualenv alongside its Python dependencies. Python 3.8 is required because upstream wheels (e.g., `open3d==0.11.2` via `mesh_processing_tools`) are not published for newer interpreters.

## Credentials & Data
- Place any CAVE authentication token in `.env` at the repository root (`CAVE_TOKEN=...`).
- Integration tests consume fixtures shipped with NEURD (`external/NEURD/tests/fixtures`); no external downloads are required.

## Smoke Tests
After syncing dependencies, validate the install:
```bash
pixi run neurd-tests
```
This executes `python -m unittest discover -s external/NEURD/tests/unit` inside the pixi + uv environment. The longer pipeline in `external/NEURD/tests/integration/test_autoproof_pipeline.py` can be invoked manually once you’re ready for full benchmarking.

## Next Steps
- Pin the NEURD submodule to commits of interest as benchmarks progress (`git submodule update --remote external/NEURD`).
- Record benchmark commands and metrics per run so results can be reproduced under the same pixi + uv lockfiles.
