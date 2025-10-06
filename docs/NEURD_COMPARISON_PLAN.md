# NEURD Comparison Plan

This document describes how we will reproduce NEURD’s published pipeline inside the official Docker image and align its outputs with the ConnectomeBench benchmark tasks. The goal is to produce like-for-like metrics so we can evaluate NEURD alongside the LLM-driven workflows highlighted in the ConnectomeBench paper.

## 1. Reproduce the Upstream NEURD Environment

1. Pull the vendor image that the NEURD README targets:
   ```bash
   docker pull celiib/neurd:v2
   ```
2. Start an interactive shell (skip the default Jupyter entrypoint), enable GPU, and mount this repo:
   ```bash
   docker run --rm -it \
     --gpus all \
     --entrypoint /bin/bash \
     -v "$PWD":/workspace \
     -w /workspace \
     celiib/neurd:v2
   ```
3. Inside the container install the vendored code exactly as the docs specify:
   ```bash
   pip3 install ./external/NEURD
   ```
4. Sanity-check with the upstream tests before collecting results:
   ```bash
   python -m unittest discover -s external/NEURD/tests/unit -v
   python external/NEURD/tests/integration/test_autoproof_pipeline.py
   ```
5. Export required credentials (e.g., `CAVE_TOKEN`) in the shell if you run pipelines beyond the fixtures. A helper script is available:
   ```bash
   uv run python scripts/caveclient_token.py --request   # prints the login URL
   uv run python scripts/caveclient_token.py --save "<token>"
   uv run python scripts/caveclient_token.py --show-path  # reveals where the token was stored
   export CAVE_TOKEN=$(jq -r '.token' ~/.cloudvolume/secrets/global.daf-apis.com-cave-secret.json)
   ```
   If you encounter a `403 missing_tos` error when using `CAVEclient`, visit
   `https://global.daf-apis.com/sticky_auth/api/v1/tos/2/accept` in a browser to accept the
   MICrONS Terms of Service once.

## Comparison Spec (Frozen)

This section fixes the evaluation choices to ensure a fair, reproducible comparison.

- Dataset: MICrONS (`minnie65_public` datastack; pin a timestamp when available).
- Primary task: Split identification. Segment typing is out-of-scope for the first pass; merge tasks are N/A.
- NEURD configuration: Use upstream defaults (decimation ratio 0.25; process/proof versions as shipped). No tuning.
- Compute: Run in `celiib/neurd:v2` with `--gpus all` when available. Record hardware specs.
- Decision mapping (NEURD → benchmark):
  - Event-level decision is positive if any NEURD suggested split lies within a fixed radius R of the event coordinates.
  - Radius R: 3000 nm (3 µm). Justification: robust to decimation and voxel quantization while selective at soma/branch scales.
  - Abstentions: if NEURD produces no nearby suggestion, count as incorrect in primary accuracy; also report abstain rate.
- Metrics: Accuracy (primary), precision/recall/F1 (secondary), 95% bootstrap CIs, and per-event runtime; report abstain rate.
- Pilot size: 25–50 events to validate R visually, then freeze for full runs.
- Outputs: Write NEURD baseline JSON/CSV under `scripts/output/neurd_baseline/` with provenance (image digest, NEURD commit, process/proof versions, R, timestamp).

## Fairness Principles

- Cohort parity: Use the same segment IDs/events for both methods.
- Time/space parity: Pin `datastack` and timestamp in both ConnectomeBench and NEURD VDI.
- Pre-registration: Freeze R, abstention policy, and any mappings before test; do not tune on test data.
- Failure handling: Count missing outputs as incorrect in primary metrics; report abstain rate and error breakdowns.

## 2. Understand ConnectomeBench Benchmark Tasks

We will reuse the existing scripts to ensure parity with the paper:

- `scripts/segmentation_classification.py` — segment type identification via `ConnectomeVisualizer` renders plus `LLMProcessor` multimodal prompts.
- `scripts/split_merge_resolution.py` — split/merge identification and comparison tasks; generates neuron/EM imagery, builds prompts, parses model responses, and records accuracy.
- `scripts/resnet_split_merge.py` — supervised vision baseline using the same JSON annotations as the LLM experiments.
- `scripts/util.py` — shared LLM client wrapper (OpenAI, Anthropic, Gemini, etc.).
- `scripts/connectome_visualizer.py` — CAVEclient access & mesh/EM rendering used by all benchmarks.

The outputs from these scripts (`scripts/output/...` JSON/CSV files) define the metrics reported in the abstract.

## 3. Map NEURD Pipeline Stages

Key components inside the submodule mirror those tasks:

- `external/NEURD/tests/integration/test_autoproof_pipeline.py` exercises the entire stack: mesh fetch → decimation → soma extraction → decomposition → multi-soma split suggestions/execution → cell typing → `auto_proof_stage`.
- `external/NEURD/neurd/neuron_pipeline_utils.py`
  - `cell_type_ax_dendr_stage` bundles preprocessing, nucleus pairing, synapse/spine loading, and intrinsic cell-type inference.
  - `auto_proof_stage` calls `proofread_neuron_full`, capturing `filtering_info`, red/blue split suggestions, split locations, and runtime metrics.
- `neuron.pipeline_products` accumulates per-stage outputs that we can serialize for comparison (soma stats, split candidates, auto-proof summaries).

## 4. Adapter Strategy for Comparison

We will implement a ConnectomeBench-side adapter (e.g., `scripts/neurd_adapter.py`) that runs inside the Docker container and emits results in the same schema the benchmark scripts expect.

1. **Input:** list of segment IDs and associated metadata (mirroring our existing JSON fixtures).
2. **Execution:** for each segment, call NEURD’s:
   - `vdi.fetch_segment_id_mesh` (optionally reusing cached meshes to avoid re-downloads).
   - `tu.decimate`, `sm.soma_indentification`, `neuron.Neuron(...).calculate_decomposition_products()`.
   - `neuron.calculate_multi_soma_split_suggestions()` / `.multi_soma_split_execution()` to obtain split candidates.
   - `neuron_pipeline_utils.cell_type_ax_dendr_stage` followed by `auto_proof_stage` to gather proofread/split output.
3. **Output Normalization:** translate stage products into the ConnectomeBench JSON layout:
   - Segment type → use NEURD’s intrinsic cell-type inference to match our segment classification labels.
   - Split identification → align NEURD’s `filtering_info`, `red_blue_suggestions`, and split locations with the `merge_coords`-driven event records used by `split_merge_resolution.py`.
   - Merge tasks → N/A in primary tables; optional exploratory appendix may be added later with clear caveats.
   - Include runtime + provenance metadata so we can profile throughput vs. LLM approaches.
4. **Serialization:** write a JSON/CSV in `scripts/output/neurd_baseline/` that the benchmark scripts can ingest for metric computation alongside LLM runs.

## 5. Execution Workflow

1. Start the container as in Section 1 and ensure `pip3 install ./external/NEURD` has been executed.
2. Install `uv` inside the container if you want to reuse the existing CLI wrappers:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. Run the adapter via `uv` (for consistency with the rest of the repo):
   ```bash
   UV_CACHE_DIR=/workspace/.uv-cache \
   UV_DATA_DIR=/workspace/.uv-data \
   uv run python scripts/neurd_adapter.py --config configs/neurd_fixture.json
   ```
4. Feed the serialized NEURD baseline into the current evaluation scripts — for example, point `scripts/resnet_split_merge.py` or analysis notebooks at the new JSON file.
5. Record all command invocations, commit hashes, and environment variables for reproducibility, matching the project’s PR expectations.

## 6. Next Steps

- Implement `scripts/neurd_adapter.py` with CLI arguments for input JSON, output path, and stage toggles.
- Validate against the bundled fixture (`segment_id=864691135510518224`) to confirm parity with the integration test.
- Extend to the paper’s datasets (FlyWire + MICrONS) once CAVE credentials are wired up.
- Automate the comparison in a notebook or results script that juxtaposes NEURD metrics with the LLM benchmarks, ready for inclusion in manuscripts or follow-up PRs.

Document commits and artifacts as we progress so reviewers can trace each result back to exact commands and environments.
