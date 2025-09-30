# Meshlab CLI Detox Plan

## Why We’re Doing This
NEURD’s mesh pipeline leans on `meshlabserver` like it’s 2015: every decimation, Poisson surface reconstruction, and soma clean-up forks a Qt binary, writes an `.mls` script, and prays Xvfb behaves. We just spent a day juggling AppImages, glibc mismatches, and `DISPLAY` gymnastics. Enough. We’re replacing the CLI calls with `pymeshlab` so the benchmarks stop depending on a headless GUI app.

## Current Blast Radius
- `mesh_tools/meshlab.py` defines `Meshlab`, `Decimator`, `Poisson`, etc.—all thin wrappers around shelling out to `meshlabserver`.
- `mesh_tools/trimesh_utils.py` calls those wrappers in `decimate`, `poisson_surface_reconstruction`, and a half-dozen follow-on helpers.
- High-level NEURD modules (`soma_extraction_utils`, `neuron_pipeline_utils`, `neuron_utils`) assume the CLI helpers exist and manipulate the temp files they produce.
- Integration test `tests/integration/test_autoproof_pipeline.py` orchestrates the whole thing, so decimation failure cascades through every downstream stage.

## Replacement Strategy
1. Re-implement the CLI filters with `pymeshlab`:
   - Load meshes directly into a `MeshSet`, apply the equivalent filter sequences, and return `trimesh.Trimesh` objects.
   - Mirror parameter defaults from the `.mls` scripts so existing heuristics stay consistent.
2. Swap out the Meshlab class hierarchy:
   - Replace `Meshlab`, `Decimator`, `Poisson` with pure-Python utilities (`pymeshlab_decimate`, `pymeshlab_poisson`, etc.).
   - Update callers to use in-memory results instead of temp `.off` files.
3. Purge the script-generation cruft:
   - Drop `Scripter` and `.mls` plumbing from `mesh_tools/meshlab.py`.
   - Delete the `temp/*.mls` fixtures in `tests/` once the tests run on the new helpers.
4. Adjust integration tests to call the new utilities and validate numeric deltas (vertex counts, manifoldness) instead of checking for temp file side effects.

## Migration Stages (Because We’re Not Paid in Therapy Bills)
- **Stage 0:** Commit the rage-inducing evidence (this doc). Done.
- **Stage 1:** Prototype `pymeshlab` decimation + Poisson on a single fixture, compare with current mesh stats, and document acceptable drift. (Prototype completed; details below.)
- **Stage 2:** Replace `mesh_tools/meshlab.py` with the `pymeshlab` backend, update `trimesh_utils.decimate`, and ensure autoproof integration test runs end-to-end without the CLI.
- **Stage 3:** Remove now-dead dependencies (AppImage, Xvfb hacks) and update `docs/NEURD_RUNBOOK.md` to reflect the simpler setup.
- **Stage 4:** Celebrate by never touching `meshlabserver` again.

## Notes of Mild Spite
- `pymeshlab` ships with the Meshlab filters embedded, so we’re not losing functionality—just the shell gymnastics.
- Benchmark reproducibility actually improves; we won’t rely on distro-specific glibc quirks or AppImages that randomly change flags.
- If upstream NEURD wants the old behaviour, they can keep printing instructions that start with “install Xvfb”. We’re done.

## Stage 1 Prototype Results
- Command: `pixi run uv --project envs/neurd run python scripts/neurd_pymeshlab_prototype.py`
- Script: `scripts/neurd_pymeshlab_prototype.py` loads fixture `external/NEURD/tests/fixtures/864691135510518224.off` and applies `pymeshlab.meshing_decimation_quadric_edge_collapse` with `TargetPerc=0.25`.
- Metrics (original → decimated):
  - Vertex count: 154,713 → 33,691 (`vertex_ratio ≈ 0.218`)
  - Face count: 323,535 → 80,883 (`face_ratio ≈ 0.250`)
  - Surface area drift: −24.86%
  - Bounding-box diagonal drift: −0.19%
- Takeaway: Quadric decimation hits the target face ratio and keeps macro geometry stable, but erodes surface area by ~25%. Next step is to benchmark against the legacy CLI output (once we have a healthy baseline) to decide whether to tweak the filter parameters or add a post-pass clean-up.
