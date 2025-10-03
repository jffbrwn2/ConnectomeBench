# NEURD Code Audit Narrative

Working through the NEURD source felt less like reviewing a maintained package and more like excavating layers of lab prototypes that never solidified. The symptoms surface immediately.

## Monolithic “Utilities”
- `external/NEURD/neurd/neuron_utils.py:1-40` opens a 10 308-line file, importing the same symbol twice and dropping straight into banner comments. There is no separation between data prep, feature computation, or plotting; everything lives in one scroll.
- Within `external/NEURD/neurd/neuron.py:592-630`, the equality helper documents interactive debugging steps (`tu = reload(tu)`, `neuron = reload(neuron)`) right inside the class definition. Production behaviour still depends on REPL-era instructions.

## Configuration by Overwrite
- `external/NEURD/neurd/proofreading_utils.py:13-19` sets `current_proof_version` to 6, then immediately overwrites it with 7; `split_version` gets four successive assignments. The “active” configuration depends on import order, not intent.
- `external/NEURD/neurd/preprocess_neuron.py:9-34` pins `process_version = 10` at the top of a 5k-line module, leaving older variants commented nearby. Reproducing an earlier run would require editing source by hand.

## Missing Dependency “Handling”
- On import, `external/NEURD/neurd/cave_client_utils.py:1-14` catches `ImportError` and prints `pip3 install …` commands (“Must install cloudvolume and caveclinet”) instead of raising. If the user misses that console spam, the program fails later with unrelated errors.
- The same module hard-codes release metadata and assumes secrets live at `/root/.cloudvolume/secrets` (`external/NEURD/neurd/cave_client_utils.py:44-64`), which collapses outside a single Linux environment.

## Dynamic Execution & Globals Everywhere
- `external/NEURD/neurd/cave_client_utils.py:87-101` uses `exec(f"global {table_name};{table_name}='{v}'", globals(), locals())` to populate configuration, mutating module globals at import time.
- `external/NEURD/neurd/parameter_utils.py:613-644` loads parameter dictionaries by executing strings: `exec(f"import {module_name}; from {module_name} import {dict_name}")`. Syntax errors or missing files explode at runtime.

## Random IDs and Temp Directories
- `external/NEURD/neurd/soma_extraction_utils.py:88-115` creates `Path.cwd() / "Poisson_temp"` on every run and assigns random `segment_id` values (`random.randint(0,999999)`), scattering non-deterministic files across the repo.
- Similar randomness recurs in preprocessing (`external/NEURD/neurd/preprocess_neuron.py:3098`, `external/NEURD/neurd/soma_extraction_utils.py:103`), making output paths unpredictable and cleanup manual.

## Visualization Coupled to Core Logic
- Importing `external/NEURD/neurd/neuron_visualizations.py:1-33` automatically loads GUI libraries (`ipyvolume`, `matplotlib`) and defines default colours. Even a headless analysis drags 3D plotting dependencies into scope.

## Fragile Testing Story
- The lone integration test (`external/NEURD/tests/integration/test_autoproof_pipeline.py:1-80`) shells out to Meshlab CLI, writes to `temp/`, and historically referenced a misspelled `pathllib`. There are no unit tests guarding the 10k-line modules.

## Hard-Coded Paths & Debug Artefacts
- `external/NEURD/neurd/cave_client_utils.py:64` pins CAVE tokens to `/root/.cloudvolume/secrets`, preventing portable setups without editing code.
- Developer notes like `"******* This ERRORED AND CALLED OUR NEURON NONE …"` remain in production (`external/NEURD/neurd/soma_extraction_utils.py:1120-1124`), signalling unresolved debugging left behind.

## Takeaway
These aren’t isolated slip-ups; they’re systemic patterns—global state mutated by `exec`, configuration encoded as random assignments, platform assumptions baked into imports, and monolithic modules that intertwine analytics, plotting, and debugging. For anyone trying to install or extend NEURD, the codebase itself becomes the first obstacle.

## Contrast With Published Claims
The Nature paper introduces NEURD as a polished toolkit that “automates proofreading,” “makes massive datasets more accessible,” and “builds on existing open source software for mesh manipulation.” The repository tells a different story:
- **Automation vs. Manual Glue** – Automated proofreading in the paper becomes a single fragile integration test that shells out to a GUI binary and expects manual temp directories (`external/NEURD/tests/integration/test_autoproof_pipeline.py:1-80`).
- **Accessibility vs. Installation Roulette** – Instead of reproducible packaging, imports emit console instructions (“Must install cloudvolume…”) and require editing hard-coded paths like `/root/.cloudvolume/secrets` (`external/NEURD/neurd/cave_client_utils.py:1-64`).
- **Building on Open Source vs. Shelling to AppImages** – The promised mesh tooling still relies on meshlabserver command lines wired through temp files (`external/mesh_tools/mesh_tools/meshlab.py:206-420` inside the vendored dependency), not the “PyMeshLab replacement” highlighted in the release notes they cite.
- **Compact Graph Representations vs. 10k-Line Scripts** – The claim of “compact annotated graphs” lands on sprawling modules (`external/NEURD/neurd/neuron_utils.py:1-40`, `external/NEURD/neurd/proofreading_utils.py:13-78`) that mix analytics, plotting, and debugging in one place.

In practice, anyone attempting to replicate the paper’s results inherits a fragile environment that contradicts the polish implied by the publication.

## Additional Findings
- **Interactive Debug Prints** – Production modules log intermediate state to stdout (`external/NEURD/neurd/parameter_utils.py:631-654`, `external/NEURD/neurd/axon_utils.py:780-838`). Long-running routines narrate every step, drowning real errors in debug chatter.
- **Developer Notes Left In** – Comments such as `"******* This ERRORED AND CALLED OUR NEURON NONE: 77697401493989254 *********"` remain in live code (`external/NEURD/neurd/soma_extraction_utils.py:1107-1138`), signalling unresolved investigations baked into the release.
- **Secretion of Temp Artefacts** – Mesh helpers repeatedly log file paths and segmentation counts while writing directly to the working directory (`external/NEURD/neurd/soma_extraction_utils.py:1080-1138`), reinforcing how tightly the pipeline depends on side effects.
