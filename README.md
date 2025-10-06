# ConnectomeBench: Can LLMs Proofread the Connectome?

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jffbrwn2/connectomebench.git
cd connectomebench
```

2. Install dependencies using `uv`:
```bash
uv sync
```
or using `pip`:
```bash
pip install -r requirements.txt
```


## Usage

### Data Processing

The toolkit provides several scripts for processing connectome data:

- `scripts/get_data.py`: Gather training data from MICrONS or FlyWire edit histories
- Example usage (collect 50 MICrONS split edits and cache EM windows):
  ```bash
  uv run python scripts/get_data.py \
    --species mouse \
    --split-only \
    --num-neurons 50 \
    --extract-em-volumes \
    --output-dir training_data
  ```
  Run `uv run python scripts/get_data.py --help` for the full CLI.
- `scripts/split_merge_resolution.py`: Process and evaluate split/merge events
- `scripts/segmentation_classification.py`: Classify segmentations

### Visualization

The `ConnectomeVisualizer` class provides tools for visualizing neurons and EM data:

```python
from scripts.connectome_visualizer import ConnectomeVisualizer

# Initialize visualizer
visualizer = ConnectomeVisualizer(species="mouse")

# Load and visualize neurons
visualizer.load_neurons([864691134965949727])
visualizer.save_3d_views(base_filename="3d_neuron_mesh")
```

`ConnectomeVisualizer` is built largely on the data organized and provided through the [`CAVEClient`](https://github.com/CAVEconnectome/CAVEclient/) library. To get access to the data, please see the [CAVEClient README](https://github.com/CAVEconnectome/CAVEclient/). Specifically, you will need authentication tokens to access to the datasets (see the link [here](https://caveconnectome.github.io/CAVEclient/tutorials/authentication/)).

### LLM for analysis

The toolkit integrates with multiple LLM providers for automated analysis:

```python
from scripts.util import LLMProcessor

# Initialize LLM processor
processor = LLMProcessor(model="gpt-4o")

# Process data
results = await processor.process_single("Write prompt here")
```

## Segmentation Classification

The `scripts/segmentation_classification.py` script provides a way to classify segmentations into different categories. To get the same results as the paper, run the following command:

```bash
python segmentation_classification.py  
```

To run the script with different parameters, see the script for more details.

## Split error resolution & merge error detection

The `scripts/split_merge_resolution.py` script provides to test the performance of LLMs at resolving split errors and detecting merge errors. As in the paper, evaluate single shot and pairwise performance. To get the same results as the paper for mouse split error correction (single shot) with Claude 3.7 Sonnet, run the following command:

```bash
python split_merge_resolution.py --input-json scripts/training_data/mouse_256nm.json --task merge_identification --species mouse --zoom-margin 2048 --models claude-3-7-sonnet-20250219 --prompt-modes informative
```

To get the results for mouse split error correction (comparison) with Claude 3.7 Sonnet, run the following command: 

```bash
python split_merge_resolution.py --input-json scripts/training_data/mouse_256nm.json --task merge_comparison --species mouse --zoom-margin 2048 --models claude-3-7-sonnet-20250219 --prompt-modes informative
```

To get the results for mouse merge error detection (single shot) with Claude 3.7 Sonnet, run the following command:  
```bash
python split_merge_resolution.py --input-json scripts/training_data/merge_error_only_mouse.json --task split_identification --species mouse --models claude-3-7-sonnet-20250219 --prompt-modes informative
```

To get the results for mouse merge error detection (comparison) with Claude 3.7 Sonnet, run the following command:  
```bash
python split_merge_resolution.py --input-json scripts/training_data/merge_error_only_mouse.json --task split_comparison --species mouse --models claude-3-7-sonnet-20250219 --prompt-modes informative
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this toolkit in your research, please cite:

```
@software{connectomebench2025,
  author = {Jeff Brown},
  title = {ConnectomeBench: Can LLMs proofread the connectome?},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/jffbrwn2/connectomebench}
}
``` 
