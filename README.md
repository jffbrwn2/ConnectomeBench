# ConnectomeBench: AI-Powered Connectome Proofreading

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jffbrwn2/connectomebench.git
cd connectomebench
```

2. Install dependencies using `uv`:
```bash
uv pip install -r requirements.txt
```

## Usage

### Data Processing

The toolkit provides several scripts for processing connectome data:

- `scripts/get_data.py`: Gather training data from MICrONS or FlyWire edit histories
- `scripts/split_merge_resolution.py`: Process and evaluate split/merge events
- `scripts/segmentation_classification.py`: Classify segmentation data

### Visualization

The `ConnectomeVisualizer` class provides tools for visualizing neurons and EM data:

```python
from scripts.connectome_visualizer import ConnectomeVisualizer

# Initialize visualizer
visualizer = ConnectomeVisualizer(species="fly")

# Load and visualize neurons
visualizer.load_neurons([720575940625431866])
visualizer.save_3d_views(base_filename="3d_neuron_mesh")
```

### LLM for analysis

The toolkit integrates with multiple LLM providers for automated analysis:

```python
from scripts.util import LLMProcessor

# Initialize LLM processor
processor = LLMProcessor(model="gpt-4o")

# Process data
results = await processor.process_data(data)
```

## Configuration

The toolkit supports various configuration options:

- Species selection (fly/mouse)
- Visualization parameters
- LLM model selection
- Processing options

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this toolkit in your research, please cite:

```
@software{connectomebench2025,
  author = {Jeffrey Brown},
  title = {ConnectomeBench: Can LLMs proofread the connectome?},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/jffbrwn2/connectomebench}
}
``` 
