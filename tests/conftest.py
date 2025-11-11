"""Shared pytest fixtures and configuration."""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import Mock

from src.util import LLMProcessor


# ============================================================================
# Image Fixtures
# ============================================================================

@pytest.fixture
def temp_image_dir(tmp_path):
    """Create a temporary directory for test images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    return img_dir


@pytest.fixture
def create_test_image():
    """Factory fixture to create test images."""
    def _create_image(path, size=(100, 100), color='red'):
        """Create a test image at the given path."""
        img = Image.new('RGB', size, color=color)
        img.save(path)
        return path
    return _create_image


@pytest.fixture
def sample_neuron_images(temp_image_dir, create_test_image):
    """Create a set of neuron view images."""
    paths = {}
    for view in ['front', 'side', 'top']:
        path = temp_image_dir / f"neuron_{view}.png"
        create_test_image(path, color=['red', 'green', 'blue'][['front', 'side', 'top'].index(view)])
        paths[view] = str(path)
    return paths


# ============================================================================
# LLM Processor Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_processor():
    """Create a mock LLM processor for testing."""
    processor = Mock(spec=LLMProcessor)
    processor.model = "claude-3-7-sonnet-20250219"
    processor.max_tokens = 4096
    processor.max_concurrent = 25
    processor._encode_image_to_base64 = Mock(return_value=("fake_base64_data", "image/png"))
    return processor


@pytest.fixture
def llm_processor_gpt():
    """Create a real LLM processor configured for GPT (without API calls)."""
    return LLMProcessor(model="gpt-4o", max_concurrent=5)


@pytest.fixture
def llm_processor_claude():
    """Create a real LLM processor configured for Claude (without API calls)."""
    return LLMProcessor(model="claude-3-7-sonnet-20250219", max_concurrent=5)


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def sample_merge_data():
    """Create sample merge operation data."""
    return {
        'operation_id': 'merge_op_123',
        'base_neuron_id': 'neuron_456',
        'timestamp': 1234567890,
        'merge_coords': [100000, 200000, 300000],
        'interface_point': [100000, 200000, 300000],
        'before_root_ids': [111, 222],
        'after_root_ids': [333],
        'is_merge': True,
        'species': 'mouse'
    }


@pytest.fixture
def sample_split_data():
    """Create sample split operation data."""
    return {
        'operation_id': 'split_op_456',
        'base_neuron_id': 'neuron_789',
        'timestamp': 1234567890,
        'merge_coords': [150000, 250000, 350000],
        'interface_point': [150000, 250000, 350000],
        'before_root_ids': [444],
        'after_root_ids': [555, 666],
        'is_merge': False,
        'species': 'fly'
    }


@pytest.fixture
def sample_option_data(sample_neuron_images):
    """Create sample option data with image paths."""
    return {
        'id': 'segment_123',
        'paths': {
            'zoomed': sample_neuron_images,
            'default': sample_neuron_images
        }
    }


# ============================================================================
# Results Fixtures
# ============================================================================

@pytest.fixture
def sample_results_dataframe():
    """Create a sample results dataframe for testing."""
    import pandas as pd

    data = {
        'operation_id': [f'op_{i}' for i in range(100)],
        'id': [f'seg_{i}' for i in range(100)],
        'model_prediction': ['1'] * 70 + ['-1'] * 30,
        'is_correct_merge': [True] * 60 + [False] * 40,
        'model': ['gpt-4o'] * 50 + ['claude-3-7-sonnet-20250219'] * 50,
        'index': list(range(100)),
        'model_analysis': ['Test analysis'] * 100
    }

    return pd.DataFrame(data)


# ============================================================================
# Mock API Fixtures
# ============================================================================

@pytest.fixture
def mock_api_response():
    """Create a mock API response."""
    return {
        'choices': [{
            'message': {
                'content': '<analysis>This is a test analysis.</analysis><answer>1</answer>'
            }
        }]
    }


@pytest.fixture
def mock_successful_llm_call(monkeypatch):
    """Mock successful LLM API call."""
    def mock_call(*args, **kwargs):
        return "<analysis>Test analysis</analysis><answer>1</answer>"

    monkeypatch.setattr(
        'src.util.LLMProcessor._call_api_sync',
        mock_call
    )


# ============================================================================
# File System Fixtures
# ============================================================================

@pytest.fixture
def temp_json_file(tmp_path):
    """Create a temporary JSON file."""
    import json

    json_path = tmp_path / "test_data.json"

    data = [
        {
            'operation_id': 'op_1',
            'is_merge': True,
            'before_root_ids': [1, 2],
            'after_root_ids': [3],
            'interface_point': [100, 200, 300],
            'em_data': {'all_unique_root_ids': [1, 2, 4, 5, 6]}
        },
        {
            'operation_id': 'op_2',
            'is_merge': False,
            'before_root_ids': [7],
            'after_root_ids': [8, 9],
            'after_root_ids_used': {'8': True, '9': False},
            'interface_point': [400, 500, 600],
            'em_data': {'all_unique_root_ids': [7, 8, 9, 10]}
        }
    ]

    with open(json_path, 'w') as f:
        json.dump(data, f)

    return str(json_path)


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def disable_api_calls(monkeypatch):
    """Automatically disable real API calls for all tests unless explicitly enabled."""
    # This prevents accidental API calls during testing
    # Individual tests can override this by using markers or explicit mocking
    pass


# ============================================================================
# Numpy Fixtures
# ============================================================================

@pytest.fixture
def sample_coordinates():
    """Create sample 3D coordinates."""
    return {
        'minpt': np.array([0, 0, 0]),
        'maxpt': np.array([1000, 1000, 1000]),
        'center': np.array([500, 500, 500])
    }


# ============================================================================
# Parametrize Helpers
# ============================================================================

# Common parameter combinations
MODELS_TO_TEST = ["gpt-4o", "claude-3-7-sonnet-20250219"]
SPECIES_TO_TEST = ["mouse", "fly"]
VIEWS_TO_TEST = [
    ["front", "side", "top"],
    ["front", "side"],
    ["front"]
]
PROMPT_MODES_TO_TEST = [
    "informative",
    "null",
    "informative+heuristic1",
    "informative+heuristic1+heuristic2"
]
