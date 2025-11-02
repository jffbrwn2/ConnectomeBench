# ConnectomeBench Tests

This directory contains the test suite for ConnectomeBench.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Shared pytest fixtures
├── test_analysis_utils.py   # Tests for analysis utility functions
├── test_util.py             # Tests for LLM processor and utilities
├── test_prompts.py          # Tests for prompt creation functions
├── test_integration.py      # End-to-end integration tests
└── README.md                # This file
```

Configuration is in `pyproject.toml` under `[tool.pytest.ini_options]`.

## Running Tests

### Install Test Dependencies

```bash
# Using uv (recommended)
uv sync --extra test

# Or using pip
pip install -e ".[test]"
```

### Run All Tests

```bash
# From project root
pytest

# With coverage report
pytest --cov=src --cov-report=html --cov-report=term-missing

# Verbose output
pytest -v

# Run specific test file
pytest tests/test_util.py

# Run specific test
pytest tests/test_util.py::TestLLMProcessorInit::test_default_initialization
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Skip tests requiring API access
pytest -m "not requires_api"
```

### Run Tests in Parallel

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run with 4 workers
pytest -n 4
```

## Test Categories

### Unit Tests
- `test_analysis_utils.py`: Tests for data analysis and metrics calculation
- `test_util.py`: Tests for LLM processor, image encoding, and response evaluation
- `test_prompts.py`: Tests for prompt creation functions

### Integration Tests
- Tests that verify end-to-end functionality
- Marked with `@pytest.mark.integration`
- May be slower and require external dependencies

## Test Fixtures

Common fixtures are defined in `conftest.py`:

- **Image Fixtures**: `temp_image_dir`, `create_test_image`, `sample_neuron_images`
- **LLM Fixtures**: `mock_llm_processor`, `llm_processor_gpt`, `llm_processor_claude`
- **Data Fixtures**: `sample_merge_data`, `sample_split_data`, `sample_option_data`
- **Results Fixtures**: `sample_results_dataframe`
- **Mock Fixtures**: `mock_api_response`, `mock_successful_llm_call`

## Writing New Tests

### Example Test

```python
import pytest
from src.util import LLMProcessor

class TestMyFeature:
    """Test description."""

    @pytest.fixture
    def my_fixture(self):
        """Fixture description."""
        return "test_data"

    def test_basic_functionality(self, my_fixture):
        """Test basic functionality."""
        assert my_fixture == "test_data"

    @pytest.mark.slow
    def test_slow_operation(self):
        """Test that takes a long time."""
        # Your slow test here
        pass

    @pytest.mark.requires_api
    async def test_api_call(self):
        """Test that requires API access."""
        # Your API test here
        pass
```

### Test Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`
- Fixtures: descriptive names (no test_ prefix)

### Best Practices

1. **Use fixtures** for common setup and teardown
2. **Mock external dependencies** (APIs, file systems) unless integration testing
3. **Use parametrize** for testing multiple inputs
4. **Use markers** to categorize tests
5. **Keep tests independent** - each test should run in isolation
6. **Use descriptive names** - test names should explain what they test
7. **Test edge cases** - not just happy paths
8. **Assert specific values** when possible, not just truthiness

## Coverage Goals

Aim for:
- **>80% overall coverage**
- **>90% coverage** for core utility functions
- **>70% coverage** for complex modules

Check coverage with:
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html  # View in browser
```

## Continuous Integration

Tests should:
- Run on every commit
- Complete in under 5 minutes
- Pass before merging PRs
- Generate coverage reports

## Debugging Tests

```bash
# Run with pdb on failure
pytest --pdb

# Run with print output
pytest -s

# Run last failed tests
pytest --lf

# Run specific test with verbose output
pytest -v tests/test_util.py::TestLLMProcessorInit::test_default_initialization

# Show local variables on failure
pytest -l
```

## Known Issues

- Some tests require mock data that simulates CAVEclient responses
- Tests assume pytest-asyncio is installed for async tests
- Image-based tests require PIL/Pillow

## Contributing

When adding new features:
1. Write tests first (TDD) or alongside implementation
2. Ensure tests pass: `pytest`
3. Check coverage: `pytest --cov=src`
4. Update this README if adding new test categories or fixtures
