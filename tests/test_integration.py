"""Integration tests for ConnectomeBench scripts."""

import pytest
import os
import sys
import json
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

# Add parent directory to path to import scripts
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


# ============================================================================
# Test Data Loading Functions
# ============================================================================

@pytest.mark.integration
class TestDataLoading:
    """Test data loading functions from scripts."""

    def test_load_merge_data_from_json(self, temp_json_file):
        """Test loading merge data from JSON file."""
        from scripts.merge_resolution import _load_and_filter_merge_data

        merge_data = _load_and_filter_merge_data(temp_json_file)

        assert isinstance(merge_data, list)
        assert len(merge_data) > 0
        assert all('is_merge' in item for item in merge_data)

    def test_load_split_data_from_json(self, temp_json_file):
        """Test loading split data from JSON file."""
        from scripts.split_resolution import _load_and_filter_split_data

        split_data = _load_and_filter_split_data(temp_json_file)

        assert isinstance(split_data, list)
        # May be empty if no split operations in test data
        if len(split_data) > 0:
            assert all('is_merge' in item for item in split_data)
            assert all(not item['is_merge'] for item in split_data)


# ============================================================================
# Test Prompt Creation Pipeline
# ============================================================================

@pytest.mark.integration
class TestPromptCreationPipeline:
    """Test end-to-end prompt creation."""

    @pytest.fixture
    def sample_event_result(self, temp_image_dir, create_test_image):
        """Create a complete event result with images."""
        # Create test images
        paths = {}
        for view in ['front', 'side', 'top']:
            path = temp_image_dir / f"test_{view}.png"
            create_test_image(path)
            paths[view] = str(path)

        return {
            'operation_id': 'test_op_123',
            'base_neuron_id': '123',
            'prompt_options': [
                {
                    'id': 'option_1',
                    'paths': {'zoomed': paths, 'default': paths}
                },
                {
                    'id': 'option_2',
                    'paths': {'zoomed': paths, 'default': paths}
                }
            ],
            'views': ['front', 'side', 'top'],
            'use_zoomed_images': True
        }

    def test_merge_identification_prompt_creation(self, sample_event_result, llm_processor_claude):
        """Test creating merge identification prompt end-to-end."""
        from src.prompts import create_merge_identification_prompt

        option_data = sample_event_result['prompt_options'][0]

        content = create_merge_identification_prompt(
            option_data=option_data,
            use_zoomed_images=True,
            views=['front', 'side'],
            llm_processor=llm_processor_claude,
            zoom_margin=2048,
            prompt_mode='informative'
        )

        assert isinstance(content, list)
        assert len(content) > 0

        # Verify structure
        has_images = any(c.get('type') == 'image' for c in content)
        has_text = any(c.get('type') == 'text' for c in content)
        assert has_images
        assert has_text

    def test_merge_comparison_prompt_creation(self, sample_event_result, llm_processor_claude):
        """Test creating merge comparison prompt end-to-end."""
        from src.prompts import create_merge_comparison_prompt

        content = create_merge_comparison_prompt(
            option_image_data=sample_event_result['prompt_options'],
            use_zoomed_images=True,
            views=['front'],
            llm_processor=llm_processor_claude,
            zoom_margin=2048,
            prompt_mode='informative'
        )

        assert isinstance(content, list)
        assert len(content) > 0


# ============================================================================
# Test Result Processing Pipeline
# ============================================================================

@pytest.mark.integration
class TestResultProcessing:
    """Test result processing and evaluation."""

    def test_response_evaluation_to_metrics(self):
        """Test complete flow from response to metrics."""
        from src.util import evaluate_response
        from src.analysis_utils import calculate_merge_identification_metrics

        # Simulate multiple LLM responses
        responses = [
            "<analysis>Good merge</analysis><answer>1</answer>",
            "<analysis>Bad merge</analysis><answer>-1</answer>",
            "<analysis>Good merge</analysis><answer>1</answer>",
            "<analysis>Bad merge</analysis><answer>-1</answer>",
        ]

        # Evaluate responses
        evaluations = [evaluate_response(r) for r in responses]

        # Create results dataframe
        df = pd.DataFrame({
            'operation_id': range(len(responses)),
            'id': [f'seg_{i}' for i in range(len(responses))],
            'model_prediction': [e['answer'] for e in evaluations],
            'is_correct_merge': [True, False, True, False],  # Ground truth
            'model_analysis': [e['analysis'] for e in evaluations]
        })

        # Calculate metrics
        metrics = calculate_merge_identification_metrics(df, include_ci=False)

        # Verify metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert metrics['total_samples'] == 4


# ============================================================================
# Test Configuration and Setup
# ============================================================================

@pytest.mark.integration
class TestConfiguration:
    """Test configuration and setup."""

    def test_import_all_modules(self):
        """Test that all main modules can be imported."""
        try:
            import src.util
            import src.prompts
            import src.analysis_utils
            import src.connectome_visualizer
        except ImportError as e:
            pytest.fail(f"Failed to import module: {e}")

    def test_llm_processor_initialization(self):
        """Test LLM processor can be initialized with different models."""
        from src.util import LLMProcessor

        for model in ["gpt-4o", "claude-3-7-sonnet-20250219", "gpt-4o-mini"]:
            processor = LLMProcessor(model=model)
            assert processor.model == model
            assert processor.max_concurrent > 0
            assert processor.max_tokens > 0


# ============================================================================
# Test Error Handling
# ============================================================================

@pytest.mark.integration
class TestErrorHandling:
    """Test error handling across the pipeline."""

    def test_missing_file_handling(self):
        """Test graceful handling of missing files."""
        from scripts.merge_resolution import _load_and_filter_merge_data

        result = _load_and_filter_merge_data('/nonexistent/file.json')

        # Should return empty list or handle gracefully
        assert isinstance(result, list)
        assert len(result) == 0

    def test_invalid_json_handling(self, tmp_path):
        """Test handling of invalid JSON files."""
        from scripts.merge_resolution import _load_and_filter_merge_data

        # Create invalid JSON file
        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text("{invalid json content")

        result = _load_and_filter_merge_data(str(invalid_json))

        assert isinstance(result, list)
        assert len(result) == 0

    def test_malformed_response_handling(self):
        """Test handling of malformed LLM responses."""
        from src.util import evaluate_response

        malformed_responses = [
            "No tags at all",
            "<analysis>Only analysis</analysis>",
            "<answer>Only answer</answer>",
            "<analysis>Incomplete",
            "Random text <answer>invalid</answer>"
        ]

        for response in malformed_responses:
            result = evaluate_response(response)

            # Should handle gracefully
            assert isinstance(result, dict)
            assert 'answer' in result
            assert 'analysis' in result


# ============================================================================
# Test Data Validation
# ============================================================================

@pytest.mark.integration
class TestDataValidation:
    """Test data validation and filtering."""

    def test_merge_event_validation(self):
        """Test that merge events are properly validated."""
        from scripts.merge_resolution import _load_and_filter_merge_data

        # Create test data with valid and invalid events
        test_data = [
            {  # Valid merge event
                'operation_id': 'valid_1',
                'is_merge': True,
                'before_root_ids': [1, 2],
                'interface_point': [100, 200, 300],
                'em_data': {'all_unique_root_ids': [1, 2, 3]}
            },
            {  # Invalid - missing interface_point
                'operation_id': 'invalid_1',
                'is_merge': True,
                'before_root_ids': [3, 4],
                'em_data': {'all_unique_root_ids': [3, 4, 5]}
            },
            {  # Invalid - not a merge
                'operation_id': 'invalid_2',
                'is_merge': False,
                'before_root_ids': [5],
                'after_root_ids': [6, 7],
                'interface_point': [400, 500, 600],
                'em_data': {'all_unique_root_ids': [5, 6, 7]}
            }
        ]

        # Write to temp file
        import tempfile
        import json

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            result = _load_and_filter_merge_data(temp_path)

            # Should only have the valid merge event
            assert len(result) == 1
            assert result[0]['operation_id'] == 'valid_1'
        finally:
            os.unlink(temp_path)


# ============================================================================
# Smoke Tests
# ============================================================================

@pytest.mark.integration
class TestSmokeTests:
    """Basic smoke tests to ensure critical functionality works."""

    def test_can_create_basic_prompt(self, temp_image_dir, create_test_image, llm_processor_claude):
        """Smoke test: Can create a basic prompt without errors."""
        from src.prompts import create_merge_identification_prompt

        # Create minimal valid input
        img_path = create_test_image(temp_image_dir / "test.png")

        option_data = {
            'id': 'test',
            'paths': {
                'zoomed': {'front': str(img_path)},
                'default': {'front': str(img_path)}
            }
        }

        # Should not raise
        content = create_merge_identification_prompt(
            option_data=option_data,
            use_zoomed_images=True,
            views=['front'],
            llm_processor=llm_processor_claude,
            zoom_margin=1024,
            prompt_mode='informative'
        )

        assert content is not None
        assert len(content) > 0

    def test_can_evaluate_basic_response(self):
        """Smoke test: Can evaluate a basic response without errors."""
        from src.util import evaluate_response

        response = "<analysis>Test</analysis><answer>1</answer>"
        result = evaluate_response(response)

        assert result is not None
        assert result['answer'] == '1'
        assert result['analysis'] == 'Test'

    def test_can_calculate_basic_metrics(self):
        """Smoke test: Can calculate metrics without errors."""
        from src.analysis_utils import calculate_merge_identification_metrics

        df = pd.DataFrame({
            'operation_id': [1, 2],
            'id': ['a', 'b'],
            'model_prediction': ['1', '-1'],
            'is_correct_merge': [True, False]
        })

        metrics = calculate_merge_identification_metrics(df, include_ci=False)

        assert metrics is not None
        assert 'accuracy' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
