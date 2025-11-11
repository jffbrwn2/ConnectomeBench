"""Tests for prompts.py prompt creation functions."""

import pytest
import tempfile
import os
from pathlib import Path
from PIL import Image
from unittest.mock import Mock, patch

from src.prompts import (
    parse_prompt_mode,
    create_merge_identification_prompt,
    create_split_identification_prompt,
    create_merge_comparison_prompt,
    create_split_comparison_prompt,
    create_segment_classification_prompt,
    MERGE_HEURISTICS
)
from src.util import LLMProcessor


# ============================================================================
# Test parse_prompt_mode
# ============================================================================

class TestParsePromptMode:
    """Test parse_prompt_mode function."""

    def test_parse_base_mode_only(self):
        """Test parsing prompt mode with no heuristics."""
        base_mode, heuristics = parse_prompt_mode("informative")
        assert base_mode == "informative"
        assert heuristics == []

    def test_parse_single_heuristic(self):
        """Test parsing prompt mode with single heuristic."""
        base_mode, heuristics = parse_prompt_mode("informative+heuristic1")
        assert base_mode == "informative"
        assert heuristics == ["heuristic1"]

    def test_parse_multiple_heuristics(self):
        """Test parsing prompt mode with multiple heuristics."""
        base_mode, heuristics = parse_prompt_mode("informative+heuristic1+heuristic3+heuristic5")
        assert base_mode == "informative"
        assert set(heuristics) == {"heuristic1", "heuristic3", "heuristic5"}

    def test_parse_null_mode_with_heuristics(self):
        """Test parsing null mode with heuristics."""
        base_mode, heuristics = parse_prompt_mode("null+heuristic2")
        assert base_mode == "null"
        assert heuristics == ["heuristic2"]

    def test_parse_ignores_non_heuristic_parts(self):
        """Test that non-heuristic parts are ignored."""
        base_mode, heuristics = parse_prompt_mode("informative+other+heuristic1+something")
        assert base_mode == "informative"
        assert heuristics == ["heuristic1"]


# ============================================================================
# Test Merge Identification Prompt
# ============================================================================

class TestCreateMergeIdentificationPrompt:
    """Test create_merge_identification_prompt function."""

    @pytest.fixture
    def test_images(self, tmp_path):
        """Create test images for prompts."""
        paths = {}
        for view in ['front', 'side', 'top']:
            img = Image.new('RGB', (100, 100), color='red')
            img_path = tmp_path / f"test_{view}.png"
            img.save(img_path)
            paths[view] = str(img_path)
        return paths

    @pytest.fixture
    def option_data(self, test_images):
        """Create option data for testing."""
        return {
            'id': 'segment_123',
            'paths': {
                'zoomed': test_images,
                'default': test_images
            }
        }

    @pytest.fixture
    def llm_processor(self):
        """Create LLM processor for testing."""
        return LLMProcessor(model="claude-3-7-sonnet-20250219")

    def test_create_prompt_basic(self, option_data, llm_processor):
        """Test basic prompt creation."""
        content = create_merge_identification_prompt(
            option_data=option_data,
            use_zoomed_images=True,
            views=['front', 'side', 'top'],
            llm_processor=llm_processor,
            zoom_margin=2048,
            prompt_mode='informative'
        )

        assert isinstance(content, list)
        assert len(content) > 0

        # Check that images are included
        image_blocks = [c for c in content if c.get('type') == 'image']
        assert len(image_blocks) == 3  # front, side, top

        # Check that text is included
        text_blocks = [c for c in content if c.get('type') == 'text']
        assert len(text_blocks) > 0

        # Check for key content
        full_text = ' '.join([c['text'] for c in text_blocks])
        assert 'merge operation' in full_text.lower()
        assert 'blue' in full_text.lower()
        assert 'orange' in full_text.lower()

    def test_create_prompt_with_heuristics(self, option_data, llm_processor):
        """Test prompt creation with heuristics."""
        content = create_merge_identification_prompt(
            option_data=option_data,
            use_zoomed_images=True,
            views=['front', 'side', 'top'],
            llm_processor=llm_processor,
            zoom_margin=2048,
            prompt_mode='informative+heuristic1+heuristic2'
        )

        # Collect all text
        text_blocks = [c for c in content if c.get('type') == 'text']
        full_text = ' '.join([c['text'] for c in text_blocks])

        # Check that heuristics are included
        assert 'additional guidance' in full_text.lower()

    def test_create_prompt_null_mode(self, option_data, llm_processor):
        """Test prompt creation with null mode."""
        content = create_merge_identification_prompt(
            option_data=option_data,
            use_zoomed_images=True,
            views=['front', 'side', 'top'],
            llm_processor=llm_processor,
            zoom_margin=2048,
            prompt_mode='null'
        )

        text_blocks = [c for c in content if c.get('type') == 'text']
        full_text = ' '.join([c['text'] for c in text_blocks])

        # In null mode, informative instructions should not be present
        assert 'progressing in the same direction' not in full_text.lower()

    def test_create_prompt_subset_of_views(self, option_data, llm_processor):
        """Test prompt creation with subset of views."""
        content = create_merge_identification_prompt(
            option_data=option_data,
            use_zoomed_images=True,
            views=['front', 'side'],  # Only 2 views
            llm_processor=llm_processor,
            zoom_margin=2048,
            prompt_mode='informative'
        )

        # Should have only 2 images
        image_blocks = [c for c in content if c.get('type') == 'image']
        assert len(image_blocks) == 2

    def test_create_prompt_missing_image_warning(self, option_data, llm_processor, capsys):
        """Test that missing images produce warnings."""
        # Remove one image
        option_data['paths']['zoomed']['top'] = '/nonexistent/path.png'

        content = create_merge_identification_prompt(
            option_data=option_data,
            use_zoomed_images=True,
            views=['front', 'side', 'top'],
            llm_processor=llm_processor,
            zoom_margin=2048,
            prompt_mode='informative'
        )

        captured = capsys.readouterr()
        assert 'Warning' in captured.out or 'warning' in captured.out.lower()

    def test_create_prompt_openai_format(self, option_data):
        """Test prompt creation for OpenAI models."""
        llm_processor = LLMProcessor(model="gpt-4o")

        content = create_merge_identification_prompt(
            option_data=option_data,
            use_zoomed_images=True,
            views=['front'],
            llm_processor=llm_processor,
            zoom_margin=2048,
            prompt_mode='informative'
        )

        # OpenAI uses different block types
        image_blocks = [c for c in content if c.get('type') == 'input_image']
        text_blocks = [c for c in content if c.get('type') == 'input_text']

        assert len(image_blocks) > 0
        assert len(text_blocks) > 0


# ============================================================================
# Test Split Identification Prompt
# ============================================================================

class TestCreateSplitIdentificationPrompt:
    """Test create_split_identification_prompt function."""

    @pytest.fixture
    def test_images(self, tmp_path):
        """Create test images for prompts."""
        paths = {}
        for view in ['front', 'side', 'top']:
            img = Image.new('RGB', (100, 100), color='blue')
            img_path = tmp_path / f"split_{view}.png"
            img.save(img_path)
            paths[view] = str(img_path)
        return paths

    @pytest.fixture
    def option_data(self, test_images):
        """Create option data for testing."""
        return {
            'id': 'neuron_456',
            'paths': {
                'zoomed': test_images,
                'default': test_images
            },
            'merge_coords': [100, 200, 300],
            'zoom_margin': 2048
        }

    @pytest.fixture
    def llm_processor(self):
        """Create LLM processor for testing."""
        return LLMProcessor(model="claude-3-7-sonnet-20250219")

    def test_create_split_prompt_basic(self, option_data, llm_processor):
        """Test basic split identification prompt creation."""
        content = create_split_identification_prompt(
            option_data=option_data,
            use_zoomed_images=True,
            views=['front', 'side', 'top'],
            llm_processor=llm_processor,
            zoom_margin=2048,
            prompt_mode='informative'
        )

        assert isinstance(content, list)
        assert len(content) > 0

        # Check for images
        image_blocks = [c for c in content if c.get('type') == 'image']
        assert len(image_blocks) > 0

        # Check for split-specific content
        text_blocks = [c for c in content if c.get('type') == 'text']
        full_text = ' '.join([c['text'] for c in text_blocks])
        assert 'split' in full_text.lower() or 'segment' in full_text.lower()


# ============================================================================
# Test Comparison Prompts
# ============================================================================

class TestCreateComparisonPrompts:
    """Test merge and split comparison prompt creation."""

    @pytest.fixture
    def test_images(self, tmp_path):
        """Create test images."""
        paths = {}
        for i, view in enumerate(['front', 'side', 'top']):
            img = Image.new('RGB', (100, 100), color=['red', 'green', 'blue'][i])
            img_path = tmp_path / f"comp_{view}.png"
            img.save(img_path)
            paths[view] = str(img_path)
        return paths

    @pytest.fixture
    def option_data_list(self, test_images):
        """Create list of option data for comparison."""
        return [
            {
                'id': 'option_1',
                'paths': {'zoomed': test_images, 'default': test_images}
            },
            {
                'id': 'option_2',
                'paths': {'zoomed': test_images, 'default': test_images}
            }
        ]

    @pytest.fixture
    def llm_processor(self):
        """Create LLM processor."""
        return LLMProcessor(model="claude-3-7-sonnet-20250219")

    def test_create_merge_comparison_prompt(self, option_data_list, llm_processor):
        """Test merge comparison prompt creation."""
        content = create_merge_comparison_prompt(
            option_image_data=option_data_list,
            use_zoomed_images=True,
            views=['front', 'side', 'top'],
            llm_processor=llm_processor,
            zoom_margin=2048,
            prompt_mode='informative'
        )

        assert isinstance(content, list)
        assert len(content) > 0

        # Should have images from both options
        image_blocks = [c for c in content if c.get('type') == 'image']
        assert len(image_blocks) == 6  # 3 views × 2 options

        # Check for comparison language
        text_blocks = [c for c in content if c.get('type') == 'text']
        full_text = ' '.join([c['text'] for c in text_blocks])
        assert 'option' in full_text.lower()

    def test_create_split_comparison_prompt(self, option_data_list, llm_processor):
        """Test split comparison prompt creation."""
        content = create_split_comparison_prompt(
            positive_option_data=option_data_list[0],
            negative_option_data=option_data_list[1],
            use_zoomed_images=True,
            views=['front', 'side'],
            llm_processor=llm_processor,
            zoom_margin=2048,
            prompt_mode='informative'
        )

        assert isinstance(content, list)
        assert len(content) > 0

        # Should have images from both examples
        image_blocks = [c for c in content if c.get('type') == 'image']
        assert len(image_blocks) > 0


# ============================================================================
# Test Segment Classification Prompt
# ============================================================================

class TestCreateSegmentClassificationPrompt:
    """Test create_segment_classification_prompt function."""

    @pytest.fixture
    def test_images(self, tmp_path):
        """Create test images."""
        paths = []
        for view in ['front', 'side', 'top']:
            img = Image.new('RGB', (100, 100), color='purple')
            img_path = tmp_path / f"seg_{view}.png"
            img.save(img_path)
            paths.append(str(img_path))
        return paths

    @pytest.fixture
    def llm_processor(self):
        """Create LLM processor."""
        return LLMProcessor(model="gpt-4o")

    def test_create_segmentation_prompt_with_description(self, test_images, llm_processor):
        """Test segmentation classification prompt with description."""
        import numpy as np

        minpt = np.array([0, 0, 0])
        maxpt = np.array([1000, 1000, 1000])

        content = create_segment_classification_prompt(
            segment_images_paths=test_images,
            minpt=minpt,
            maxpt=maxpt,
            llm_processor=llm_processor,
            species='mouse',
            add_guidance=True
        )

        assert isinstance(content, list)
        assert len(content) > 0

        # Should include images
        if llm_processor.model in ['gpt-4o']:
            image_blocks = [c for c in content if c.get('type') == 'input_image']
        else:
            image_blocks = [c for c in content if c.get('type') == 'image']
        assert len(image_blocks) == 3

        # Check for description text
        if llm_processor.model in ['gpt-4o']:
            text_blocks = [c for c in content if c.get('type') == 'input_text']
        else:
            text_blocks = [c for c in content if c.get('type') == 'text']
        full_text = ' '.join([c.get('text', '') for c in text_blocks])
        assert 'segment' in full_text.lower()

    def test_create_segmentation_prompt_without_description(self, test_images, llm_processor):
        """Test segmentation classification prompt without description."""
        import numpy as np

        minpt = np.array([0, 0, 0])
        maxpt = np.array([1000, 1000, 1000])

        content = create_segment_classification_prompt(
            segment_images_paths=test_images,
            minpt=minpt,
            maxpt=maxpt,
            llm_processor=llm_processor,
            species='fly',
            add_guidance=False
        )

        assert isinstance(content, list)
        assert len(content) > 0


# ============================================================================
# Test MERGE_HEURISTICS
# ============================================================================

class TestMergeHeuristics:
    """Test MERGE_HEURISTICS dictionary."""

    def test_heuristics_exist(self):
        """Test that heuristics dictionary contains expected keys."""
        assert 'heuristic1' in MERGE_HEURISTICS
        assert 'heuristic2' in MERGE_HEURISTICS
        assert 'heuristic3' in MERGE_HEURISTICS
        assert 'heuristic4' in MERGE_HEURISTICS

    def test_heuristics_are_strings(self):
        """Test that all heuristics are non-empty strings."""
        for key, value in MERGE_HEURISTICS.items():
            assert isinstance(value, str)
            assert len(value) > 0

    def test_heuristic_content(self):
        """Test that heuristics contain expected keywords."""
        # Each heuristic should mention segments or merge
        for key, value in MERGE_HEURISTICS.items():
            text_lower = value.lower()
            assert 'segment' in text_lower or 'merge' in text_lower or \
                   'blue' in text_lower or 'orange' in text_lower


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
