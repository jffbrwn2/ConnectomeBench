"""Tests for util.py LLMProcessor and utility functions."""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import base64
from pathlib import Path
import tempfile
from PIL import Image
import io

from src.util import (
    LLMProcessor,
    evaluate_response,
    create_unified_result_structure,
    openai_models,
    anthropic_models
)


# ============================================================================
# Test LLMProcessor Initialization
# ============================================================================

class TestLLMProcessorInit:
    """Test LLMProcessor initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        processor = LLMProcessor()
        assert processor.model == "gpt-4o"
        assert processor.max_concurrent == 25
        assert processor.max_tokens == 4096
        assert processor.parsable == (processor.model in ["gpt-4o", "o1"])

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        processor = LLMProcessor(
            model="claude-3-7-sonnet-20250219",
            max_concurrent=10,
            max_tokens=8192
        )
        assert processor.model == "claude-3-7-sonnet-20250219"
        assert processor.max_concurrent == 10
        assert processor.max_tokens == 8192
        assert processor.parsable == False

    def test_parsable_model(self):
        """Test parsable flag for supported models."""
        processor = LLMProcessor(model="gpt-4o")
        assert processor.parsable == True

        processor = LLMProcessor(model="o1")
        assert processor.parsable == True

        processor = LLMProcessor(model="claude-3-7-sonnet-20250219")
        assert processor.parsable == False


# ============================================================================
# Test Image Encoding
# ============================================================================

class TestImageEncoding:
    """Test image encoding functionality."""

    @pytest.fixture
    def test_image_path(self, tmp_path):
        """Create a test PNG image."""
        img = Image.new('RGB', (100, 100), color='red')
        img_path = tmp_path / "test.png"
        img.save(img_path)
        return str(img_path)

    @pytest.fixture
    def test_jpeg_path(self, tmp_path):
        """Create a test JPEG image."""
        img = Image.new('RGB', (100, 100), color='blue')
        img_path = tmp_path / "test.jpg"
        img.save(img_path)
        return str(img_path)

    def test_encode_png_image(self, test_image_path):
        """Test encoding PNG image from file path."""
        processor = LLMProcessor()
        base64_data, media_type = processor._encode_image_to_base64(test_image_path)

        assert isinstance(base64_data, str)
        assert media_type == "image/png"
        # Verify it's valid base64
        decoded = base64.b64decode(base64_data)
        assert len(decoded) > 0

    def test_encode_jpeg_image(self, test_jpeg_path):
        """Test encoding JPEG image from file path."""
        processor = LLMProcessor()
        base64_data, media_type = processor._encode_image_to_base64(test_jpeg_path)

        assert isinstance(base64_data, str)
        assert media_type == "image/jpeg"

    def test_encode_image_with_url(self):
        """Test encoding image from URL (mocked)."""
        processor = LLMProcessor()

        # Create a mock image response
        img = Image.new('RGB', (10, 10), color='green')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        with patch('httpx.get') as mock_get:
            mock_response = Mock()
            mock_response.content = img_bytes.read()
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            base64_data, media_type = processor._encode_image_to_base64('http://example.com/image.png')

            assert isinstance(base64_data, str)
            assert 'image/' in media_type

    def test_encode_image_missing_file(self):
        """Test encoding non-existent file raises error."""
        processor = LLMProcessor()

        with pytest.raises(FileNotFoundError):
            processor._encode_image_to_base64('/nonexistent/path/image.png')


# ============================================================================
# Test Message Formatting
# ============================================================================

class TestMessageFormatting:
    """Test message formatting functionality."""

    def test_format_string_message(self):
        """Test formatting a simple string message."""
        processor = LLMProcessor()
        messages = processor._format_message_with_images("Test prompt")

        assert len(messages) == 1
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == "Test prompt"

    def test_format_list_message(self):
        """Test formatting a list of content blocks."""
        processor = LLMProcessor()
        content = [
            {"type": "text", "text": "Test"},
            {"type": "image", "source": {"type": "base64", "data": "abc123"}}
        ]
        messages = processor._format_message_with_images(content)

        assert len(messages) == 1
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == content


# ============================================================================
# Test Response Evaluation
# ============================================================================

class TestEvaluateResponse:
    """Test evaluate_response function."""

    def test_valid_response_with_tags(self):
        """Test parsing valid response with both tags."""
        response = """
        <analysis>This is the analysis text.</analysis>
        <answer>1</answer>
        """
        result = evaluate_response(response)

        assert result['analysis'] == "This is the analysis text."
        assert result['answer'] == "1"

    def test_response_with_none(self):
        """Test response with 'none' answer."""
        response = """
        <analysis>Cannot determine.</analysis>
        <answer>none</answer>
        """
        result = evaluate_response(response)

        assert result['analysis'] == "Cannot determine."
        assert result['answer'] == "none"

    def test_response_with_minus_one(self):
        """Test response with '-1' answer (treated as none)."""
        response = """
        <analysis>Not applicable.</analysis>
        <answer>-1</answer>
        """
        result = evaluate_response(response)

        assert result['answer'] == "none"

    def test_response_with_positive_integer(self):
        """Test response with positive integer."""
        response = """
        <analysis>Option 2 is best.</analysis>
        <answer>2</answer>
        """
        result = evaluate_response(response)

        assert result['answer'] == "2"

    def test_response_with_invalid_integer(self):
        """Test response with invalid integer (negative)."""
        response = """
        <analysis>Invalid.</analysis>
        <answer>-5</answer>
        """
        result = evaluate_response(response)

        # Negative integers should be treated as none
        assert result['answer'] == "none"

    def test_response_missing_analysis_tags(self, capsys):
        """Test response without analysis tags."""
        response = """
        Some text without tags.
        <answer>1</answer>
        """
        result = evaluate_response(response)

        assert result['analysis'] == "Analysis tags not found in response."
        assert result['answer'] == "1"

        captured = capsys.readouterr()
        assert "Warning: Could not find <analysis> tags" in captured.out

    def test_response_missing_answer_tags(self, capsys):
        """Test response without answer tags."""
        response = """
        <analysis>Some analysis.</analysis>
        No answer tags here.
        """
        result = evaluate_response(response)

        assert result['analysis'] == "Some analysis."
        assert result['answer'] == "none"

        captured = capsys.readouterr()
        assert "Warning: Could not find <answer> tags" in captured.out

    def test_response_with_unparseable_answer(self, capsys):
        """Test response with unparseable answer."""
        response = """
        <analysis>Analysis.</analysis>
        <answer>not_a_number</answer>
        """
        result = evaluate_response(response)

        assert result['answer'] == "none"

        captured = capsys.readouterr()
        assert "Warning: Could not parse model answer" in captured.out

    def test_case_insensitive_answer(self):
        """Test that answer parsing is case insensitive."""
        response = """
        <analysis>Analysis.</analysis>
        <answer>NONE</answer>
        """
        result = evaluate_response(response)

        assert result['answer'] == "none"


# ============================================================================
# Test Unified Result Structure
# ============================================================================

class TestCreateUnifiedResultStructure:
    """Test create_unified_result_structure function."""

    @pytest.fixture
    def base_event_result(self):
        """Create base event result for testing."""
        return {
            'operation_id': 'op_123',
            'base_neuron_id': 'neuron_456',
            'timestamp': 1234567890,
            'merge_coords': [100, 200, 300],
            'interface_point': [100, 200, 300],
            'before_root_ids': [111, 222],
            'after_root_ids': [333],
            'views': ['front', 'side', 'top'],
            'use_zoomed_images': True,
            'image_paths': {}
        }

    @pytest.fixture
    def answer_analysis(self):
        """Create sample answer analysis."""
        return {
            'answer': '1',
            'analysis': 'This is correct.'
        }

    def test_merge_comparison_result(self, base_event_result, answer_analysis):
        """Test result structure for merge comparison task."""
        base_event_result.update({
            'expected_choice_ids': ['seg_1'],
            'options_presented_ids': ['seg_1', 'seg_2'],
            'num_options_presented': 2,
            'correct_merged_pair': ['neuron_456', 'seg_1']
        })

        result = create_unified_result_structure(
            task='merge_comparison',
            event_result=base_event_result,
            response="<answer>1</answer>",
            answer_analysis=answer_analysis,
            index=0,
            model='gpt-4o',
            zoom_margin=2048,
            prompt_mode='informative'
        )

        assert result['task'] == 'merge_comparison'
        assert result['operation_id'] == 'op_123'
        assert result['model'] == 'gpt-4o'
        assert result['model_prediction'] == '1'
        assert result['zoom_margin'] == 2048
        assert result['correct_merged_pair'] == ['neuron_456', 'seg_1']

    def test_merge_identification_result(self, base_event_result, answer_analysis):
        """Test result structure for merge identification task."""
        base_event_result['expected_choice_ids'] = ['seg_1']

        option_data = {
            'id': 'seg_1',
            'paths': {}
        }

        result = create_unified_result_structure(
            task='merge_identification',
            event_result=base_event_result,
            option_data=option_data,
            response="<answer>1</answer>",
            answer_analysis=answer_analysis,
            index=0,
            model='claude-3-7-sonnet-20250219'
        )

        assert result['task'] == 'merge_identification'
        assert result['id'] == 'seg_1'
        assert result['is_correct_merge'] == True  # seg_1 is in expected_choice_ids
        assert result['model_answer'] == '1'

    def test_split_identification_result(self, base_event_result, answer_analysis):
        """Test result structure for split identification task."""
        option_data = {
            'id': '111',  # This is in before_root_ids
            'paths': {}
        }

        result = create_unified_result_structure(
            task='split_identification',
            event_result=base_event_result,
            option_data=option_data,
            response="<answer>1</answer>",
            answer_analysis=answer_analysis,
            index=0
        )

        assert result['task'] == 'split_identification'
        assert result['id'] == '111'
        assert result['is_split'] == True  # 111 is in before_root_ids

    def test_split_comparison_result(self, base_event_result, answer_analysis):
        """Test result structure for split comparison task."""
        base_event_result.update({
            'root_id_requires_split': 'split_123',
            'root_id_does_not_require_split': 'no_split_456'
        })

        result = create_unified_result_structure(
            task='split_comparison',
            event_result=base_event_result,
            response="<answer>1</answer>",
            answer_analysis=answer_analysis,
            correct_answer='1',
            index=0
        )

        assert result['task'] == 'split_comparison'
        assert result['root_id_requires_split'] == 'split_123'
        assert result['root_id_does_not_require_split'] == 'no_split_456'
        assert result['correct_answer'] == '1'

    def test_common_fields_present(self, base_event_result, answer_analysis):
        """Test that common fields are present in all task types."""
        for task in ['merge_comparison', 'merge_identification',
                    'split_comparison', 'split_identification']:

            # Use numeric ID for identification tasks (they need to be convertible to int)
            option_data = {'id': '123456'} if 'identification' in task else None

            result = create_unified_result_structure(
                task=task,
                event_result=base_event_result,
                option_data=option_data,
                response="test",
                answer_analysis=answer_analysis,
                index=0
            )

            # Common fields that should always be present
            assert 'task' in result
            assert 'operation_id' in result
            assert 'model' in result
            assert 'model_raw_answer' in result
            assert 'model_analysis' in result
            assert 'model_prediction' in result
            assert 'views' in result
            assert 'use_zoomed_images' in result
            assert 'zoom_margin' in result


# ============================================================================
# Test Batch Processing (Mocked)
# ============================================================================

class TestBatchProcessing:
    """Test batch processing with mocked API calls."""

    @pytest.mark.asyncio
    async def test_process_batch_success(self):
        """Test successful batch processing."""
        processor = LLMProcessor(model="gpt-4o")

        # Mock the _call_api_async method
        with patch.object(processor, '_call_api_async', new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = ["Response 1", "Response 2", "Response 3"]

            prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
            results = await processor.process_batch(prompts)

            assert len(results) == 3
            assert results[0] == "Response 1"
            assert results[1] == "Response 2"
            assert results[2] == "Response 3"
            assert mock_call.call_count == 3

    @pytest.mark.asyncio
    async def test_process_batch_with_errors(self):
        """Test batch processing with some errors."""
        processor = LLMProcessor(model="gpt-4o")

        # Mock with one error
        with patch.object(processor, '_call_api_async', new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = ["Response 1", Exception("API Error"), "Response 3"]

            prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
            results = await processor.process_batch(prompts)

            assert len(results) == 3
            assert results[0] == "Response 1"
            assert results[1] is None  # Error converted to None
            assert results[2] == "Response 3"

    @pytest.mark.asyncio
    async def test_process_single(self):
        """Test single prompt processing."""
        processor = LLMProcessor(model="claude-3-7-sonnet-20250219")

        with patch.object(processor, '_call_api_async', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "Single response"

            result = await processor.process_single("Single prompt")

            assert result == "Single response"
            assert mock_call.call_count == 1


# ============================================================================
# Test Model Type Categorization
# ============================================================================

class TestModelTypeCategorization:
    """Test that models are correctly categorized."""

    def test_openai_models_list(self):
        """Test openai_models list contains expected models."""
        assert "gpt-4o" in openai_models
        assert "gpt-4o-mini" in openai_models
        assert "o1" in openai_models
        assert "o1-mini" in openai_models

    def test_anthropic_models_list(self):
        """Test anthropic_models list contains expected models."""
        assert "claude-3-5-sonnet-20240620" in anthropic_models
        assert "claude-3-5-sonnet-20241022" in anthropic_models
        assert "claude-3-7-sonnet-20250219" in anthropic_models


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
