"""Tests for analysis_utils.py utility functions."""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from pathlib import Path

from src.analysis_utils import (
    load_results,
    bootstrap_confidence_interval,
    bootstrap_classification_metrics,
    calculate_merge_identification_metrics,
    analyze_by_group,
    analyze_error_patterns,
    perform_majority_voting,
    analyze_voting_patterns,
    extract_heuristics_from_prompt_mode,
    analyze_heuristic_combinations,
    format_metric_with_ci,
    convert_numpy_types,
    clean_report_for_json
)


# ============================================================================
# Test Data Loading
# ============================================================================

class TestLoadResults:
    """Test load_results function."""

    def test_load_csv(self, tmp_path):
        """Test loading results from CSV file."""
        # Create test CSV
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value': ['a', 'b', 'c']
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        # Load and verify
        loaded = load_results(str(csv_path))
        pd.testing.assert_frame_equal(loaded, df)

    def test_load_json(self, tmp_path):
        """Test loading results from JSON file."""
        # Create test JSON
        data = [
            {'id': 1, 'value': 'a'},
            {'id': 2, 'value': 'b'}
        ]
        json_path = tmp_path / "test.json"
        with open(json_path, 'w') as f:
            json.dump(data, f)

        # Load and verify
        loaded = load_results(str(json_path))
        assert len(loaded) == 2
        assert list(loaded.columns) == ['id', 'value']

    def test_load_invalid_format(self, tmp_path):
        """Test that invalid file format raises ValueError."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("test")

        with pytest.raises(ValueError, match="File must be CSV or JSON"):
            load_results(str(txt_path))


# ============================================================================
# Test Bootstrap Functions
# ============================================================================

class TestBootstrapFunctions:
    """Test bootstrap confidence interval functions."""

    def test_bootstrap_confidence_interval(self):
        """Test basic bootstrap confidence interval calculation."""
        np.random.seed(42)
        data = np.random.normal(10, 2, size=100)

        def mean_func(x):
            return np.mean(x)

        lower, upper = bootstrap_confidence_interval(data, mean_func, n_bootstrap=100)

        # Check that CI contains the mean
        assert lower < np.mean(data) < upper
        # Check reasonable width
        assert 0 < (upper - lower) < 2

    def test_bootstrap_classification_metrics(self):
        """Test bootstrap for classification metrics."""
        # Perfect classifier
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0] * 10)
        y_pred = np.array([1, 1, 0, 0, 1, 0, 1, 0] * 10)

        ci_results = bootstrap_classification_metrics(y_true, y_pred, n_bootstrap=100)

        # Check all metrics are present
        assert 'accuracy' in ci_results
        assert 'precision' in ci_results
        assert 'recall' in ci_results
        assert 'f1_score' in ci_results
        assert 'specificity' in ci_results

        # For perfect classifier, CIs should be tight around 1.0
        for metric, (lower, upper) in ci_results.items():
            assert 0.9 <= lower <= 1.0
            assert 0.9 <= upper <= 1.0

    def test_bootstrap_with_imperfect_classifier(self):
        """Test bootstrap with realistic classifier performance."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=100)
        y_pred = y_true.copy()
        # Add some errors
        error_indices = np.random.choice(100, size=20, replace=False)
        y_pred[error_indices] = 1 - y_pred[error_indices]

        ci_results = bootstrap_classification_metrics(y_true, y_pred, n_bootstrap=100)

        # Accuracy should be around 0.8
        acc_lower, acc_upper = ci_results['accuracy']
        assert 0.6 < acc_lower < acc_upper < 1.0


# ============================================================================
# Test Metrics Calculation
# ============================================================================

class TestCalculateMetrics:
    """Test calculate_merge_identification_metrics function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing."""
        return pd.DataFrame({
            'operation_id': range(100),
            'id': range(100),
            'model_prediction': ['1'] * 70 + ['-1'] * 30,  # 70 predicted positive
            'is_correct_merge': [True] * 60 + [False] * 40  # 60 actual positive
        })

    def test_basic_metrics(self, sample_df):
        """Test basic metric calculation without CI."""
        metrics = calculate_merge_identification_metrics(sample_df, include_ci=False)

        # Check required keys
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'specificity' in metrics
        assert 'total_samples' in metrics

        # Verify counts
        assert metrics['total_samples'] == 100
        assert metrics['true_positives'] + metrics['false_negatives'] == 60
        assert metrics['true_negatives'] + metrics['false_positives'] == 40

    def test_metrics_with_ci(self, sample_df):
        """Test metric calculation with confidence intervals."""
        metrics = calculate_merge_identification_metrics(sample_df, include_ci=True)

        # Check CI keys exist
        assert 'accuracy_ci_lower' in metrics
        assert 'accuracy_ci_upper' in metrics
        assert 'confidence_level' in metrics

        # Verify CI bounds
        assert metrics['accuracy_ci_lower'] < metrics['accuracy']
        assert metrics['accuracy'] < metrics['accuracy_ci_upper']

    def test_perfect_classifier(self):
        """Test metrics for perfect classifier."""
        df = pd.DataFrame({
            'operation_id': range(50),
            'id': range(50),
            'model_prediction': ['1'] * 25 + ['-1'] * 25,
            'is_correct_merge': [True] * 25 + [False] * 25
        })

        metrics = calculate_merge_identification_metrics(df, include_ci=False)

        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
        assert metrics['specificity'] == 1.0

    def test_all_positive_predictions(self):
        """Test metrics when model predicts all positive."""
        df = pd.DataFrame({
            'operation_id': range(50),
            'id': range(50),
            'model_prediction': ['1'] * 50,
            'is_correct_merge': [True] * 30 + [False] * 20
        })

        metrics = calculate_merge_identification_metrics(df, include_ci=False)

        assert metrics['recall'] == 1.0  # Catches all positives
        assert metrics['specificity'] == 0.0  # Catches no negatives
        assert 0 < metrics['precision'] < 1  # Precision is 30/50


# ============================================================================
# Test Error Analysis
# ============================================================================

class TestErrorAnalysis:
    """Test error analysis functions."""

    @pytest.fixture
    def error_df(self):
        """Create dataframe with known errors."""
        return pd.DataFrame({
            'operation_id': range(10),
            'id': [f'seg_{i}' for i in range(10)],
            'model_prediction': ['1', '1', '1', '1', '1', '-1', '-1', '-1', '-1', '-1'],
            'is_correct_merge': [True, True, False, False, False, True, True, True, False, False],
            'model_analysis': ['good'] * 10
        })

    def test_analyze_error_patterns(self, error_df):
        """Test error pattern analysis."""
        patterns = analyze_error_patterns(error_df, id_column='id')

        # Check counts
        assert patterns['false_positive_count'] == 3  # Predicted merge but shouldn't
        assert patterns['false_negative_count'] == 3  # Predicted no merge but should

        # Check examples are included
        assert 'false_positive_examples' in patterns
        assert 'false_negative_examples' in patterns

    def test_analyze_error_patterns_with_base_neuron_id(self):
        """Test error analysis with base_neuron_id column."""
        df = pd.DataFrame({
            'operation_id': range(5),
            'base_neuron_id': [f'neuron_{i}' for i in range(5)],
            'model_prediction': ['1', '1', '-1', '-1', '1'],
            'is_correct_merge': [True, False, True, False, True]
        })

        patterns = analyze_error_patterns(df, id_column='base_neuron_id')
        assert patterns['false_positive_count'] == 1
        assert patterns['false_negative_count'] == 1

    def test_analyze_error_patterns_invalid_column(self, error_df):
        """Test that invalid column raises ValueError."""
        with pytest.raises(ValueError, match="Column 'invalid_col' not found"):
            analyze_error_patterns(error_df, id_column='invalid_col')


# ============================================================================
# Test Majority Voting
# ============================================================================

class TestMajorityVoting:
    """Test majority voting functions."""

    @pytest.fixture
    def voting_df(self):
        """Create dataframe with multiple runs."""
        data = []
        for op_id in range(5):
            for seg_id in ['A', 'B']:
                for run in range(5):
                    # 3 votes for '1', 2 votes for '-1'
                    prediction = '1' if run < 3 else '-1'
                    data.append({
                        'operation_id': op_id,
                        'id': seg_id,
                        'model_prediction': prediction,
                        'is_correct_merge': True,
                        'model_analysis': 'test'
                    })
        return pd.DataFrame(data)

    def test_perform_majority_voting(self, voting_df):
        """Test majority voting reduces multiple runs."""
        result = perform_majority_voting(voting_df, id_column='id')

        # Should have 10 rows (5 ops * 2 segments)
        assert len(result) == 10

        # Check majority prediction is correct
        assert all(result['model_prediction'] == '1')

        # Check confidence
        assert all(result['majority_confidence'] == 0.6)  # 3/5

    def test_unanimous_voting(self):
        """Test unanimous voting results."""
        df = pd.DataFrame({
            'operation_id': [0, 0, 0],
            'id': ['A', 'A', 'A'],
            'model_prediction': ['1', '1', '1'],
            'is_correct_merge': [True, True, True]
        })

        result = perform_majority_voting(df, id_column='id')

        assert len(result) == 1
        assert result.iloc[0]['majority_confidence'] == 1.0
        assert result.iloc[0]['model_prediction'] == '1'

    def test_analyze_voting_patterns(self, voting_df):
        """Test voting pattern analysis."""
        majority_df = perform_majority_voting(voting_df, id_column='id')
        analysis = analyze_voting_patterns(majority_df)

        assert 'confidence_stats' in analysis
        stats = analysis['confidence_stats']
        assert 'mean_confidence' in stats
        assert 'unanimous_decisions' in stats
        assert 'split_decisions' in stats

        # All decisions should be split (0.6 confidence)
        assert stats['split_decisions'] == 10
        assert stats['unanimous_decisions'] == 0


# ============================================================================
# Test Heuristic Analysis
# ============================================================================

class TestHeuristicAnalysis:
    """Test heuristic-related functions."""

    def test_extract_heuristics_from_prompt_mode(self):
        """Test heuristic extraction from prompt mode string."""
        # No heuristics
        assert extract_heuristics_from_prompt_mode('informative') == []

        # Single heuristic
        result = extract_heuristics_from_prompt_mode('informative+heuristic1')
        assert result == ['heuristic1']

        # Multiple heuristics
        result = extract_heuristics_from_prompt_mode('null+heuristic1+heuristic3')
        assert result == ['heuristic1', 'heuristic3']

        # Non-heuristic additions ignored
        result = extract_heuristics_from_prompt_mode('informative+other+heuristic2')
        assert result == ['heuristic2']

    def test_analyze_heuristic_combinations(self):
        """Test heuristic combination analysis."""
        df = pd.DataFrame({
            'operation_id': range(30),
            'id': range(30),
            'prompt_mode': ['informative'] * 10 +
                          ['informative+heuristic1'] * 10 +
                          ['informative+heuristic1+heuristic2'] * 10,
            'model_prediction': ['1'] * 20 + ['-1'] * 10,
            'is_correct_merge': [True] * 15 + [False] * 15
        })

        analysis = analyze_heuristic_combinations(df, include_ci=False)

        assert 'by_heuristic_count' in analysis
        assert 'by_combination' in analysis

        # Check counts
        by_count = analysis['by_heuristic_count']
        assert '0_heuristics' in by_count
        assert '1_heuristics' in by_count
        assert '2_heuristics' in by_count


# ============================================================================
# Test Formatting Utilities
# ============================================================================

class TestFormattingUtilities:
    """Test formatting and type conversion utilities."""

    def test_format_metric_with_ci(self):
        """Test metric formatting with confidence intervals."""
        metrics = {
            'accuracy': 0.856,
            'accuracy_ci_lower': 0.823,
            'accuracy_ci_upper': 0.891,
            'confidence_level': 0.95
        }

        formatted = format_metric_with_ci('accuracy', metrics)
        assert '0.856' in formatted
        assert '95% CI' in formatted
        assert '0.823' in formatted
        assert '0.891' in formatted

    def test_format_metric_without_ci(self):
        """Test metric formatting without CI."""
        metrics = {'accuracy': 0.856}
        formatted = format_metric_with_ci('accuracy', metrics)
        assert formatted == '0.856'

    def test_convert_numpy_types(self):
        """Test numpy type conversion."""
        assert isinstance(convert_numpy_types(np.int64(5)), int)
        assert isinstance(convert_numpy_types(np.float32(5.5)), float)
        assert isinstance(convert_numpy_types(np.array([1, 2, 3])), list)
        assert convert_numpy_types('string') == 'string'

    def test_clean_report_for_json(self):
        """Test nested numpy type conversion."""
        data = {
            'count': np.int64(10),
            'accuracy': np.float32(0.95),
            'values': np.array([1, 2, 3]),
            'nested': {
                'metric': np.float64(0.85)
            },
            'list': [np.int32(1), np.int32(2)]
        }

        cleaned = clean_report_for_json(data)

        # Verify all types are JSON-serializable
        json.dumps(cleaned)  # Should not raise

        assert isinstance(cleaned['count'], int)
        assert isinstance(cleaned['accuracy'], float)
        assert isinstance(cleaned['values'], list)
        assert isinstance(cleaned['nested']['metric'], float)


# ============================================================================
# Test Group Analysis
# ============================================================================

class TestGroupAnalysis:
    """Test analyze_by_group function."""

    def test_analyze_by_group(self):
        """Test performance analysis by group."""
        df = pd.DataFrame({
            'operation_id': range(20),
            'id': range(20),
            'model': ['gpt-4o'] * 10 + ['claude'] * 10,
            'model_prediction': ['1'] * 15 + ['-1'] * 5,
            'is_correct_merge': [True] * 12 + [False] * 8
        })

        results = analyze_by_group(df, 'model', include_ci=False)

        assert 'gpt-4o' in results
        assert 'claude' in results
        assert results['gpt-4o']['total_samples'] == 10
        assert results['claude']['total_samples'] == 10

    def test_analyze_by_group_skips_nan(self):
        """Test that NaN group values are skipped."""
        df = pd.DataFrame({
            'operation_id': range(15),
            'id': range(15),
            'species': ['mouse'] * 5 + ['fly'] * 5 + [np.nan] * 5,
            'model_prediction': ['1'] * 10 + ['-1'] * 5,
            'is_correct_merge': [True] * 8 + [False] * 7
        })

        results = analyze_by_group(df, 'species', include_ci=False)

        assert 'mouse' in results
        assert 'fly' in results
        assert 'nan' not in results
        assert len(results) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
