#!/usr/bin/env python3
"""
Utility functions for merge identification analysis.
Shared across analysis and visualization scripts.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional


# ============================================================================
# Data Loading
# ============================================================================

def load_results(file_path: str) -> pd.DataFrame:
    """Load results from CSV or JSON file."""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        # Replace NaN in prompt_mode column with "null" string
        # This handles cases where "null" was written to CSV but read as NaN
        if 'prompt_mode' in df.columns:
            df['prompt_mode'] = df['prompt_mode'].fillna('null')
        return df
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        raise ValueError("File must be CSV or JSON format")


# ============================================================================
# Bootstrap Confidence Intervals
# ============================================================================

def bootstrap_confidence_interval(data: np.ndarray, metric_func,
                                 confidence_level: float = 0.95,
                                 n_bootstrap: int = 1000) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for a metric.

    Args:
        data: Input data array
        metric_func: Function to calculate metric (should take data and return scalar)
        confidence_level: Confidence level (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    bootstrap_stats = []
    n_samples = len(data)

    np.random.seed(42)  # For reproducibility

    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=n_samples, replace=True)
        bootstrap_stat = metric_func(bootstrap_sample)
        bootstrap_stats.append(bootstrap_stat)

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)

    return lower_bound, upper_bound


def bootstrap_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                    confidence_level: float = 0.95,
                                    n_bootstrap: int = 1000) -> Dict[str, Tuple[float, float]]:
    """
    Calculate bootstrap confidence intervals for classification metrics.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        confidence_level: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dictionary with confidence intervals for each metric
    """
    def accuracy_func(indices):
        return np.mean(y_true[indices] == y_pred[indices])

    def precision_func(indices):
        tp = np.sum((y_pred[indices] == 1) & (y_true[indices] == 1))
        fp = np.sum((y_pred[indices] == 1) & (y_true[indices] == 0))
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    def recall_func(indices):
        tp = np.sum((y_pred[indices] == 1) & (y_true[indices] == 1))
        fn = np.sum((y_pred[indices] == 0) & (y_true[indices] == 1))
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    def f1_func(indices):
        prec = precision_func(indices)
        rec = recall_func(indices)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    def specificity_func(indices):
        tn = np.sum((y_pred[indices] == 0) & (y_true[indices] == 0))
        fp = np.sum((y_pred[indices] == 1) & (y_true[indices] == 0))
        return tn / (tn + fp) if (tn + fp) > 0 else 0

    n_samples = len(y_true)
    bootstrap_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'specificity': []
    }

    np.random.seed(42)  # For reproducibility

    for _ in range(n_bootstrap):
        # Generate bootstrap indices
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)

        # Calculate metrics for this bootstrap sample
        bootstrap_results['accuracy'].append(accuracy_func(bootstrap_indices))
        bootstrap_results['precision'].append(precision_func(bootstrap_indices))
        bootstrap_results['recall'].append(recall_func(bootstrap_indices))
        bootstrap_results['f1_score'].append(f1_func(bootstrap_indices))
        bootstrap_results['specificity'].append(specificity_func(bootstrap_indices))

    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    confidence_intervals = {}
    for metric, values in bootstrap_results.items():
        lower_bound = np.percentile(values, lower_percentile)
        upper_bound = np.percentile(values, upper_percentile)
        confidence_intervals[metric] = (lower_bound, upper_bound)

    return confidence_intervals


# ============================================================================
# Core Metrics Calculation
# ============================================================================

def calculate_merge_identification_metrics(df: pd.DataFrame,
                                          include_ci: bool = True,
                                          confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Calculate performance metrics for merge identification task.

    For merge identification:
    - Positive class (1): Segment should be merged (is_correct_merge = True)
    - Negative class (0): Segment should not be merged (is_correct_merge = False)

    Args:
        df: DataFrame with results
        include_ci: Whether to include bootstrap confidence intervals
        confidence_level: Confidence level for intervals (default 0.95)

    Returns:
        Dictionary with metrics and optional confidence intervals
    """
    metrics = {}

    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Convert model predictions to binary (1 for merge, 0 for no merge)
    df['model_pred_binary'] = (df['model_prediction'] == '1').astype(int)
    df['ground_truth_binary'] = df['is_correct_merge'].astype(int)

    # Basic counts
    total_samples = len(df)

    # True/False Positives/Negatives
    tp = ((df['model_pred_binary'] == 1) & (df['ground_truth_binary'] == 1)).sum()
    tn = ((df['model_pred_binary'] == 0) & (df['ground_truth_binary'] == 0)).sum()
    fp = ((df['model_pred_binary'] == 1) & (df['ground_truth_binary'] == 0)).sum()
    fn = ((df['model_pred_binary'] == 0) & (df['ground_truth_binary'] == 1)).sum()

    # Basic metrics
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Additional metrics
    positive_rate = (df['ground_truth_binary'] == 1).mean()
    predicted_positive_rate = (df['model_pred_binary'] == 1).mean()

    metrics.update({
        'total_samples': int(total_samples),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'specificity': float(specificity),
        'positive_rate': float(positive_rate),
        'predicted_positive_rate': float(predicted_positive_rate)
    })

    # Add bootstrap confidence intervals if requested
    if include_ci and total_samples > 10:  # Only calculate CI if we have enough samples
        try:
            y_true = df['ground_truth_binary'].values
            y_pred = df['model_pred_binary'].values

            confidence_intervals = bootstrap_classification_metrics(
                y_true, y_pred, confidence_level=confidence_level
            )

            # Add confidence intervals to metrics
            for metric_name, (lower, upper) in confidence_intervals.items():
                metrics[f'{metric_name}_ci_lower'] = float(lower)
                metrics[f'{metric_name}_ci_upper'] = float(upper)
                metrics[f'{metric_name}_ci_width'] = float(upper - lower)

            metrics['confidence_level'] = confidence_level

        except Exception as e:
            # If bootstrap fails, continue without CI
            print(f"Warning: Could not calculate confidence intervals: {e}")

    return metrics


def calculate_merge_comparison_metrics(df: pd.DataFrame,
                                       include_ci: bool = True,
                                       confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Calculate performance metrics for merge comparison task.

    For merge comparison:
    - Correct: model_chosen_id matches one of the IDs in correct_answer
    - Incorrect: model_chosen_id does not match or is 'none'

    Args:
        df: DataFrame with results (must have 'correct_answer' and 'model_chosen_id' columns)
        include_ci: Whether to include bootstrap confidence intervals
        confidence_level: Confidence level for intervals (default 0.95)

    Returns:
        Dictionary with metrics and optional confidence intervals
    """
    import ast

    metrics = {}

    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Parse correct_answer if it's a string representation of a list
    def parse_correct_answer(answer):
        if pd.isna(answer):
            return []
        if isinstance(answer, str):
            try:
                # Handle string representation of list
                parsed = ast.literal_eval(answer)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
                else:
                    return [str(parsed)]
            except (ValueError, SyntaxError):
                # If parsing fails, treat as single value
                return [str(answer)]
        elif isinstance(answer, list):
            return [str(x) for x in answer]
        else:
            return [str(answer)]

    # Determine correctness
    def is_correct(row):
        correct_ids = parse_correct_answer(row['correct_answer'])
        chosen_id = str(row['model_chosen_id']) if not pd.isna(row['model_chosen_id']) else 'none'

        # Handle 'none' explicitly
        if chosen_id == 'none' or chosen_id == 'nan':
            return False

        # Check if chosen ID is in correct answers
        return chosen_id in correct_ids

    df['is_correct'] = df.apply(is_correct, axis=1)

    # Basic counts
    total_samples = len(df)
    correct_count = df['is_correct'].sum()
    incorrect_count = total_samples - correct_count

    # Calculate accuracy
    accuracy = correct_count / total_samples if total_samples > 0 else 0

    # For merge comparison, we treat it as a binary classification where:
    # - True Positive: Model chose correct option (is_correct = True)
    # - False Negative: Model chose wrong option or none (is_correct = False)
    # We can also analyze by whether it's a split operation

    metrics.update({
        'total_samples': int(total_samples),
        'correct': int(correct_count),
        'incorrect': int(incorrect_count),
        'accuracy': float(accuracy),
    })

    # Count 'none' responses (model couldn't decide)
    none_count = (df['model_chosen_id'].astype(str).isin(['none', 'nan'])).sum()
    metrics['none_responses'] = int(none_count)
    metrics['none_rate'] = float(none_count / total_samples) if total_samples > 0 else 0

    # If is_split column exists, analyze by split status
    if 'is_split' in df.columns:
        split_df = df[df['is_split'] == True]
        non_split_df = df[df['is_split'] == False]

        if len(split_df) > 0:
            metrics['split_accuracy'] = float(split_df['is_correct'].mean())
            metrics['split_samples'] = int(len(split_df))

        if len(non_split_df) > 0:
            metrics['non_split_accuracy'] = float(non_split_df['is_correct'].mean())
            metrics['non_split_samples'] = int(len(non_split_df))

    # Add bootstrap confidence intervals if requested
    if include_ci and total_samples > 10:
        try:
            # Bootstrap confidence interval for accuracy
            indices = np.arange(len(df))

            def accuracy_func(boot_indices):
                return df.iloc[boot_indices]['is_correct'].mean()

            ci_lower, ci_upper = bootstrap_confidence_interval(
                indices, accuracy_func, confidence_level=confidence_level
            )

            metrics['accuracy_ci_lower'] = float(ci_lower)
            metrics['accuracy_ci_upper'] = float(ci_upper)
            metrics['accuracy_ci_width'] = float(ci_upper - ci_lower)
            metrics['confidence_level'] = confidence_level

        except Exception as e:
            # If bootstrap fails, continue without CI
            print(f"Warning: Could not calculate confidence intervals: {e}")

    return metrics


def analyze_by_group(df: pd.DataFrame, group_column: str,
                    include_ci: bool = True,
                    confidence_level: float = 0.95) -> Dict[str, Dict[str, float]]:
    """Analyze performance metrics grouped by a specific column."""
    results = {}

    for group_value in df[group_column].unique():
        if pd.isna(group_value):
            continue

        group_df = df[df[group_column] == group_value]
        results[str(group_value)] = calculate_merge_identification_metrics(
            group_df, include_ci=include_ci, confidence_level=confidence_level
        )

    return results


# ============================================================================
# Error Analysis
# ============================================================================

def analyze_error_patterns(df: pd.DataFrame, id_column: str = 'id') -> Dict[str, Any]:
    """
    Analyze patterns in incorrect predictions.

    Args:
        df: DataFrame with predictions
        id_column: Name of column containing segment/neuron IDs ('id' or 'base_neuron_id')
    """
    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in DataFrame. Available: {list(df.columns)}")

    df = df.copy()
    df['model_pred_binary'] = (df['model_prediction'] == '1').astype(int)
    df['ground_truth_binary'] = df['is_correct_merge'].astype(int)
    df['correct_prediction'] = (df['model_pred_binary'] == df['ground_truth_binary'])

    error_patterns = {}

    # False positives: Model said merge, but shouldn't merge
    false_positives = df[(df['model_pred_binary'] == 1) & (df['ground_truth_binary'] == 0)]
    error_patterns['false_positive_count'] = len(false_positives)

    # False negatives: Model said no merge, but should merge
    false_negatives = df[(df['model_pred_binary'] == 0) & (df['ground_truth_binary'] == 1)]
    error_patterns['false_negative_count'] = len(false_negatives)

    # Sample some examples if available
    if len(false_positives) > 0:
        columns = ['operation_id', id_column, 'model_analysis'] if 'model_analysis' in df.columns else ['operation_id', id_column]
        error_patterns['false_positive_examples'] = false_positives[columns].head(3).to_dict('records')

    if len(false_negatives) > 0:
        columns = ['operation_id', id_column, 'model_analysis'] if 'model_analysis' in df.columns else ['operation_id', id_column]
        error_patterns['false_negative_examples'] = false_negatives[columns].head(3).to_dict('records')

    return error_patterns


# ============================================================================
# Majority Voting
# ============================================================================

def perform_majority_voting(df: pd.DataFrame, id_column: str = 'id') -> pd.DataFrame:
    """
    Perform majority voting for multiple runs of the same prompt.
    Groups by unique identifier and takes majority vote of predictions.

    Args:
        df: DataFrame with predictions
        id_column: Name of column containing segment/neuron IDs ('id' or 'base_neuron_id')
    """
    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in DataFrame. Available: {list(df.columns)}")

    # Create unique identifier for each prompt instance
    # Use operation_id + segment_id to group multiple runs
    df['unique_id'] = df['operation_id'].astype(str) + '_' + df[id_column].astype(str)

    # Group by unique_id and aggregate
    majority_results = []

    for unique_id, group in df.groupby('unique_id'):
        # Take majority vote of predictions
        predictions = group['model_prediction'].tolist()

        # Count votes
        vote_counts = {}
        for pred in predictions:
            vote_counts[pred] = vote_counts.get(pred, 0) + 1

        # Get majority prediction
        majority_pred = max(vote_counts.keys(), key=lambda x: vote_counts[x])

        # Calculate confidence (percentage of votes for majority)
        majority_count = vote_counts[majority_pred]
        total_votes = len(predictions)
        confidence = majority_count / total_votes

        # Take first row as template and update with majority vote
        majority_row = group.iloc[0].copy()
        majority_row['model_prediction'] = majority_pred
        majority_row['majority_confidence'] = confidence
        majority_row['total_votes'] = total_votes
        majority_row['vote_distribution'] = str(vote_counts)
        majority_row['all_predictions'] = str(predictions)

        majority_results.append(majority_row)

    majority_df = pd.DataFrame(majority_results)

    print(f"Majority voting: Reduced {len(df)} individual predictions to {len(majority_df)} majority votes")
    print(f"Average confidence: {majority_df['majority_confidence'].mean():.3f}")

    return majority_df


def analyze_voting_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze patterns in majority voting results."""
    if 'majority_confidence' not in df.columns:
        return {}

    analysis = {}

    # Confidence distribution
    confidence_stats = {
        'mean_confidence': float(df['majority_confidence'].mean()),
        'median_confidence': float(df['majority_confidence'].median()),
        'min_confidence': float(df['majority_confidence'].min()),
        'max_confidence': float(df['majority_confidence'].max()),
        'unanimous_decisions': int((df['majority_confidence'] == 1.0).sum()),
        'split_decisions': int((df['majority_confidence'] < 1.0).sum())
    }

    analysis['confidence_stats'] = confidence_stats

    # Performance by confidence level
    confidence_bins = pd.cut(df['majority_confidence'], bins=[0, 0.6, 0.8, 1.0],
                           labels=['Low (≤0.6)', 'Medium (0.6-0.8)', 'High (1.0)'])

    performance_by_confidence = {}
    for conf_level in confidence_bins.cat.categories:
        conf_df = df[confidence_bins == conf_level]
        if len(conf_df) > 0:
            performance_by_confidence[conf_level] = calculate_merge_identification_metrics(
                conf_df, include_ci=False
            )

    analysis['performance_by_confidence'] = performance_by_confidence

    return analysis


# ============================================================================
# Heuristic Analysis
# ============================================================================

def extract_heuristics_from_prompt_mode(prompt_mode: str) -> List[str]:
    """Extract heuristics from prompt mode string."""

    # Handle NaN or non-string values (e.g., from CSVs with missing prompt_mode)
    if not isinstance(prompt_mode, str):
        return []

    if '+' not in prompt_mode:
        return []
    parts = prompt_mode.split('+')
    return [part for part in parts[1:] if part.startswith('heuristic')]


def analyze_heuristic_combinations(df: pd.DataFrame,
                                  include_ci: bool = True,
                                  confidence_level: float = 0.95) -> Dict[str, Any]:
    """Analyze performance by heuristic combinations."""
    if 'prompt_mode' not in df.columns:
        return {}

    heuristic_analysis = {}

    # Group by heuristic combinations
    df_copy = df.copy()
    df_copy['heuristics'] = df_copy['prompt_mode'].apply(extract_heuristics_from_prompt_mode)
    df_copy['heuristic_count'] = df_copy['heuristics'].apply(len)
    df_copy['heuristic_str'] = df_copy['heuristics'].apply(lambda x: '+'.join(sorted(x)) if x else 'none')

    # Analyze by number of heuristics
    by_count = {}
    for count in sorted(df_copy['heuristic_count'].unique()):
        count_df = df_copy[df_copy['heuristic_count'] == count]
        by_count[f"{count}_heuristics"] = calculate_merge_identification_metrics(
            count_df, include_ci=include_ci, confidence_level=confidence_level
        )

    heuristic_analysis['by_heuristic_count'] = by_count

    # Analyze by specific combinations
    by_combination = {}
    for combination in df_copy['heuristic_str'].unique():
        combo_df = df_copy[df_copy['heuristic_str'] == combination]
        by_combination[combination] = calculate_merge_identification_metrics(
            combo_df, include_ci=include_ci, confidence_level=confidence_level
        )

    heuristic_analysis['by_combination'] = by_combination

    return heuristic_analysis


# ============================================================================
# Formatting Utilities
# ============================================================================

def format_metric_with_ci(metric_name: str, metrics: Dict) -> str:
    """Format a metric value with its confidence interval if available."""
    value = metrics[metric_name]
    ci_lower_key = f"{metric_name}_ci_lower"
    ci_upper_key = f"{metric_name}_ci_upper"

    if ci_lower_key in metrics and ci_upper_key in metrics:
        ci_lower = metrics[ci_lower_key]
        ci_upper = metrics[ci_upper_key]
        confidence_level = metrics.get('confidence_level', 0.95)
        ci_pct = int(confidence_level * 100)
        return f"{value:.3f} ({ci_pct}% CI: {ci_lower:.3f}-{ci_upper:.3f})"
    else:
        return f"{value:.3f}"


def convert_numpy_types(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def clean_report_for_json(data):
    """Recursively convert numpy types in nested dictionaries."""
    if isinstance(data, dict):
        return {k: clean_report_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_report_for_json(item) for item in data]
    else:
        return convert_numpy_types(data)
