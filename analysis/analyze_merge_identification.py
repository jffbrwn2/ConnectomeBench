#!/usr/bin/env python3
"""
Script to analyze performance on merge_identification task results.
Calculates accuracy, precision, recall, F1-score, and other metrics.
"""

import pandas as pd
import numpy as np
import json
import argparse
import os
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy import stats

def bootstrap_confidence_interval(data: np.ndarray, metric_func, confidence_level: float = 0.95, n_bootstrap: int = 1000) -> Tuple[float, float]:
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

def bootstrap_pass_at_k_metrics(df: pd.DataFrame, k: int = 1, confidence_level: float = 0.95, n_bootstrap: int = 1000) -> Dict[str, Tuple[float, float]]:
    """
    Calculate bootstrap confidence intervals for pass@k metrics.
    
    Args:
        df: DataFrame with results
        k: Number of attempts to consider
        confidence_level: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        Dictionary with confidence intervals for pass@k
    """
    def pass_at_k_func(bootstrap_df):
        pass_metrics = calculate_pass_at_k_metrics(bootstrap_df, k=k)
        return pass_metrics[f'pass_at_{k}']
    
    n_samples = len(df)
    bootstrap_stats = []
    
    np.random.seed(42)  # For reproducibility
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_df = df.iloc[bootstrap_indices].reset_index(drop=True)
        
        # Calculate pass@k for this bootstrap sample
        bootstrap_stat = pass_at_k_func(bootstrap_df)
        bootstrap_stats.append(bootstrap_stat)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)
    
    return {f'pass_at_{k}': (lower_bound, upper_bound)}

def bootstrap_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   confidence_level: float = 0.95, n_bootstrap: int = 1000) -> Dict[str, Tuple[float, float]]:
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

def load_results(file_path: str) -> pd.DataFrame:
    """Load results from CSV or JSON file."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        raise ValueError("File must be CSV or JSON format")

def calculate_pass_at_k_metrics(df: pd.DataFrame, k: int = 1) -> Dict[str, float]:
    """
    Calculate pass@k metrics for merge identification task.
    
    Pass@k measures the probability that at least one of the top k predictions is correct.
    For merge identification, this is the fraction of problems where at least one of k attempts
    produces the correct merge decision.
    
    Args:
        df: DataFrame with results (must have multiple attempts per problem)
        k: Number of attempts to consider (default 1 for pass@1)
    
    Returns:
        Dictionary with pass@k metrics
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    if 'unique_id' not in df.columns or 'majority_confidence' not in df.columns:
        # If not majority voting data, calculate pass@1 directly
        df['model_pred_binary'] = (df['model_prediction'] == '1').astype(int)
        df['ground_truth_binary'] = df['is_correct_merge'].astype(int)
        df['correct_prediction'] = (df['model_pred_binary'] == df['ground_truth_binary'])
        
        pass_at_1 = df['correct_prediction'].mean()
        return {
            'pass_at_1': pass_at_1,
            'total_problems': len(df),
            'correct_solutions': df['correct_prediction'].sum()
        }
    
    # For majority voting data, we need to look at individual attempts
    # Group by unique problem and check if any of k attempts are correct
    results = []
    
    for unique_id, group in df.groupby('unique_id'):
        # Parse the individual predictions from the voting data
        all_predictions = eval(group.iloc[0]['all_predictions']) if 'all_predictions' in group.columns else [group.iloc[0]['model_prediction']]
        ground_truth = group.iloc[0]['is_correct_merge']
        
        # Convert to binary
        pred_binary = [(pred == '1') for pred in all_predictions[:k]]
        truth_binary = ground_truth
        
        # Check if any of the k predictions is correct
        any_correct = any(pred == truth_binary for pred in pred_binary)
        results.append(any_correct)
    
    pass_at_k = np.mean(results)
    
    return {
        f'pass_at_{k}': pass_at_k,
        'total_problems': len(results),
        'problems_solved': sum(results)
    }

def calculate_merge_identification_metrics(df: pd.DataFrame, include_ci: bool = True, confidence_level: float = 0.95) -> Dict[str, float]:
    """
    Calculate performance metrics for merge identification task.
    
    For merge identification:
    - Positive class (1): Segment should be merged (is_correct_merge = True)
    - Negative class (-1): Segment should not be merged (is_correct_merge = False)
    
    Args:
        df: DataFrame with results
        include_ci: Whether to include bootstrap confidence intervals
        confidence_level: Confidence level for intervals (default 0.95)
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
        'total_samples': total_samples,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'positive_rate': positive_rate,
        'predicted_positive_rate': predicted_positive_rate
    })
    
    # Add pass@k metrics
    pass_at_k_metrics = calculate_pass_at_k_metrics(df, k=1)
    metrics.update(pass_at_k_metrics)
    
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
                metrics[f'{metric_name}_ci_lower'] = lower
                metrics[f'{metric_name}_ci_upper'] = upper
                metrics[f'{metric_name}_ci_width'] = upper - lower
            
            # Add pass@k confidence intervals
            pass_at_k_ci = bootstrap_pass_at_k_metrics(df, k=1, confidence_level=confidence_level)
            for metric_name, (lower, upper) in pass_at_k_ci.items():
                metrics[f'{metric_name}_ci_lower'] = lower
                metrics[f'{metric_name}_ci_upper'] = upper
                metrics[f'{metric_name}_ci_width'] = upper - lower
            
            metrics['confidence_level'] = confidence_level
            
        except Exception as e:
            # If bootstrap fails, continue without CI
            print(f"Warning: Could not calculate confidence intervals: {e}")
    
    return metrics

def analyze_by_group(df: pd.DataFrame, group_column: str, include_ci: bool = True, confidence_level: float = 0.95) -> Dict[str, Dict[str, float]]:
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

def create_confusion_matrix_plot(df: pd.DataFrame, save_path: str = None):
    """Create and optionally save confusion matrix plot."""
    from sklearn.metrics import confusion_matrix
    
    df['model_pred_binary'] = (df['model_prediction'] == '1').astype(int)
    df['ground_truth_binary'] = df['is_correct_merge'].astype(int)
    
    cm = confusion_matrix(df['ground_truth_binary'], df['model_pred_binary'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Merge', 'Merge'],
                yticklabels=['No Merge', 'Merge'])
    plt.title('Confusion Matrix - Merge Identification')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to prevent hanging
    else:
        plt.show()

def analyze_error_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze patterns in incorrect predictions."""
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
        error_patterns['false_positive_examples'] = false_positives[['operation_id', 'id', 'model_analysis']].head(3).to_dict('records')
    
    if len(false_negatives) > 0:
        error_patterns['false_negative_examples'] = false_negatives[['operation_id', 'id', 'model_analysis']].head(3).to_dict('records')
    
    return error_patterns

def compare_prompt_modes(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compare performance across different prompt modes."""
    if 'prompt_mode' not in df.columns:
        return {}
    
    return analyze_by_group(df, 'prompt_mode')

def compare_models(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compare performance across different models."""
    if 'model' not in df.columns:
        return {}
    
    return analyze_by_group(df, 'model')

def get_filename_suffix(df: pd.DataFrame) -> str:
    """Generate a filename suffix based on model and prompt information."""
    suffix_parts = []
    
    # Add model information
    if 'model' in df.columns:
        unique_models = df['model'].unique()
        if len(unique_models) == 1:
            model_name = str(unique_models[0]).replace('/', '_').replace(':', '_')
            suffix_parts.append(model_name)
        elif len(unique_models) > 1:
            suffix_parts.append(f"{len(unique_models)}models")
    
    # Add prompt mode information
    if 'prompt_mode' in df.columns:
        unique_modes = df['prompt_mode'].unique()
        if len(unique_modes) == 1:
            mode_name = str(unique_modes[0]).replace('/', '_').replace(':', '_')
            suffix_parts.append(mode_name)
        elif len(unique_modes) > 1:
            suffix_parts.append(f"{len(unique_modes)}modes")
    
    return '_' + '_'.join(suffix_parts) if suffix_parts else ''

def generate_performance_report(df: pd.DataFrame, output_dir: str = None, include_ci: bool = True, confidence_level: float = 0.95):
    """Generate comprehensive performance report."""
    report = {}
    
    # Overall metrics
    report['overall_metrics'] = calculate_merge_identification_metrics(df, include_ci=include_ci, confidence_level=confidence_level)
    
    # Group analyses
    if 'prompt_mode' in df.columns:
        report['by_prompt_mode'] = analyze_by_group(df, 'prompt_mode', include_ci=include_ci, confidence_level=confidence_level)
    
    if 'model' in df.columns:
        report['by_model'] = analyze_by_group(df, 'model', include_ci=include_ci, confidence_level=confidence_level)
    
    # Error analysis
    report['error_patterns'] = analyze_error_patterns(df)
    
    # Sample distribution
    report['sample_distribution'] = {
        'total_samples': len(df),
        'positive_samples': (df['is_correct_merge'] == True).sum(),
        'negative_samples': (df['is_correct_merge'] == False).sum(),
        'unique_operations': df['operation_id'].nunique() if 'operation_id' in df.columns else 'N/A'
    }
    
    # Print report
    print("=" * 60)
    print("MERGE IDENTIFICATION PERFORMANCE REPORT")
    print("=" * 60)
    
    # Overall performance
    overall = report['overall_metrics']
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Total Samples: {overall['total_samples']}")
    
    # Helper function to format metrics with confidence intervals
    def format_metric_with_ci(metric_name: str, metrics: Dict) -> str:
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
    
    print(f"  Accuracy: {format_metric_with_ci('accuracy', overall)}")
    print(f"  Precision: {format_metric_with_ci('precision', overall)}")
    print(f"  Recall: {format_metric_with_ci('recall', overall)}")
    print(f"  F1-Score: {format_metric_with_ci('f1_score', overall)}")
    print(f"  Specificity: {format_metric_with_ci('specificity', overall)}")
    
    # Pass@k metrics
    if 'pass_at_1' in overall:
        print(f"  Pass@1: {format_metric_with_ci('pass_at_1', overall)}")
    
    print(f"  True Positives: {overall['true_positives']}")
    print(f"  False Positives: {overall['false_positives']}")
    print(f"  True Negatives: {overall['true_negatives']}")
    print(f"  False Negatives: {overall['false_negatives']}")
    
    # Distribution
    print(f"\nSAMPLE DISTRIBUTION:")
    dist = report['sample_distribution']
    print(f"  Positive samples (should merge): {dist['positive_samples']}")
    print(f"  Negative samples (should not merge): {dist['negative_samples']}")
    print(f"  Unique operations: {dist['unique_operations']}")
    
    # By prompt mode
    if 'by_prompt_mode' in report and report['by_prompt_mode']:
        print(f"\nPERFORMANCE BY PROMPT MODE:")
        for mode, metrics in report['by_prompt_mode'].items():
            print(f"  {mode}:")
            print(f"    Accuracy: {format_metric_with_ci('accuracy', metrics)}")
            print(f"    F1-Score: {format_metric_with_ci('f1_score', metrics)}")
            print(f"    Samples: {metrics['total_samples']}")
    
    # By model
    if 'by_model' in report and report['by_model']:
        print(f"\nPERFORMANCE BY MODEL:")
        for model, metrics in report['by_model'].items():
            print(f"  {model}:")
            print(f"    Accuracy: {format_metric_with_ci('accuracy', metrics)}")
            print(f"    F1-Score: {format_metric_with_ci('f1_score', metrics)}")
            print(f"    Samples: {metrics['total_samples']}")
    
    # Error patterns
    errors = report['error_patterns']
    print(f"\nERROR ANALYSIS:")
    print(f"  False Positives: {errors['false_positive_count']}")
    print(f"  False Negatives: {errors['false_negative_count']}")
    
    if errors.get('false_positive_examples'):
        print(f"  Sample False Positive Operations:")
        for example in errors['false_positive_examples']:
            print(f"    - Op {example['operation_id']}, ID {example['id']}")
    
    if errors.get('false_negative_examples'):
        print(f"  Sample False Negative Operations:")
        for example in errors['false_negative_examples']:
            print(f"    - Op {example['operation_id']}, ID {example['id']}")
    
    # Save report
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename suffix based on model/prompt info
        suffix = get_filename_suffix(df)
        
        # Save JSON report
        report_file = os.path.join(output_dir, f'merge_identification_analysis{suffix}.json')
        with open(report_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Recursively convert numpy types
            def clean_report(data):
                if isinstance(data, dict):
                    return {k: clean_report(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [clean_report(item) for item in data]
                else:
                    return convert_numpy(data)
            
            clean_report_data = clean_report(report)
            json.dump(clean_report_data, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_file}")
        
        # Create confusion matrix plot
        confusion_path = os.path.join(output_dir, f'confusion_matrix{suffix}.png')
        create_confusion_matrix_plot(df, confusion_path)
        print(f"Confusion matrix saved to: {confusion_path}")
    
    return report

def perform_majority_voting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform majority voting for multiple runs of the same prompt.
    Groups by unique identifier and takes majority vote of predictions.
    """
    # Create unique identifier for each prompt instance
    # Use operation_id + segment_id to group multiple runs
    df['unique_id'] = df['operation_id'].astype(str) + '_' + df['id'].astype(str)
    
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
        'mean_confidence': df['majority_confidence'].mean(),
        'median_confidence': df['majority_confidence'].median(),
        'min_confidence': df['majority_confidence'].min(),
        'max_confidence': df['majority_confidence'].max(),
        'unanimous_decisions': (df['majority_confidence'] == 1.0).sum(),
        'split_decisions': (df['majority_confidence'] < 1.0).sum()
    }
    
    analysis['confidence_stats'] = confidence_stats
    
    # Performance by confidence level
    confidence_bins = pd.cut(df['majority_confidence'], bins=[0, 0.6, 0.8, 1.0], 
                           labels=['Low (≤0.6)', 'Medium (0.6-0.8)', 'High (1.0)'])
    
    performance_by_confidence = {}
    for conf_level in confidence_bins.cat.categories:
        conf_df = df[confidence_bins == conf_level]
        if len(conf_df) > 0:
            performance_by_confidence[conf_level] = calculate_merge_identification_metrics(conf_df)
    
    analysis['performance_by_confidence'] = performance_by_confidence
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description="Analyze merge identification task performance")
    parser.add_argument("results_file", help="Path to results CSV or JSON file")
    parser.add_argument("--output-dir", help="Directory to save analysis outputs (defaults to directory of input file)", default=None)
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--majority-vote", action="store_true", 
                       help="Perform majority voting for multiple runs (K repetitions)")
    parser.add_argument("--confidence-intervals", action="store_true", default=True,
                       help="Calculate bootstrap confidence intervals (default: True)")
    parser.add_argument("--no-confidence-intervals", action="store_true",
                       help="Skip confidence interval calculations")
    parser.add_argument("--confidence-level", type=float, default=0.95,
                       help="Confidence level for intervals (default: 0.95)")
    parser.add_argument("--pass-at-k", type=int, nargs="+", default=[1],
                       help="Calculate pass@k for specified k values (default: 1)")
    
    args = parser.parse_args()
    
    # Determine whether to calculate confidence intervals
    include_ci = args.confidence_intervals and not args.no_confidence_intervals
    if args.confidence_level < 0.5 or args.confidence_level >= 1.0:
        print("Error: Confidence level must be between 0.5 and 1.0")
        return
    
    # Set default output directory to the directory containing the CSV file
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.results_file))
    
    # Load results
    print(f"Loading results from: {args.results_file}")
    df = load_results(args.results_file)
    
    # Validate that this is merge identification data
    required_columns = ['is_correct_merge', 'model_prediction']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns for merge identification analysis: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    print(f"Loaded {len(df)} result rows")
    print(f"Output directory: {args.output_dir}")
    
    # Perform majority voting if requested
    if args.majority_vote:
        print("Performing majority voting...")
        df = perform_majority_voting(df)
        
        # Analyze voting patterns
        voting_analysis = analyze_voting_patterns(df)
        if voting_analysis:
            print(f"\nVOTING ANALYSIS:")
            conf_stats = voting_analysis['confidence_stats']
            print(f"  Average Confidence: {conf_stats['mean_confidence']:.3f}")
            print(f"  Unanimous Decisions: {conf_stats['unanimous_decisions']}/{len(df)} ({conf_stats['unanimous_decisions']/len(df):.1%})")
            print(f"  Split Decisions: {conf_stats['split_decisions']}/{len(df)} ({conf_stats['split_decisions']/len(df):.1%})")
            
            if voting_analysis.get('performance_by_confidence'):
                print(f"\n  Performance by Confidence Level:")
                for conf_level, metrics in voting_analysis['performance_by_confidence'].items():
                    print(f"    {conf_level}: Accuracy {metrics['accuracy']:.3f}, F1 {metrics['f1_score']:.3f} (n={metrics['total_samples']})")
    
    # Calculate additional pass@k metrics if requested
    if args.pass_at_k and len(args.pass_at_k) > 1:
        print(f"\nADDITIONAL PASS@K METRICS:")
        for k in args.pass_at_k:
            if k > 1:
                pass_k_metrics = calculate_pass_at_k_metrics(df, k=k)
                pass_k_value = pass_k_metrics[f'pass_at_{k}']
                
                # Calculate confidence intervals if requested
                if include_ci and len(df) > 10:
                    try:
                        pass_k_ci = bootstrap_pass_at_k_metrics(df, k=k, confidence_level=args.confidence_level)
                        lower, upper = pass_k_ci[f'pass_at_{k}']
                        ci_pct = int(args.confidence_level * 100)
                        print(f"  Pass@{k}: {pass_k_value:.3f} ({ci_pct}% CI: {lower:.3f}-{upper:.3f})")
                    except Exception as e:
                        print(f"  Pass@{k}: {pass_k_value:.3f} (CI calculation failed)")
                else:
                    print(f"  Pass@{k}: {pass_k_value:.3f}")
    
    # Generate report
    report = generate_performance_report(df, args.output_dir, include_ci=include_ci, confidence_level=args.confidence_level)
    
    # Add voting analysis to report if available
    if args.majority_vote and 'voting_analysis' in locals():
        report['voting_analysis'] = voting_analysis
    
    # Optional plotting
    if args.plot and args.output_dir:
        try:
            create_confusion_matrix_plot(df, os.path.join(args.output_dir, 'confusion_matrix.png'))
        except ImportError:
            print("Matplotlib/seaborn not available for plotting")

if __name__ == "__main__":
    main()