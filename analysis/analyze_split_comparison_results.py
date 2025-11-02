#!/usr/bin/env python3
"""
Master script for comprehensive split comparison analysis.

Analyzes split comparison performance (choosing which neuron needs splitting)
with full metrics, confidence intervals, and automatic prompt mode inference.

Features:
- Accuracy metrics with bootstrap confidence intervals
- Automatic grouping by model and prompt mode
- Majority voting support
- Confusion matrix visualization
- Clean table output
- JSON export
- Automatic prompt_mode inference from filename

Usage:
  # Single file
  python analyze_split_comparison_results.py results.csv

  # Multiple files with labels
  python analyze_split_comparison_results.py results1.csv results2.csv --labels "Exp1" "Exp2"

  # Full analysis with output
  python analyze_split_comparison_results.py results.csv --output-dir ./reports
"""

import sys
import os
import re

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis_utils import (
    load_results,
    calculate_merge_identification_metrics,
    perform_majority_voting,
    analyze_voting_patterns,
    format_metric_with_ci,
    clean_report_for_json
)

import pandas as pd
import json
import argparse
from typing import Dict, List, Any


# ============================================================================
# Filename Parsing
# ============================================================================

def infer_prompt_mode_from_filename(filename: str) -> str:
    """
    Infer prompt_mode from filename pattern.
    Expected: {model}_{task}_{prompt_mode}_analysis_results_K{K}.csv
    """
    basename = os.path.basename(filename)
    match = re.search(r'split_comparison_(\w+)_analysis_results', basename)
    if match:
        return match.group(1)

    if 'informative' in basename.lower():
        return 'informative'
    elif 'null' in basename.lower():
        return 'null'

    return 'unknown'


# ============================================================================
# Data Processing
# ============================================================================

def detect_k_repetitions(df: pd.DataFrame) -> int:
    """
    Detect K (number of repetitions per unique sample).
    For comparison tasks, extract problem_id from operation_id pattern.

    For split_comparison, operation_id format is: split_comparison_{problem}_{index}
    Since problems are shown in multiple orientations with different correct_answers,
    we group by (problem_id, correct_answer) to get K per unique presentation.
    """
    if 'operation_id' not in df.columns:
        return 1

    df_copy = df.copy()

    # For split_comparison, extract problem_id from operation_id
    if df_copy['operation_id'].str.contains('split_comparison_').any():
        df_copy['problem_id'] = df_copy['operation_id'].str.extract(r'split_comparison_(\d+)_\d+')[0]
        # Group by (problem_id, correct_answer) since different orientations have different answers
        if 'correct_answer' in df_copy.columns:
            repetition_counts = df_copy.groupby(['problem_id', 'correct_answer']).size()
        else:
            repetition_counts = df_copy.groupby('problem_id').size()
    else:
        repetition_counts = df_copy.groupby('operation_id').size()

    unique_counts = repetition_counts.unique()
    if len(unique_counts) == 1:
        return int(unique_counts[0])

    return int(repetition_counts.mode()[0])


def prepare_comparison_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare split comparison data for analysis by converting to binary format.

    For comparison tasks, we treat it as a binary classification:
    - Ground truth: always True (we expect the model to get it right)
    - Prediction: '1' if model chose correct option, '-1' if incorrect

    This allows us to use standard metrics (accuracy, precision, etc.)
    """
    df = df.copy()

    # Convert to strings for comparison
    df['correct_answer_str'] = df['correct_answer'].astype(str)
    df['model_prediction_orig'] = df['model_prediction'].astype(str)

    # Model is correct if prediction matches correct answer
    df['is_correct'] = df['model_prediction_orig'] == df['correct_answer_str']

    # For metrics: treat as binary classification where ground truth is always "should be correct"
    # Ground truth: always True (positive class = model should choose correctly)
    df['is_correct_merge'] = True

    # Prediction: '1' if model chose correctly, '-1' if not
    df['model_prediction'] = df['is_correct'].apply(lambda x: '1' if x else '-1')

    return df


def perform_comparison_majority_voting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform majority voting for comparison task.
    Groups by problem_id (extracted from operation_id) and correct_answer, then takes majority vote.

    For split_comparison tasks, operation_id format is: split_comparison_{problem}_{index}
    We group by (problem_id, correct_answer) since the same problem shown in different orders
    has different correct answers.
    """
    if 'operation_id' not in df.columns:
        raise ValueError("Column 'operation_id' not found in DataFrame")

    df_copy = df.copy()

    # For split_comparison, extract problem_id from operation_id
    if df_copy['operation_id'].str.contains('split_comparison_').any():
        df_copy['problem_id'] = df_copy['operation_id'].str.extract(r'split_comparison_(\d+)_\d+')[0]
        grouping_key = ['problem_id', 'correct_answer']
    else:
        grouping_key = ['operation_id']

    # Group and aggregate
    majority_results = []

    for group_vals, group in df_copy.groupby(grouping_key):
        # Take majority vote of model_prediction_orig (the actual predictions)
        predictions = group['model_prediction_orig'].tolist()

        # Count votes for each prediction
        from collections import Counter
        vote_counts = Counter(predictions)

        # Get majority prediction
        majority_prediction = vote_counts.most_common(1)[0][0]
        majority_count = vote_counts[majority_prediction]

        # Calculate confidence (percentage agreement)
        total_votes = len(predictions)
        confidence = majority_count / total_votes

        # Compare majority prediction to correct answer
        correct_answer = str(group.iloc[0]['correct_answer'])
        is_correct = (majority_prediction == correct_answer)

        # Use first row as template
        result_row = group.iloc[0].copy()
        result_row['model_prediction_orig'] = majority_prediction
        result_row['is_correct'] = is_correct
        # Keep is_correct_merge as True (ground truth: model should get it right)
        result_row['is_correct_merge'] = True
        # model_prediction: '1' if majority vote was correct, '-1' if wrong
        result_row['model_prediction'] = '1' if is_correct else '-1'
        result_row['majority_vote_confidence'] = confidence
        result_row['vote_count'] = total_votes
        result_row['unanimous'] = (confidence == 1.0)

        majority_results.append(result_row)

    result_df = pd.DataFrame(majority_results)

    print(f"Majority voting: Reduced {len(df)} individual predictions to {len(result_df)} majority votes")
    if 'majority_vote_confidence' in result_df.columns:
        avg_conf = result_df['majority_vote_confidence'].mean()
        print(f"Average confidence: {avg_conf:.3f}")

    return result_df


# ============================================================================
# Configuration Analysis
# ============================================================================

def identify_configurations(dfs: List[pd.DataFrame], labels: List[str]) -> List[Dict]:
    """Identify all unique configurations (file x model x prompt_mode)."""
    configurations = []

    for df, label in zip(dfs, labels):
        has_model = 'model' in df.columns and df['model'].nunique() > 1
        has_prompt = 'prompt_mode' in df.columns and df['prompt_mode'].nunique() > 1

        if has_model and has_prompt:
            for model in df['model'].unique():
                for prompt_mode in df['prompt_mode'].unique():
                    filtered_df = df[(df['model'] == model) & (df['prompt_mode'] == prompt_mode)]
                    if len(filtered_df) > 0:
                        configurations.append({
                            'label': label,
                            'model': str(model),
                            'prompt_mode': str(prompt_mode),
                            'df': filtered_df
                        })
        elif has_model:
            for model in df['model'].unique():
                filtered_df = df[df['model'] == model]
                if len(filtered_df) > 0:
                    configurations.append({
                        'label': label,
                        'model': str(model),
                        'prompt_mode': 'all',
                        'df': filtered_df
                    })
        elif has_prompt:
            for prompt_mode in df['prompt_mode'].unique():
                filtered_df = df[df['prompt_mode'] == prompt_mode]
                if len(filtered_df) > 0:
                    configurations.append({
                        'label': label,
                        'model': 'all',
                        'prompt_mode': str(prompt_mode),
                        'df': filtered_df
                    })
        else:
            configurations.append({
                'label': label,
                'model': 'all',
                'prompt_mode': 'all',
                'df': df
            })

    return configurations


def analyze_all_configurations(configurations: List[Dict],
                               include_ci: bool = True,
                               confidence_level: float = 0.95) -> List[Dict]:
    """Analyze all configurations with metrics and confidence intervals."""
    results = []

    print(f"\nAnalyzing {len(configurations)} configuration(s)...")

    for i, config in enumerate(configurations, 1):
        print(f"  [{i}/{len(configurations)}] {config['label']} | {config['model']} | {config['prompt_mode']} (n={len(config['df'])})")

        metrics = calculate_merge_identification_metrics(
            config['df'],
            include_ci=include_ci,
            confidence_level=confidence_level
        )

        result = {
            'label': config['label'],
            'model': config['model'],
            'prompt_mode': config['prompt_mode'],
            'metrics': metrics
        }

        results.append(result)

    return results


def generate_unified_report(dfs: List[pd.DataFrame],
                           labels: List[str],
                           include_ci: bool = True,
                           confidence_level: float = 0.95):
    """Generate comprehensive report with full analysis."""
    report = {}

    # Detect K value
    k_values = [detect_k_repetitions(df) for df in dfs]
    max_k = max(k_values)
    report['k_value'] = max_k
    report['k_values_per_file'] = dict(zip(labels, k_values))

    # ========================================================================
    # Individual predictions analysis
    # ========================================================================
    print("\n" + "=" * 60)
    print("ANALYZING INDIVIDUAL PREDICTIONS")
    print("=" * 60)

    configurations_individual = identify_configurations(dfs, labels)
    results_individual = analyze_all_configurations(
        configurations_individual,
        include_ci=include_ci,
        confidence_level=confidence_level
    )
    report['configurations_individual'] = results_individual

    # ========================================================================
    # Majority vote analysis (if K > 1)
    # ========================================================================
    if max_k > 1:
        print("\n" + "=" * 60)
        print(f"ANALYZING MAJORITY VOTE (K={max_k})")
        print("=" * 60)

        dfs_majority = []
        for df, label, k in zip(dfs, labels, k_values):
            if k > 1:
                print(f"  Applying majority voting to {label} (K={k})...")
                df_voted = perform_comparison_majority_voting(df)
                dfs_majority.append(df_voted)

                if 'majority_vote_confidence' in df_voted.columns:
                    avg_conf = df_voted['majority_vote_confidence'].mean()
                    unanimous = df_voted['unanimous'].sum()
                    print(f"    Average confidence: {avg_conf:.3f}")
                    print(f"    Unanimous: {unanimous}/{len(df_voted)}")
            else:
                print(f"  Skipping {label} (K=1, no repetitions)")
                dfs_majority.append(df)

        configurations_majority = identify_configurations(dfs_majority, labels)
        results_majority = analyze_all_configurations(
            configurations_majority,
            include_ci=include_ci,
            confidence_level=confidence_level
        )
        report['configurations_majority'] = results_majority
    else:
        report['configurations_majority'] = None

    # Overall statistics
    report['overall_stats'] = {
        'total_individual_predictions': sum(len(config['df']) for config in configurations_individual),
        'total_unique_samples': sum(len(config['df']) for config in configurations_majority) if max_k > 1 else None,
        'n_configurations': len(configurations_individual),
        'n_files': len(labels),
        'k_value': max_k
    }

    return report


# ============================================================================
# Table Output
# ============================================================================

def print_compact_table(results: List[Dict]):
    """Print compact table with accuracy."""
    print("\n" + "=" * 100)
    print("ACCURACY SUMMARY")
    print("=" * 100)

    print(f"{'File':<15} {'Model':<20} {'Prompt Mode':<25} {'N':>6} {'Accuracy':>25}")
    print("-" * 100)

    for result in results:
        m = result['metrics']
        label = result['label'][:14]
        model = result['model'][:19]
        prompt = result['prompt_mode'][:24]
        n = m['total_samples']

        if 'accuracy_ci_lower' in m:
            acc_str = f"{m['accuracy']:.3f} [{m['accuracy_ci_lower']:.3f}-{m['accuracy_ci_upper']:.3f}]"
        else:
            acc_str = f"{m['accuracy']:.3f}"

        print(f"{label:<15} {model:<20} {prompt:<25} {n:>6} {acc_str:>25}")


def print_full_table(results: List[Dict]):
    """Print full table with all metrics."""
    print("\n" + "=" * 165)
    print("COMPREHENSIVE RESULTS")
    print("=" * 165)

    print(f"{'File':<12} {'Model':<15} {'Prompt':<20} {'N':>5} "
          f"{'Accuracy':>25} {'Precision':>25} {'Recall':>25} {'F1-Score':>25}")
    print("-" * 165)

    for result in results:
        m = result['metrics']
        label = result['label'][:11]
        model = result['model'][:14]
        prompt = result['prompt_mode'][:19]
        n = m['total_samples']

        def fmt(metric_name):
            if f'{metric_name}_ci_lower' in m:
                return f"{m[metric_name]:.3f} [{m[f'{metric_name}_ci_lower']:.3f}-{m[f'{metric_name}_ci_upper']:.3f}]"
            else:
                return f"{m[metric_name]:.3f}"

        print(f"{label:<12} {model:<15} {prompt:<20} {n:>5} "
              f"{fmt('accuracy'):>25} {fmt('precision'):>25} {fmt('recall'):>25} {fmt('f1_score'):>25}")


def print_confusion_matrix_table(results: List[Dict]):
    """Print confusion matrix values."""
    print("\n" + "=" * 120)
    print("CONFUSION MATRIX VALUES")
    print("=" * 120)

    print(f"{'File':<15} {'Model':<20} {'Prompt Mode':<25} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6} {'FPR':>8}")
    print("-" * 120)

    for result in results:
        m = result['metrics']
        label = result['label'][:14]
        model = result['model'][:19]
        prompt = result['prompt_mode'][:24]

        tp = m['true_positives']
        fp = m['false_positives']
        tn = m['true_negatives']
        fn = m['false_negatives']
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"{label:<15} {model:<20} {prompt:<25} {tp:>6} {fp:>6} {tn:>6} {fn:>6} {fp_rate:>8.3f}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze split comparison task performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  %(prog)s results.csv

  # Multiple files with labels
  %(prog)s results1.csv results2.csv --labels "Informative" "Null"

  # Using a config file
  %(prog)s --config experiments.txt --output-dir ./reports

Config file format:
  Informative,output/o4-mini_split_comparison_informative_K10.csv
  Null,output/o4-mini_split_comparison_null_K10.csv

Note:
  - Analyzes which neuron the model correctly identified as needing splitting
  - K repetitions detected automatically from operation_id grouping
  - prompt_mode inferred from filename if not in CSV
        """
    )

    parser.add_argument("results_files", nargs='*',
                       help="Path(s) to results CSV or JSON file(s)")
    parser.add_argument("--config",
                       help="Config file with label,path per line")
    parser.add_argument("--labels", nargs='+',
                       help="Labels for each file")
    parser.add_argument("--output-dir",
                       help="Directory to save analysis outputs")
    parser.add_argument("--table", choices=['compact', 'full', 'both'], default='both',
                       help="Table style (default: both)")
    parser.add_argument("--confusion-matrix", action="store_true",
                       help="Show confusion matrix table")
    parser.add_argument("--no-confidence-intervals", action="store_true",
                       help="Skip confidence interval calculations")
    parser.add_argument("--confidence-level", type=float, default=0.95,
                       help="Confidence level for intervals (default: 0.95)")

    args = parser.parse_args()

    include_ci = not args.no_confidence_intervals
    if args.confidence_level < 0.5 or args.confidence_level >= 1.0:
        print("Error: Confidence level must be between 0.5 and 1.0")
        return

    # Parse input
    file_paths = []
    labels = []

    if args.config:
        if args.results_files:
            print("Error: Cannot specify both --config and direct file arguments")
            return

        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            return

        with open(args.config, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split(',', 1)
                if len(parts) != 2:
                    print(f"Error: Line {line_num} must be 'label,path' format")
                    return

                label, path = parts[0].strip(), parts[1].strip()
                if not os.path.exists(path):
                    print(f"Error: File not found (line {line_num}): {path}")
                    return

                labels.append(label)
                file_paths.append(path)

    else:
        if not args.results_files:
            print("Error: Must provide either --config or result files")
            parser.print_help()
            return

        file_paths = args.results_files

        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}")
                return

        if args.labels is None:
            labels = [os.path.basename(f).replace('.csv', '').replace('.json', '') for f in file_paths]
        else:
            labels = args.labels

        if len(labels) != len(file_paths):
            print(f"Error: Number of labels ({len(labels)}) must match files ({len(file_paths)})")
            return

    # ========================================================================
    # Load Data
    # ========================================================================
    print("=" * 60)
    print("SPLIT COMPARISON ANALYSIS")
    print("=" * 60)

    dfs = []
    for file_path, label in zip(file_paths, labels):
        print(f"\nLoading {label}: {file_path}")
        df = load_results(file_path)

        # Validate columns
        required_columns = ['correct_answer', 'model_prediction']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return

        # Infer prompt_mode if not present
        if 'prompt_mode' not in df.columns:
            inferred_mode = infer_prompt_mode_from_filename(file_path)
            print(f"  Inferred prompt_mode: {inferred_mode}")
            df['prompt_mode'] = inferred_mode

        # Prepare data for analysis
        df = prepare_comparison_data(df)

        print(f"  Loaded {len(df)} samples")
        dfs.append(df)

    if not dfs:
        print("Error: No valid data found")
        return

    # ========================================================================
    # Analysis
    # ========================================================================
    report = generate_unified_report(
        dfs,
        labels,
        include_ci=include_ci,
        confidence_level=args.confidence_level
    )

    # ========================================================================
    # Output
    # ========================================================================
    k_value = report['k_value']

    if k_value > 1:
        print("\n" + "=" * 60)
        print(f"DETECTED K={k_value} REPETITIONS PER SAMPLE")
        print("=" * 60)
        print("Showing both individual predictions and majority vote (cons@K) results")

    # Individual predictions results
    results_individual = report['configurations_individual']

    print("\n" + "=" * 60)
    print("INDIVIDUAL PREDICTIONS RESULTS")
    print("=" * 60)

    if args.table in ['compact', 'both']:
        print_compact_table(results_individual)

    if args.table in ['full', 'both']:
        print_full_table(results_individual)

    if args.confusion_matrix:
        print_confusion_matrix_table(results_individual)

    # Majority vote results
    if k_value > 1 and report['configurations_majority'] is not None:
        results_majority = report['configurations_majority']

        print("\n" + "=" * 60)
        print(f"MAJORITY VOTE (cons@{k_value}) RESULTS")
        print("=" * 60)

        if args.table in ['compact', 'both']:
            print_compact_table(results_majority)

        if args.table in ['full', 'both']:
            print_full_table(results_majority)

        if args.confusion_matrix:
            print_confusion_matrix_table(results_majority)

    # ========================================================================
    # Save Outputs
    # ========================================================================
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        report_file = os.path.join(args.output_dir, 'split_comparison_analysis_report.json')
        with open(report_file, 'w') as f:
            json.dump(clean_report_for_json(report), f, indent=2)
        print(f"\nReport saved to: {report_file}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
