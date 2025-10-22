#!/usr/bin/env python3
"""
Master script for comprehensive split identification analysis.

Analyzes split identification performance with full metrics, confidence intervals,
and automatic prompt mode inference from filenames.

Features:
- Full metrics with bootstrap confidence intervals
- Automatic grouping by model and prompt mode
- Majority voting support
- Confusion matrix visualization
- Error pattern analysis
- Clean table output (compact or full)
- JSON and plot export
- Automatic prompt_mode inference from filename

Usage:
  # Single file
  python analyze_split_identification_results.py results.csv

  # Multiple files with labels
  python analyze_split_identification_results.py results1.csv results2.csv --labels "Exp1" "Exp2"

  # Full analysis with plots
  python analyze_split_identification_results.py results.csv --output-dir ./reports --plot
"""

import sys
import os
import re

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis_utils import (
    load_results,
    calculate_merge_identification_metrics,
    analyze_error_patterns,
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

    Expected pattern: {model}_{task}_{prompt_mode}_analysis_results_K{K}.csv
    Example: o4-mini_split_identification_informative_analysis_results_K5.csv

    Returns 'informative', 'null', or 'unknown'
    """
    basename = os.path.basename(filename)

    # Pattern: task followed by prompt_mode followed by "analysis_results"
    match = re.search(r'split_identification_(\w+)_analysis_results', basename)
    if match:
        return match.group(1)

    # Fallback: look for known modes
    if 'informative' in basename.lower():
        return 'informative'
    elif 'null' in basename.lower():
        return 'null'

    return 'unknown'


# ============================================================================
# Unified Analysis Pipeline
# ============================================================================

def identify_configurations(dfs: List[pd.DataFrame], labels: List[str]) -> List[Dict]:
    """
    Identify all unique configurations (file x model x prompt_mode).

    Returns list of configuration dictionaries.
    """
    configurations = []

    for df, label in zip(dfs, labels):
        # Check what grouping columns are available
        has_model = 'model' in df.columns and df['model'].nunique() > 1
        has_prompt = 'prompt_mode' in df.columns and df['prompt_mode'].nunique() > 1

        if has_model and has_prompt:
            # Group by both
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
            # Group by model only
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
            # Group by prompt_mode only
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
            # No grouping, use entire dataframe
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
    """
    Analyze all configurations with full metrics and confidence intervals.
    """
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


def detect_id_column(df: pd.DataFrame) -> str:
    """
    Detect which ID column is present in the DataFrame.

    Returns:
        'id' for merge data, 'base_neuron_id' for split data

    Raises:
        ValueError if neither column is found
    """
    if 'id' in df.columns:
        return 'id'
    elif 'base_neuron_id' in df.columns:
        return 'base_neuron_id'
    else:
        raise ValueError(f"No ID column found. Expected 'id' or 'base_neuron_id'. Available: {list(df.columns)}")


def detect_k_repetitions(df: pd.DataFrame, id_column: str) -> int:
    """
    Detect K (number of repetitions per unique sample).

    Args:
        df: DataFrame with predictions
        id_column: Name of ID column ('id' or 'base_neuron_id')
    """
    if 'operation_id' not in df.columns or id_column not in df.columns:
        return 1

    # Count repetitions per unique operation_id + id
    df_copy = df.copy()
    df_copy['unique_id'] = df_copy['operation_id'].astype(str) + '_' + df_copy[id_column].astype(str)
    repetition_counts = df_copy.groupby('unique_id').size()

    # If all samples have the same number of repetitions, return that number
    unique_counts = repetition_counts.unique()
    if len(unique_counts) == 1:
        return int(unique_counts[0])

    # Otherwise return the most common count
    return int(repetition_counts.mode()[0])


def generate_unified_report(dfs: List[pd.DataFrame],
                           labels: List[str],
                           include_ci: bool = True,
                           confidence_level: float = 0.95):
    """
    Generate comprehensive report with full analysis for all configurations.
    """
    report = {}

    # Detect ID column (should be consistent across all files)
    id_column = detect_id_column(dfs[0])
    print(f"Using ID column: {id_column}")

    # Detect K value (number of repetitions)
    k_values = [detect_k_repetitions(df, id_column) for df in dfs]
    max_k = max(k_values)
    report['k_value'] = max_k
    report['k_values_per_file'] = dict(zip(labels, k_values))

    # ========================================================================
    # Individual predictions analysis (always)
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

        # Apply majority voting to each dataframe
        dfs_majority = []
        for df, label, k in zip(dfs, labels, k_values):
            if k > 1:
                print(f"  Applying majority voting to {label} (K={k})...")
                df_voted = perform_majority_voting(df, id_column=id_column)
                dfs_majority.append(df_voted)

                # Show voting stats
                voting_analysis = analyze_voting_patterns(df_voted)
                if voting_analysis and 'confidence_stats' in voting_analysis:
                    conf_stats = voting_analysis['confidence_stats']
                    print(f"    Average confidence: {conf_stats['mean_confidence']:.3f}")
                    print(f"    Unanimous: {conf_stats['unanimous_decisions']}/{len(df_voted)}")
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

    # Error patterns per configuration (individual)
    report['error_patterns_individual'] = {}
    for config in configurations_individual:
        key = f"{config['label']}_{config['model']}_{config['prompt_mode']}"
        report['error_patterns_individual'][key] = analyze_error_patterns(config['df'], id_column=id_column)

    # Error patterns for majority vote (if applicable)
    if max_k > 1:
        report['error_patterns_majority'] = {}
        for config in configurations_majority:
            key = f"{config['label']}_{config['model']}_{config['prompt_mode']}"
            report['error_patterns_majority'][key] = analyze_error_patterns(config['df'], id_column=id_column)

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
    """Print compact table with just accuracy and CIs."""
    print("\n" + "=" * 100)
    print("ACCURACY SUMMARY")
    print("=" * 100)

    # Header
    print(f"{'File':<15} {'Model':<20} {'Prompt Mode':<25} {'N':>6} {'Accuracy':>25}")
    print("-" * 100)

    # Rows
    for result in results:
        m = result['metrics']
        label = result['label'][:14]
        model = result['model'][:19]
        prompt = result['prompt_mode'][:24]
        n = m['total_samples']

        # Format accuracy with CI if available
        if 'accuracy_ci_lower' in m:
            acc_str = f"{m['accuracy']:.3f} [{m['accuracy_ci_lower']:.3f}-{m['accuracy_ci_upper']:.3f}]"
        else:
            acc_str = f"{m['accuracy']:.3f}"

        print(f"{label:<15} {model:<20} {prompt:<25} {n:>6} {acc_str:>25}")


def print_full_table(results: List[Dict]):
    """Print full table with all metrics and CIs."""
    print("\n" + "=" * 165)
    print("COMPREHENSIVE RESULTS")
    print("=" * 165)

    # Header
    print(f"{'File':<12} {'Model':<15} {'Prompt':<20} {'N':>5} "
          f"{'Accuracy':>25} {'Precision':>25} {'Recall':>25} {'F1-Score':>25}")
    print("-" * 165)

    # Rows
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
    """Print table showing confusion matrix values."""
    print("\n" + "=" * 120)
    print("CONFUSION MATRIX VALUES")
    print("=" * 120)

    # Header
    print(f"{'File':<15} {'Model':<20} {'Prompt Mode':<25} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6} {'FPR':>8}")
    print("-" * 120)

    # Rows
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


def print_error_summary(error_patterns: Dict):
    """Print error pattern summary."""
    if not error_patterns:
        return

    print("\n" + "=" * 80)
    print("ERROR SUMMARY")
    print("=" * 80)

    for config_key, errors in error_patterns.items():
        fp_count = errors['false_positive_count']
        fn_count = errors['false_negative_count']
        total_errors = fp_count + fn_count

        if total_errors > 0:
            print(f"\n{config_key}:")
            print(f"  False Positives (said split, shouldn't): {fp_count}")
            print(f"  False Negatives (said no split, should):  {fn_count}")
            print(f"  Total Errors: {total_errors}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze split identification task performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  %(prog)s results.csv

  # Multiple files with labels
  %(prog)s results1.csv results2.csv --labels "GPT-4" "Claude"

  # Using a config file (label,path per line)
  %(prog)s --config experiments.txt

  # Full analysis with config file
  %(prog)s --config experiments.txt --table full --confusion-matrix --show-errors --output-dir ./reports

Config file format (experiments.txt):
  Informative,output/o4-mini_split_identification_informative_analysis_results_K5.csv
  Null,output/o4-mini_split_identification_null_analysis_results_K5.csv

Note:
  - If K repetitions are detected (K>1), both individual predictions AND
    majority vote (cons@K) metrics are automatically calculated and shown
  - K is detected automatically from operation_id + base_neuron_id groupings
  - prompt_mode is inferred from filename if not present in CSV
        """
    )

    parser.add_argument("results_files", nargs='*',
                       help="Path(s) to results CSV or JSON file(s)")
    parser.add_argument("--config",
                       help="Config file with label,path per line (alternative to providing files directly)")
    parser.add_argument("--labels", nargs='+',
                       help="Labels for each file (defaults to File_1, File_2, ...)")
    parser.add_argument("--output-dir",
                       help="Directory to save analysis outputs")
    parser.add_argument("--table", choices=['compact', 'full', 'both'], default='both',
                       help="Table style: compact (accuracy only), full (all metrics), or both (default: both)")
    parser.add_argument("--confusion-matrix", action="store_true",
                       help="Show confusion matrix table")
    parser.add_argument("--show-errors", action="store_true",
                       help="Show error summary")
    parser.add_argument("--no-confidence-intervals", action="store_true",
                       help="Skip confidence interval calculations (faster)")
    parser.add_argument("--confidence-level", type=float, default=0.95,
                       help="Confidence level for intervals (default: 0.95)")

    args = parser.parse_args()

    # Configuration
    include_ci = not args.no_confidence_intervals
    if args.confidence_level < 0.5 or args.confidence_level >= 1.0:
        print("Error: Confidence level must be between 0.5 and 1.0")
        return

    # Parse input: either config file or direct file arguments
    file_paths = []
    labels = []

    if args.config:
        # Load from config file
        if args.results_files:
            print("Error: Cannot specify both --config and direct file arguments")
            return

        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            return

        print(f"Loading configuration from: {args.config}")
        with open(args.config, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                # Parse label,path
                parts = line.split(',', 1)
                if len(parts) != 2:
                    print(f"Error: Line {line_num} in config file must be 'label,path' format: {line}")
                    return

                label, path = parts[0].strip(), parts[1].strip()

                if not label:
                    print(f"Error: Empty label on line {line_num}")
                    return

                if not os.path.exists(path):
                    print(f"Error: File not found (line {line_num}): {path}")
                    return

                labels.append(label)
                file_paths.append(path)

        if not file_paths:
            print("Error: No valid entries found in config file")
            return

        print(f"  Loaded {len(file_paths)} file(s) from config")

    else:
        # Use direct file arguments
        if not args.results_files:
            print("Error: Must provide either --config or result files")
            parser.print_help()
            return

        file_paths = args.results_files

        # Validate files exist
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}")
                return

        # Generate labels if not provided
        if args.labels is None:
            if len(file_paths) == 1:
                labels = [os.path.basename(file_paths[0]).replace('.csv', '').replace('.json', '')]
            else:
                labels = [f"File_{i+1}" for i in range(len(file_paths))]
        else:
            labels = args.labels

        if len(labels) != len(file_paths):
            print(f"Error: Number of labels ({len(labels)}) must match number of files ({len(file_paths)})")
            return

    # ========================================================================
    # Load Data
    # ========================================================================
    print("=" * 60)
    print("SPLIT IDENTIFICATION ANALYSIS")
    print("=" * 60)

    dfs = []
    for file_path, label in zip(file_paths, labels):
        print(f"\nLoading {label}: {file_path}")
        df = load_results(file_path)

        # Validate split identification data
        required_columns = ['is_split', 'model_prediction']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return

        # Infer prompt_mode if not present
        if 'prompt_mode' not in df.columns:
            inferred_mode = infer_prompt_mode_from_filename(file_path)
            print(f"  Inferred prompt_mode: {inferred_mode}")
            df['prompt_mode'] = inferred_mode

        # Rename column for compatibility with analysis utils
        if 'is_split' in df.columns and 'is_correct_merge' not in df.columns:
            df['is_correct_merge'] = df['is_split']

        print(f"  Loaded {len(df)} samples")
        dfs.append(df)

    if not dfs:
        print("Error: No valid data found in any files")
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

    # Print K value information
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

    # Majority vote results (if K > 1)
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

    # Error summary
    if args.show_errors:
        print("\n" + "=" * 60)
        print("ERRORS - INDIVIDUAL PREDICTIONS")
        print("=" * 60)
        print_error_summary(report['error_patterns_individual'])

        if k_value > 1:
            print("\n" + "=" * 60)
            print(f"ERRORS - MAJORITY VOTE (cons@{k_value})")
            print("=" * 60)
            print_error_summary(report['error_patterns_majority'])

    # ========================================================================
    # Save Outputs
    # ========================================================================
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        # Save JSON report
        report_file = os.path.join(args.output_dir, 'split_identification_analysis_report.json')
        with open(report_file, 'w') as f:
            json.dump(clean_report_for_json(report), f, indent=2)
        print(f"\nReport saved to: {report_file}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
