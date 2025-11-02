#!/usr/bin/env python3
"""
Master script for comprehensive merge comparison analysis.

Analyzes merge comparison performance with full metrics, confidence intervals,
and heuristic analysis. Works with single or multiple files, always providing
the same depth of analysis.

Features:
- Accuracy metrics with bootstrap confidence intervals
- Automatic grouping by model and prompt mode
- Heuristic combination analysis
- Majority voting support (K repetitions)
- Split vs non-split analysis
- Error pattern analysis
- Clean table output (compact or full)
- JSON and plot export

Usage:
  # Single file
  python analyze_merge_comparison_results.py results.csv

  # Multiple files with labels
  python analyze_merge_comparison_results.py results1.csv results2.csv --labels "Exp1" "Exp2"

  # Using config file
  python analyze_merge_comparison_results.py --config experiments.txt --output-dir ./reports
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis_utils import (
    load_results,
    calculate_merge_comparison_metrics,
    analyze_heuristic_combinations,
    format_metric_with_ci,
    clean_report_for_json
)

import pandas as pd
import json
import argparse
from typing import Dict, List, Any


# ============================================================================
# Unified Analysis Pipeline
# ============================================================================

def identify_configurations(dfs: List[pd.DataFrame], labels: List[str]) -> List[Dict]:
    """
    Identify all unique configurations (file x model x prompt_mode).

    Returns list of configuration dictionaries with:
    - label: file label
    - model: model name (or 'all' if mixed/absent)
    - prompt_mode: prompt mode (or 'all' if mixed/absent)
    - df: filtered dataframe for this configuration
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

    Returns list of results with metrics for each configuration.
    """
    results = []

    print(f"\nAnalyzing {len(configurations)} configuration(s)...")

    for i, config in enumerate(configurations, 1):
        print(f"  [{i}/{len(configurations)}] {config['label']} | {config['model']} | {config['prompt_mode']} (n={len(config['df'])})")

        metrics = calculate_merge_comparison_metrics(
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


def detect_k_repetitions(df: pd.DataFrame) -> int:
    """
    Detect K (number of repetitions per unique sample).
    Returns K if repetitions are detected, otherwise returns 1.
    """
    if 'operation_id' not in df.columns:
        return 1

    # Count repetitions per unique operation_id
    repetition_counts = df.groupby('operation_id').size()

    # If all samples have the same number of repetitions, return that number
    unique_counts = repetition_counts.unique()
    if len(unique_counts) == 1:
        return int(unique_counts[0])

    # Otherwise return the most common count
    return int(repetition_counts.mode()[0])


def perform_majority_voting_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform majority voting for merge comparison task.
    Groups by operation_id and takes the most common model_chosen_id.
    """
    import ast

    def vote_for_group(group):
        # Get the most common model_chosen_id (excluding 'none')
        chosen_ids = group['model_chosen_id'].astype(str)
        chosen_ids = chosen_ids[~chosen_ids.isin(['none', 'nan'])]

        if len(chosen_ids) == 0:
            # All were 'none', keep as 'none'
            voted_id = 'none'
            confidence = 0.0
        else:
            # Get most common
            value_counts = chosen_ids.value_counts()
            voted_id = value_counts.index[0]
            confidence = value_counts.iloc[0] / len(group)

        # Take first row as template
        result = group.iloc[0].copy()
        result['model_chosen_id'] = voted_id
        result['vote_confidence'] = confidence
        result['vote_count'] = len(group)

        return result

    # Group by operation_id and vote
    # Use list comprehension to avoid pandas groupby.apply deprecation warning
    results = []
    for operation_id, group in df.groupby('operation_id'):
        results.append(vote_for_group(group))

    voted_df = pd.DataFrame(results)
    voted_df = voted_df.reset_index(drop=True)

    return voted_df


def generate_unified_report(dfs: List[pd.DataFrame],
                           labels: List[str],
                           include_ci: bool = True,
                           confidence_level: float = 0.95):
    """
    Generate comprehensive report with full analysis for all configurations.

    Returns dictionary with:
    - configurations_individual: individual prediction metrics
    - configurations_majority: majority vote metrics (if K > 1)
    - k_value: number of repetitions detected
    - heuristic_analysis: heuristic analysis if applicable
    - overall_stats: summary statistics
    """
    report = {}

    # Detect K value (number of repetitions)
    k_values = [detect_k_repetitions(df) for df in dfs]
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
                df_voted = perform_majority_voting_comparison(df)
                dfs_majority.append(df_voted)

                # Show voting stats
                if 'vote_confidence' in df_voted.columns:
                    mean_conf = df_voted['vote_confidence'].mean()
                    unanimous = (df_voted['vote_confidence'] == 1.0).sum()
                    print(f"    Average confidence: {mean_conf:.3f}")
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

    # ========================================================================
    # Combined analyses
    # ========================================================================
    # Combined heuristic analysis if applicable (use individual predictions)
    combined_df = pd.concat([config['df'] for config in configurations_individual], ignore_index=True)
    if 'prompt_mode' in combined_df.columns:
        # For merge comparison, we need to adapt heuristic analysis
        # We'll create a temporary 'is_correct_merge' column for compatibility
        report['heuristic_analysis'] = analyze_heuristic_combinations_comparison(
            combined_df,
            include_ci=include_ci,
            confidence_level=confidence_level
        )

    # Overall statistics
    report['overall_stats'] = {
        'total_individual_predictions': sum(len(config['df']) for config in configurations_individual),
        'total_unique_samples': sum(len(config['df']) for config in configurations_majority) if max_k > 1 else None,
        'n_configurations': len(configurations_individual),
        'n_files': len(labels),
        'k_value': max_k
    }

    return report


def analyze_heuristic_combinations_comparison(df: pd.DataFrame,
                                              include_ci: bool = True,
                                              confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Analyze performance by heuristic combinations for merge comparison task.
    Adapted from analyze_heuristic_combinations for merge comparison.
    """
    from analysis_utils import extract_heuristics_from_prompt_mode

    results = {}

    # Extract heuristics from prompt_mode
    df = df.copy()
    df['heuristics'] = df['prompt_mode'].apply(extract_heuristics_from_prompt_mode)
    df['heuristic_count'] = df['heuristics'].apply(len)
    df['heuristic_combo'] = df['heuristics'].apply(lambda x: '+'.join(sorted(x)) if x else 'none')

    # Analyze by heuristic count
    by_count = {}
    for count in sorted(df['heuristic_count'].unique()):
        subset = df[df['heuristic_count'] == count]
        if len(subset) > 0:
            metrics = calculate_merge_comparison_metrics(
                subset,
                include_ci=include_ci,
                confidence_level=confidence_level
            )
            by_count[f"{count}_heuristics"] = metrics

    results['by_heuristic_count'] = by_count

    # Analyze by specific combination (only if reasonable number of combos)
    unique_combos = df['heuristic_combo'].nunique()
    if unique_combos <= 20:
        by_combo = {}
        for combo in sorted(df['heuristic_combo'].unique()):
            subset = df[df['heuristic_combo'] == combo]
            if len(subset) > 10:  # Only analyze if enough samples
                metrics = calculate_merge_comparison_metrics(
                    subset,
                    include_ci=include_ci,
                    confidence_level=confidence_level
                )
                by_combo[combo] = metrics

        results['by_combination'] = by_combo

    return results


# ============================================================================
# Table Output
# ============================================================================

def print_compact_table(results: List[Dict]):
    """Print compact table with just accuracy and CIs."""
    print("\n" + "=" * 120)
    print("ACCURACY SUMMARY")
    print("=" * 120)

    # Header
    print(f"{'File':<15} {'Model':<20} {'Prompt Mode':<30} {'N':>6} {'Accuracy':>25} {'None%':>8}")
    print("-" * 120)

    # Rows
    for result in results:
        m = result['metrics']
        label = result['label'][:14]
        model = result['model'][:19]
        prompt = result['prompt_mode'][:29]
        n = m['total_samples']

        # Format accuracy with CI if available
        if 'accuracy_ci_lower' in m:
            acc_str = f"{m['accuracy']:.3f} [{m['accuracy_ci_lower']:.3f}-{m['accuracy_ci_upper']:.3f}]"
        else:
            acc_str = f"{m['accuracy']:.3f}"

        none_pct = m.get('none_rate', 0) * 100

        print(f"{label:<15} {model:<20} {prompt:<30} {n:>6} {acc_str:>25} {none_pct:>7.1f}%")


def print_full_table(results: List[Dict]):
    """Print full table with all metrics."""
    print("\n" + "=" * 145)
    print("COMPREHENSIVE RESULTS")
    print("=" * 145)

    # Header
    print(f"{'File':<12} {'Model':<15} {'Prompt':<25} {'N':>5} "
          f"{'Accuracy':>25} {'Correct':>8} {'Wrong':>8} {'None':>8} {'None%':>8}")
    print("-" * 145)

    # Rows
    for result in results:
        m = result['metrics']
        label = result['label'][:11]
        model = result['model'][:14]
        prompt = result['prompt_mode'][:24]
        n = m['total_samples']

        # Format accuracy with CI if available
        if 'accuracy_ci_lower' in m:
            acc_str = f"{m['accuracy']:.3f} [{m['accuracy_ci_lower']:.3f}-{m['accuracy_ci_upper']:.3f}]"
        else:
            acc_str = f"{m['accuracy']:.3f}"

        correct = m['correct']
        incorrect = m['incorrect']
        none_count = m.get('none_responses', 0)
        none_pct = m.get('none_rate', 0) * 100

        print(f"{label:<12} {model:<15} {prompt:<25} {n:>5} "
              f"{acc_str:>25} {correct:>8} {incorrect:>8} {none_count:>8} {none_pct:>7.1f}%")


def print_split_analysis_table(results: List[Dict]):
    """Print table showing split vs non-split performance."""
    print("\n" + "=" * 120)
    print("SPLIT vs NON-SPLIT ANALYSIS")
    print("=" * 120)

    # Check if any results have split data
    has_split_data = any('split_accuracy' in r['metrics'] for r in results)
    if not has_split_data:
        print("  No split/non-split data available")
        return

    # Header
    print(f"{'File':<15} {'Model':<20} {'Prompt Mode':<25} "
          f"{'Split Acc':>12} {'Non-Split Acc':>15}")
    print("-" * 120)

    # Rows
    for result in results:
        m = result['metrics']
        if 'split_accuracy' not in m and 'non_split_accuracy' not in m:
            continue

        label = result['label'][:14]
        model = result['model'][:19]
        prompt = result['prompt_mode'][:24]

        split_acc = f"{m['split_accuracy']:.3f}" if 'split_accuracy' in m else "N/A"
        non_split_acc = f"{m['non_split_accuracy']:.3f}" if 'non_split_accuracy' in m else "N/A"

        print(f"{label:<15} {model:<20} {prompt:<25} {split_acc:>12} {non_split_acc:>15}")


def print_heuristic_summary(heuristic_analysis: Dict):
    """Print heuristic analysis summary."""
    if not heuristic_analysis:
        return

    if 'by_heuristic_count' in heuristic_analysis and heuristic_analysis['by_heuristic_count']:
        print("\n" + "=" * 90)
        print("PERFORMANCE BY NUMBER OF HEURISTICS")
        print("=" * 90)

        for count, metrics in sorted(heuristic_analysis['by_heuristic_count'].items()):
            acc_str = format_metric_with_ci('accuracy', metrics)
            print(f"  {count:<20} Accuracy: {acc_str:>35}  (n={metrics['total_samples']})")

    if 'by_combination' in heuristic_analysis and heuristic_analysis['by_combination']:
        print("\n" + "=" * 90)
        print("PERFORMANCE BY HEURISTIC COMBINATION")
        print("=" * 90)

        for combo, metrics in sorted(heuristic_analysis['by_combination'].items()):
            acc_str = format_metric_with_ci('accuracy', metrics)
            print(f"  {combo:<30} Accuracy: {acc_str:>35}  (n={metrics['total_samples']})")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze merge comparison task performance",
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
  %(prog)s --config experiments.txt --table full --split-analysis --output-dir ./reports

  # Compact table only (accuracy + CI)
  %(prog)s --config experiments.txt --table compact

Config file format (experiments.txt):
  GPT-4,results/gpt4_baseline.csv
  Claude,results/claude_baseline.csv
  GPT-4+Hints,results/gpt4_with_hints.csv

Note:
  - If K repetitions are detected (K>1), both individual predictions AND
    majority vote (cons@K) metrics are automatically calculated and shown
  - K is detected automatically from operation_id groupings
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
    parser.add_argument("--split-analysis", action="store_true",
                       help="Show split vs non-split analysis")
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
    print("MERGE COMPARISON ANALYSIS")
    print("=" * 60)

    dfs = []
    for file_path, label in zip(file_paths, labels):
        print(f"\nLoading {label}: {file_path}")
        df = load_results(file_path)

        # Validate merge comparison data
        required_columns = ['correct_answer', 'model_chosen_id']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return

        # Filter for merge comparison if task column exists
        if 'task' in df.columns:
            df = df[df['task'] == 'merge_comparison']

        if len(df) == 0:
            print(f"Warning: No merge comparison data found in {label}")
            continue

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

    if args.split_analysis:
        print_split_analysis_table(results_individual)

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

        if args.split_analysis:
            print_split_analysis_table(results_majority)

    # Heuristic analysis
    if 'heuristic_analysis' in report:
        print_heuristic_summary(report['heuristic_analysis'])

    # ========================================================================
    # Save Outputs
    # ========================================================================
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        # Save JSON report
        report_file = os.path.join(args.output_dir, 'merge_comparison_analysis_report.json')
        with open(report_file, 'w') as f:
            json.dump(clean_report_for_json(report), f, indent=2)
        print(f"\nReport saved to: {report_file}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
