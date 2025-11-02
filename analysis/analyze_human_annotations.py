#!/usr/bin/env python3
"""
Master script for comprehensive human annotation analysis.

Analyzes human annotations for both pointwise (identification) and pairwise (comparison)
tasks with full metrics, bootstrapped confidence intervals, and automatic task detection.

Features:
- Accuracy metrics with bootstrap confidence intervals
- Automatic grouping by model and prompt mode
- Support for both comparison and identification tasks
- Confusion matrix visualization
- Clean table output
- JSON export
- Automatic task type inference from data

Usage:
  # Single file
  python analyze_human_annotations.py results.json

  # Multiple files with labels
  python analyze_human_annotations.py results1.json results2.json --labels "Exp1" "Exp2"

  # Using config file
  python analyze_human_annotations.py --config human_annotation_configs.txt --output-dir ./reports
"""

import sys
import os
import re

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis_utils import (
    calculate_merge_identification_metrics,
    format_metric_with_ci,
    clean_report_for_json
)

import pandas as pd
import json
import argparse
from typing import Dict, List, Any, Tuple


# ============================================================================
# Data Loading
# ============================================================================

def load_json_data(file_path: str) -> pd.DataFrame:
    """Load data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Expected list of records in {file_path}")


def infer_task_type(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Infer task type from DataFrame columns and content.

    Returns:
        Tuple of (error_type, task_type)
        error_type: 'merge' or 'split'
        task_type: 'comparison' or 'identification'
    """
    if 'task' in df.columns:
        task_values = df['task'].unique()
        if 'merge_comparison' in task_values:
            return ('merge', 'comparison')
        elif 'split_comparison' in task_values:
            return ('split', 'comparison')
        elif 'merge_identification' in task_values:
            return ('merge', 'identification')
        elif 'split_identification' in task_values:
            return ('split', 'identification')

    # Fallback: check for task-specific columns
    if 'root_id_requires_split' in df.columns:
        return ('split', 'comparison')
    elif 'is_split' in df.columns:
        return ('split', 'identification')
    elif 'is_correct_merge' in df.columns:
        return ('merge', 'identification')
    else:
        return ('merge', 'comparison')


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_merge_comparison_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare merge comparison task data for analysis.

    Accuracy: str(correct_answer[0]) == str(manual_choice)
    """
    df = df.copy()

    # Filter to only records with manual_choice
    df = df[df['manual_choice'].notna()].copy()

    # Convert correct_answer to string for comparison
    def normalize_answer(val):
        if isinstance(val, list):
            return str(val[0]) if len(val) > 0 else 'none'
        return str(val)

    df['correct_answer_str'] = df['correct_answer'].apply(normalize_answer)
    df['manual_choice_str'] = df['manual_choice'].astype(str)

    # Human is correct if manual_choice matches correct answer
    df['is_correct'] = df['correct_answer_str'] == df['manual_choice_str']

    # For metrics: binary classification where ground truth is always "should be correct"
    df['is_correct_merge'] = True
    df['model_prediction'] = df['is_correct'].apply(lambda x: '1' if x else '-1')

    return df


def prepare_split_comparison_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare split comparison task data for analysis.

    Accuracy: root_id_requires_split == manual_choice
    """
    df = df.copy()

    # Filter to only records with manual_choice
    df = df[df['manual_choice'].notna()].copy()

    df['root_id_requires_split_str'] = df['root_id_requires_split'].astype(str)
    df['manual_choice_str'] = df['manual_choice'].astype(str)

    # Human is correct if manual_choice matches root_id_requires_split
    df['is_correct'] = df['root_id_requires_split_str'] == df['manual_choice_str']

    # For metrics: binary classification where ground truth is always "should be correct"
    df['is_correct_merge'] = True
    df['model_prediction'] = df['is_correct'].apply(lambda x: '1' if x else '-1')

    return df


def prepare_merge_identification_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare merge identification task data for analysis.

    Accuracy: is_correct_merge == manual_choice
    """
    df = df.copy()

    # Filter to only records with manual_choice
    df = df[df['manual_choice'].notna()].copy()

    # Ground truth is is_correct_merge
    df['ground_truth'] = df['is_correct_merge'].fillna(False)

    # Convert manual_choice to boolean
    def normalize_choice(val):
        val_str = str(val).lower()
        if val_str in ['yes', '1', '+1', 'true']:
            return True
        elif val_str in ['no', '-1', 'false', 'none']:
            return False
        return bool(val)

    df['manual_choice_bool'] = df['manual_choice'].apply(normalize_choice)

    # Human is correct if manual_choice matches ground truth
    df['is_correct'] = df['ground_truth'] == df['manual_choice_bool']

    # For metrics calculation
    df['is_correct_merge'] = True
    df['model_prediction'] = df['is_correct'].apply(lambda x: '1' if x else '-1')

    return df


def prepare_split_identification_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare split identification task data for analysis.

    Accuracy: is_split == manual_choice
    """
    df = df.copy()

    # Filter to only records with manual_choice
    df = df[df['manual_choice'].notna()].copy()

    # Ground truth is is_split
    df['ground_truth'] = df['is_split'].fillna(False)

    # Convert manual_choice to boolean
    def normalize_choice(val):
        val_str = str(val).lower()
        if val_str in ['yes', '1', '+1', 'true']:
            return True
        elif val_str in ['no', '-1', 'false', 'none']:
            return False
        return bool(val)

    df['manual_choice_bool'] = df['manual_choice'].apply(normalize_choice)

    # Human is correct if manual_choice matches ground truth
    df['is_correct'] = df['ground_truth'] == df['manual_choice_bool']

    # For metrics calculation
    df['is_correct_merge'] = True
    df['model_prediction'] = df['is_correct'].apply(lambda x: '1' if x else '-1')

    return df


# ============================================================================
# Configuration Analysis
# ============================================================================

def identify_configurations(dfs: List[pd.DataFrame], labels: List[str],
                           task_infos: List[Tuple[str, str]]) -> List[Dict]:
    """Identify all unique configurations (file x model x prompt_mode x task_type)."""
    configurations = []

    for df, label, (error_type, task_type) in zip(dfs, labels, task_infos):
        has_model = 'model' in df.columns and df['model'].nunique() > 1
        has_prompt = 'prompt_mode' in df.columns and df['prompt_mode'].nunique() > 1

        task_label = f"{error_type}_{task_type}"

        if has_model and has_prompt:
            for model in df['model'].unique():
                for prompt_mode in df['prompt_mode'].unique():
                    filtered_df = df[(df['model'] == model) & (df['prompt_mode'] == prompt_mode)]
                    if len(filtered_df) > 0:
                        configurations.append({
                            'label': label,
                            'model': str(model),
                            'prompt_mode': str(prompt_mode),
                            'error_type': error_type,
                            'task_type': task_type,
                            'task_label': task_label,
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
                        'error_type': error_type,
                        'task_type': task_type,
                        'task_label': task_label,
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
                        'error_type': error_type,
                        'task_type': task_type,
                        'task_label': task_label,
                        'df': filtered_df
                    })
        else:
            configurations.append({
                'label': label,
                'model': 'all',
                'prompt_mode': 'all',
                'error_type': error_type,
                'task_type': task_type,
                'task_label': task_label,
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
        print(f"  [{i}/{len(configurations)}] {config['label']} | {config['model']} | "
              f"{config['prompt_mode']} | {config['task_label']} (n={len(config['df'])})")

        metrics = calculate_merge_identification_metrics(
            config['df'],
            include_ci=include_ci,
            confidence_level=confidence_level
        )

        result = {
            'label': config['label'],
            'model': config['model'],
            'prompt_mode': config['prompt_mode'],
            'error_type': config['error_type'],
            'task_type': config['task_type'],
            'task_label': config['task_label'],
            'metrics': metrics
        }

        results.append(result)

    return results


def generate_unified_report(dfs: List[pd.DataFrame],
                           labels: List[str],
                           task_infos: List[Tuple[str, str]],
                           include_ci: bool = True,
                           confidence_level: float = 0.95) -> Dict:
    """Generate comprehensive report with full analysis."""
    report = {}

    print("\n" + "=" * 80)
    print("ANALYZING HUMAN ANNOTATIONS")
    print("=" * 80)

    configurations = identify_configurations(dfs, labels, task_infos)
    results = analyze_all_configurations(
        configurations,
        include_ci=include_ci,
        confidence_level=confidence_level
    )
    report['configurations'] = results

    # Group results by task type
    comparison_results = [r for r in results if r['task_type'] == 'comparison']
    identification_results = [r for r in results if r['task_type'] == 'identification']

    report['comparison_results'] = comparison_results
    report['identification_results'] = identification_results

    # Overall statistics
    report['overall_stats'] = {
        'total_samples': sum(len(config['df']) for config in configurations),
        'n_configurations': len(configurations),
        'n_files': len(labels),
        'n_comparison_tasks': len(comparison_results),
        'n_identification_tasks': len(identification_results)
    }

    return report


# ============================================================================
# Table Output
# ============================================================================

def print_compact_table(results: List[Dict], title: str = "ACCURACY SUMMARY"):
    """Print compact table with accuracy."""
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    print(f"{'File':<25} {'Model':<15} {'Prompt':<20} {'Task':<15} {'N':>6} {'Accuracy':>25}")
    print("-" * 120)

    for result in results:
        m = result['metrics']
        label = result['label'][:24]
        model = result['model'][:14]
        prompt = result['prompt_mode'][:19]
        task = result['task_label'][:14]
        n = m['total_samples']

        if 'accuracy_ci_lower' in m:
            acc_str = f"{m['accuracy']:.3f} [{m['accuracy_ci_lower']:.3f}-{m['accuracy_ci_upper']:.3f}]"
        else:
            acc_str = f"{m['accuracy']:.3f}"

        print(f"{label:<25} {model:<15} {prompt:<20} {task:<15} {n:>6} {acc_str:>25}")


def print_full_table(results: List[Dict], title: str = "COMPREHENSIVE RESULTS"):
    """Print full table with all metrics."""
    print("\n" + "=" * 165)
    print(title)
    print("=" * 165)

    print(f"{'File':<20} {'Model':<12} {'Prompt':<18} {'Task':<12} {'N':>5} "
          f"{'Accuracy':>25} {'Precision':>25} {'Recall':>25} {'F1':>25}")
    print("-" * 165)

    for result in results:
        m = result['metrics']
        label = result['label'][:19]
        model = result['model'][:11]
        prompt = result['prompt_mode'][:17]
        task = result['task_label'][:11]
        n = m['total_samples']

        def fmt(metric_name):
            if f'{metric_name}_ci_lower' in m:
                return f"{m[metric_name]:.3f} [{m[f'{metric_name}_ci_lower']:.3f}-{m[f'{metric_name}_ci_upper']:.3f}]"
            else:
                return f"{m[metric_name]:.3f}"

        print(f"{label:<20} {model:<12} {prompt:<18} {task:<12} {n:>5} "
              f"{fmt('accuracy'):>25} {fmt('precision'):>25} {fmt('recall'):>25} {fmt('f1_score'):>25}")


def print_confusion_matrix_table(results: List[Dict]):
    """Print confusion matrix values."""
    print("\n" + "=" * 140)
    print("CONFUSION MATRIX VALUES")
    print("=" * 140)

    print(f"{'File':<25} {'Model':<15} {'Prompt':<20} {'Task':<15} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6} {'FPR':>8}")
    print("-" * 140)

    for result in results:
        m = result['metrics']
        label = result['label'][:24]
        model = result['model'][:14]
        prompt = result['prompt_mode'][:19]
        task = result['task_label'][:14]

        tp = m['true_positives']
        fp = m['false_positives']
        tn = m['true_negatives']
        fn = m['false_negatives']
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"{label:<25} {model:<15} {prompt:<20} {task:<15} {tp:>6} {fp:>6} {tn:>6} {fn:>6} {fp_rate:>8.3f}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze human annotations for pointwise and pairwise tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  %(prog)s results.json

  # Multiple files with labels
  %(prog)s results1.json results2.json --labels "Fly merge" "Mouse merge"

  # Using a config file
  %(prog)s --config human_annotation_configs.txt --output-dir ./reports

Config file format:
  Fly merge (comparison),output/fly_merge_2048nm/comparison_manual_anotation.json
  Fly merge (identification),output/fly_merge_2048nm/identification_manual_anotation.json

Note:
  - Analyzes both comparison (pairwise) and identification (pointwise) tasks
  - Task type is automatically inferred from data
  - Bootstrap confidence intervals are calculated by default
        """
    )

    parser.add_argument("results_files", nargs='*',
                       help="Path(s) to results JSON file(s)")
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
            labels = [os.path.basename(f).replace('.json', '') for f in file_paths]
        else:
            labels = args.labels

        if len(labels) != len(file_paths):
            print(f"Error: Number of labels ({len(labels)}) must match files ({len(file_paths)})")
            return

    # ========================================================================
    # Load Data
    # ========================================================================
    print("=" * 80)
    print("HUMAN ANNOTATION ANALYSIS")
    print("=" * 80)

    dfs = []
    task_infos = []

    for file_path, label in zip(file_paths, labels):
        print(f"\nLoading {label}: {file_path}")
        df = load_json_data(file_path)

        # Infer task type
        error_type, task_type = infer_task_type(df)
        print(f"  Task type: {error_type}_{task_type}")
        task_infos.append((error_type, task_type))

        # Count records with manual_choice before filtering
        total_records = len(df)
        records_with_choice = sum(1 for _, row in df.iterrows() if 'manual_choice' in row and pd.notna(row['manual_choice']))

        # Prepare data based on task type
        if error_type == 'merge' and task_type == 'comparison':
            df = prepare_merge_comparison_data(df)
        elif error_type == 'split' and task_type == 'comparison':
            df = prepare_split_comparison_data(df)
        elif error_type == 'merge' and task_type == 'identification':
            df = prepare_merge_identification_data(df)
        elif error_type == 'split' and task_type == 'identification':
            df = prepare_split_identification_data(df)
        else:
            print(f"  Error: Unknown task type {error_type}_{task_type}")
            return

        print(f"  Loaded {len(df)} annotated samples (out of {total_records} total)")
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
        task_infos,
        include_ci=include_ci,
        confidence_level=args.confidence_level
    )

    # ========================================================================
    # Output
    # ========================================================================

    # Print results grouped by task type
    if report['comparison_results']:
        if args.table in ['compact', 'both']:
            print_compact_table(report['comparison_results'],
                              title="PAIRWISE COMPARISON TASKS - ACCURACY SUMMARY")

        if args.table in ['full', 'both']:
            print_full_table(report['comparison_results'],
                           title="PAIRWISE COMPARISON TASKS - COMPREHENSIVE RESULTS")

        if args.confusion_matrix:
            print("\n" + "=" * 140)
            print("PAIRWISE COMPARISON TASKS - CONFUSION MATRIX")
            print("=" * 140)
            print_confusion_matrix_table(report['comparison_results'])

    if report['identification_results']:
        if args.table in ['compact', 'both']:
            print_compact_table(report['identification_results'],
                              title="POINTWISE IDENTIFICATION TASKS - ACCURACY SUMMARY")

        if args.table in ['full', 'both']:
            print_full_table(report['identification_results'],
                           title="POINTWISE IDENTIFICATION TASKS - COMPREHENSIVE RESULTS")

        if args.confusion_matrix:
            print("\n" + "=" * 140)
            print("POINTWISE IDENTIFICATION TASKS - CONFUSION MATRIX")
            print("=" * 140)
            print_confusion_matrix_table(report['identification_results'])

    # Overall summary
    stats = report['overall_stats']
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total samples analyzed: {stats['total_samples']}")
    print(f"Total configurations: {stats['n_configurations']}")
    print(f"Files analyzed: {stats['n_files']}")
    print(f"Pairwise comparison tasks: {stats['n_comparison_tasks']}")
    print(f"Pointwise identification tasks: {stats['n_identification_tasks']}")

    # ========================================================================
    # Save Outputs
    # ========================================================================
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        report_file = os.path.join(args.output_dir, 'human_annotation_analysis_report.json')
        with open(report_file, 'w') as f:
            json.dump(clean_report_for_json(report), f, indent=2)
        print(f"\nReport saved to: {report_file}")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
