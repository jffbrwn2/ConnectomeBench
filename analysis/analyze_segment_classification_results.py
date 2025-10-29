#!/usr/bin/env python3
"""
Analyze and compare segment classification results across multiple models.
Usage: python analyze_segment_classification_results.py config.txt
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats


def load_config(config_path):
    """Load config file with format: Label,path

    Supports multiple paths per label - they will be concatenated.
    """
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',', 1)  # Split on first comma only
            if len(parts) == 2:
                label, path = parts[0].strip(), parts[1].strip()
                if label not in config:
                    config[label] = []
                config[label].append(path)
    return config


def load_and_concat_csvs(paths, label):
    """Load multiple CSVs and concatenate them"""
    dfs = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  WARNING: File not found: {path}")
            continue
        df = pd.read_csv(path)
        dfs.append(df)
        print(f"  Loaded {len(df)} samples from {os.path.basename(path)}")

    if not dfs:
        raise ValueError(f"No valid CSV files found for {label}")

    concatenated = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(concatenated)} samples from {len(dfs)} file(s)")
    return concatenated


def deduplicate_ground_truth(df):
    """Remove duplicate entries from ground truth based on root IDs"""
    df_dedup = df.drop_duplicates(subset=['proofread root id', 'current root id'], keep='first')

    if len(df_dedup) < len(df):
        print(f"  Deduplicated: {len(df)} → {len(df_dedup)} samples")

    return df_dedup


def match_predictions_to_ground_truth(ground_truth_df, predictions_df, use_majority_vote=True,
                                      filter_nones=True):
    """Match model predictions to ground truth based on root IDs

    Args:
        ground_truth_df: DataFrame with ground truth labels
        predictions_df: DataFrame with model predictions
        use_majority_vote: If True, use majority vote for multiple predictions (consensus@K)
                          If False, return all individual predictions (pass@1)
        filter_nones: If True, filters out None values (new behavior)
                     If False, keeps None values in result (original behavior)

    Returns:
        matched_results: Array of predictions matched to ground truth
        matched_ground_truth: Array of ground truth labels (repeated if not using majority vote)
    """
    # Standardize column names
    predictions_df = predictions_df.rename(columns={
        "proofread_root_id": "proofread root id",
        "current_root_id": "current root id"
    })

    if use_majority_vote and not filter_nones:
        # Original behavior: pre-allocate array and overwrite at index i
        matched_results = np.array([None] * len(ground_truth_df))
        matched_ground_truth = ground_truth_df['human answer 1'].values

        for i in range(len(ground_truth_df)):
            proofread_id = ground_truth_df.iloc[i]['proofread root id']
            current_id = ground_truth_df.iloc[i]['current root id']

            # Find matching predictions
            matches = predictions_df[
                (predictions_df['proofread root id'] == proofread_id) &
                (predictions_df['current root id'] == current_id)
            ]

            if len(matches) > 0:
                # Format answers to single letter
                answers = matches['llm_answer'].values
                labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
                formatted = [str(a)[:1] if a not in labels else a for a in answers]

                # Use most common answer (overwrites at index i)
                most_common = Counter(formatted).most_common(1)[0][0]
                matched_results[i] = most_common
            # else: leave as None (already initialized)

        return matched_results, matched_ground_truth

    else:
        # New behavior: append to list and optionally filter Nones
        matched_results = []
        matched_ground_truth = []

        for i in range(len(ground_truth_df)):
            proofread_id = ground_truth_df.iloc[i]['proofread root id']
            current_id = ground_truth_df.iloc[i]['current root id']
            ground_truth_label = ground_truth_df.iloc[i]['human answer 1']

            # Find matching predictions
            matches = predictions_df[
                (predictions_df['proofread root id'] == proofread_id) &
                (predictions_df['current root id'] == current_id)
            ]

            if len(matches) > 0:
                # Format answers to single letter
                answers = matches['llm_answer'].values
                labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
                formatted = [str(a)[:1] if a not in labels else a for a in answers]

                if use_majority_vote:
                    # Consensus@K: Use most common answer
                    most_common = Counter(formatted).most_common(1)[0][0]
                    matched_results.append(most_common)
                    matched_ground_truth.append(ground_truth_label)
                else:
                    # Pass@1: Include all individual predictions
                    for prediction in formatted:
                        matched_results.append(prediction)
                        matched_ground_truth.append(ground_truth_label)
            else:
                if use_majority_vote:
                    matched_results.append(None)
                    matched_ground_truth.append(ground_truth_label)

        return np.array(matched_results), np.array(matched_ground_truth)


def calculate_metrics(ground_truth, predictions, categories):
    """Calculate accuracy, precision, and recall for each category"""
    accuracy = []
    precision = []
    recall = []

    for category in categories:
        # True positives: model predicted this category and it was correct
        tp = sum((predictions == category) & (ground_truth == category))
        # False positives: model predicted this category but it was wrong
        fp = sum((predictions == category) & (ground_truth != category))
        # False negatives: model didn't predict this category when it should have
        fn = sum((predictions != category) & (ground_truth == category))

        # Calculate metrics
        total_in_category = sum(ground_truth == category)
        acc = tp / total_in_category if total_in_category > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0

        accuracy.append(acc)
        precision.append(prec)
        recall.append(rec)

    return accuracy, precision, recall


def bootstrap_confidence_interval(ground_truth, predictions, categories,
                                  n_bootstrap=1000, confidence_level=0.95):
    """Calculate bootstrap confidence intervals for metrics

    Args:
        ground_truth: Array of ground truth labels
        predictions: Array of predicted labels
        categories: List of category labels
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 0.95)

    Returns:
        Dictionary with confidence intervals for each metric
    """
    n_samples = len(ground_truth)
    alpha = 1 - confidence_level

    # Store bootstrap results
    bootstrap_accuracy = []
    bootstrap_precision = []
    bootstrap_recall = []
    bootstrap_bulk_accuracy = []
    bootstrap_balanced_accuracy = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        gt_sample = ground_truth[indices]
        pred_sample = predictions[indices]

        # Calculate metrics for this bootstrap sample
        acc, prec, rec = calculate_metrics(gt_sample, pred_sample, categories)
        bulk_acc = sum(pred_sample == gt_sample) / n_samples
        balanced_acc = np.mean(rec)

        bootstrap_accuracy.append(acc)
        bootstrap_precision.append(prec)
        bootstrap_recall.append(rec)
        bootstrap_bulk_accuracy.append(bulk_acc)
        bootstrap_balanced_accuracy.append(balanced_acc)

    # Convert to numpy arrays for easier indexing
    bootstrap_accuracy = np.array(bootstrap_accuracy)  # shape: (n_bootstrap, n_categories)
    bootstrap_precision = np.array(bootstrap_precision)
    bootstrap_recall = np.array(bootstrap_recall)

    # Calculate confidence intervals
    results = {
        'per_category': {},
        'overall_accuracy': {
            'lower': np.percentile(bootstrap_bulk_accuracy, 100 * alpha / 2),
            'upper': np.percentile(bootstrap_bulk_accuracy, 100 * (1 - alpha / 2))
        },
        'balanced_accuracy': {
            'lower': np.percentile(bootstrap_balanced_accuracy, 100 * alpha / 2),
            'upper': np.percentile(bootstrap_balanced_accuracy, 100 * (1 - alpha / 2))
        }
    }

    for i, category in enumerate(categories):
        results['per_category'][category] = {
            'accuracy': {
                'lower': np.percentile(bootstrap_accuracy[:, i], 100 * alpha / 2),
                'upper': np.percentile(bootstrap_accuracy[:, i], 100 * (1 - alpha / 2))
            },
            'precision': {
                'lower': np.percentile(bootstrap_precision[:, i], 100 * alpha / 2),
                'upper': np.percentile(bootstrap_precision[:, i], 100 * (1 - alpha / 2))
            },
            'recall': {
                'lower': np.percentile(bootstrap_recall[:, i], 100 * alpha / 2),
                'upper': np.percentile(bootstrap_recall[:, i], 100 * (1 - alpha / 2))
            }
        }

    return results


def detect_k_value(ground_truth_df, predictions_df):
    """Detect K (number of repetitions per sample)"""
    # Standardize column names
    predictions_df = predictions_df.rename(columns={
        "proofread_root_id": "proofread root id",
        "current_root_id": "current root id"
    })

    # Count how many predictions per ground truth sample (only for samples with predictions)
    k_values = []
    for i in range(len(ground_truth_df)):
        proofread_id = ground_truth_df.iloc[i]['proofread root id']
        current_id = ground_truth_df.iloc[i]['current root id']

        matches = predictions_df[
            (predictions_df['proofread root id'] == proofread_id) &
            (predictions_df['current root id'] == current_id)
        ]

        # Only count samples that have at least one prediction
        if len(matches) > 0:
            k_values.append(len(matches))

    # Return most common K value (excluding samples with no predictions)
    if k_values:
        return Counter(k_values).most_common(1)[0][0]
    return 1


def plot_results(models, metrics_by_model, categories, label_names, title, save_path=None):
    """Create comparison plot for all models"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    bar_width = 0.8 / len(models)
    base = np.arange(len(categories))

    metric_names = ['Accuracy', 'Precision', 'Recall']
    legend_handles = []

    for ax_idx, (ax, metric_name) in enumerate(zip(axes, metric_names)):
        for model_idx, model in enumerate(models):
            offset = base + model_idx * bar_width
            bars = ax.bar(offset, metrics_by_model[model][ax_idx], width=bar_width, label=model)
            if ax_idx == 0:
                legend_handles.append(bars[0])

        ax.set_xticks(base + bar_width * (len(models) - 1) / 2)
        ax.set_xticklabels(label_names, rotation=45, ha="right")
        ax.set_ylim([0, 1.1])
        ax.set_ylabel(metric_name)

    fig.legend(legend_handles, models, loc="lower center",
               bbox_to_anchor=(0.5, 0), ncol=min(len(models), 4))
    fig.suptitle(title, fontsize=16, y=0.95)
    fig.tight_layout(rect=[0, 0.1, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Analyze segment classification results with pass@1 and consensus@K metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config file format:
  Each line: Label,path/to/csv
  Must include one entry labeled 'human' for ground truth
  Multiple paths with same label will be concatenated

Metrics:
  - pass@1: Accuracy on individual predictions (all K predictions evaluated separately)
  - consensus@K: Accuracy using majority vote across K predictions per sample

  If K=1 (single prediction per sample), only pass@1 is shown.
  If K>1 (multiple predictions per sample), both pass@1 and consensus@K are computed.

Confidence Intervals:
  - Bootstrap confidence intervals are calculated by default (1000 samples, 95%% CI)
  - Use --no-confidence-intervals to skip (faster)
  - Customize with --n-bootstrap and --confidence-level

Deduplication:
  - Ground truth samples are automatically deduplicated by (proofread_root_id, current_root_id)
  - Duplicate rows with same answer are merged, keeping first occurrence
  - Use --no-dedup to disable this behavior

Example config.txt:
  # Single file per model
  human,/path/to/human_analysis_results.csv
  Claude Sonnet,/path/to/claude_analysis_results.csv
  GPT-4.1,/path/to/gpt4_analysis_results.csv

  # Multiple files per model (will be concatenated)
  human,/path/to/dir1/human_analysis_results.csv
  human,/path/to/dir2/human_analysis_results.csv
  Claude Sonnet,/path/to/dir1/claude_results.csv
  Claude Sonnet,/path/to/dir2/claude_results.csv

Usage examples:
  # Basic analysis with plot display
  %(prog)s config.txt

  # Save results to directory
  %(prog)s config.txt --output-dir ./reports

  # Save results and plots
  %(prog)s config.txt --output-dir ./reports --save-plot

  # Custom categories and labels
  %(prog)s config.txt --categories "a,b,c" --labels "Type A,Type B,Type C"

  # Use original behavior (match samples without predictions as errors)
  %(prog)s config.txt --original-behavior

  # Skip confidence intervals (faster)
  %(prog)s config.txt --no-confidence-intervals

  # Custom confidence interval settings
  %(prog)s config.txt --n-bootstrap 5000 --confidence-level 0.99
        """
    )
    parser.add_argument('config', help='Config file with format: Label,path')
    parser.add_argument('--categories', default='a,b,c,d,e',
                        help='Comma-separated category labels (default: a,b,c,d,e)')
    parser.add_argument('--labels',
                        default='single neuron,multiple somas,processes,nucleus,non-neuronal',
                        help='Comma-separated category names')
    parser.add_argument('--title', default='Model Performance Comparison',
                        help='Plot title')
    parser.add_argument('--output-dir',
                        help='Directory to save analysis outputs (JSON report and plots)')
    parser.add_argument('--save-plot', action='store_true',
                        help='Save plot to output directory (requires --output-dir)')
    parser.add_argument('--original-behavior', action='store_true',
                        help='Use original matching behavior (overwrites at index, includes None as errors). Default is new behavior (filters Nones).')
    parser.add_argument('--no-dedup', action='store_true',
                        help='Do not deduplicate ground truth samples (by default, duplicates are removed)')
    parser.add_argument('--no-confidence-intervals', action='store_true',
                        help='Skip confidence interval calculations (faster)')
    parser.add_argument('--confidence-level', type=float, default=0.95,
                        help='Confidence level for bootstrap intervals (default: 0.95)')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='Number of bootstrap samples (default: 1000)')
    args = parser.parse_args()

    # Parse categories and labels
    categories = [c.strip() for c in args.categories.split(',')]
    label_names = [l.strip() for l in args.labels.split(',')]

    if len(categories) != len(label_names):
        raise ValueError("Number of categories must match number of labels")

    # Load config
    config = load_config(args.config)

    # Identify ground truth
    ground_truth_key = None
    for key in ['human', 'ground_truth', 'Human', 'Ground Truth']:
        if key in config:
            ground_truth_key = key
            break

    if not ground_truth_key:
        raise ValueError("Config must include 'human' or 'ground_truth' entry")

    # Load ground truth
    print(f"\nLoading ground truth ({ground_truth_key}):")
    ground_truth_df = load_and_concat_csvs(config[ground_truth_key], ground_truth_key)

    # Deduplicate ground truth (unless --no-dedup flag is set)
    if not args.no_dedup:
        ground_truth_df = deduplicate_ground_truth(ground_truth_df)

    # Process each model
    models = [k for k in config.keys() if k != ground_truth_key]

    # Separate results for pass@1 and consensus@K
    pass1_metrics = {}
    consensus_metrics = {}
    pass1_results = {}
    consensus_results = {}
    k_values = {}

    print(f"\nAnalyzing {len(models)} models...")
    print("-" * 80)

    for model in models:
        print(f"\nProcessing: {model}")
        predictions_df = load_and_concat_csvs(config[model], model)

        # Detect K value
        k = detect_k_value(ground_truth_df, predictions_df)
        k_values[model] = k
        print(f"  Detected K={k} predictions per sample")

        # ====================================================================
        # Pass@1: Individual predictions
        # ====================================================================
        print(f"  Computing pass@1 (individual predictions)...")
        matched_predictions, matched_ground_truth = match_predictions_to_ground_truth(
            ground_truth_df, predictions_df, use_majority_vote=False
        )

        # Filter out None values
        valid_mask = matched_predictions != None
        n_valid = sum(valid_mask)
        n_total = len(matched_predictions)
        print(f"    Matched {n_valid}/{n_total} individual predictions")

        if n_valid > 0:
            # Calculate metrics
            accuracy, precision, recall = calculate_metrics(
                matched_ground_truth[valid_mask],
                matched_predictions[valid_mask],
                categories
            )

            pass1_metrics[model] = [accuracy, precision, recall]

            # Summary statistics
            bulk_accuracy = sum(matched_predictions[valid_mask] == matched_ground_truth[valid_mask]) / n_valid
            balanced_accuracy = np.mean(recall)

            # Calculate confidence intervals if requested
            ci_results = None
            if not args.no_confidence_intervals:
                print(f"    Computing confidence intervals...")
                ci_results = bootstrap_confidence_interval(
                    matched_ground_truth[valid_mask],
                    matched_predictions[valid_mask],
                    categories,
                    n_bootstrap=args.n_bootstrap,
                    confidence_level=args.confidence_level
                )
                print(f"    Overall Accuracy: {bulk_accuracy:.3f} [{ci_results['overall_accuracy']['lower']:.3f}-{ci_results['overall_accuracy']['upper']:.3f}]")
                print(f"    Balanced Accuracy: {balanced_accuracy:.3f} [{ci_results['balanced_accuracy']['lower']:.3f}-{ci_results['balanced_accuracy']['upper']:.3f}]")
            else:
                print(f"    Overall Accuracy: {bulk_accuracy:.3f}")
                print(f"    Balanced Accuracy: {balanced_accuracy:.3f}")

            pass1_results[model] = {
                'k_value': int(k),
                'matched_samples': int(n_valid),
                'total_samples': int(n_total),
                'overall_accuracy': float(bulk_accuracy),
                'balanced_accuracy': float(balanced_accuracy),
                'per_category_metrics': {
                    cat: {
                        'accuracy': float(accuracy[i]),
                        'precision': float(precision[i]),
                        'recall': float(recall[i])
                    }
                    for i, cat in enumerate(categories)
                }
            }

            # Add confidence intervals to results if computed
            if ci_results:
                pass1_results[model]['overall_accuracy_ci'] = {
                    'lower': float(ci_results['overall_accuracy']['lower']),
                    'upper': float(ci_results['overall_accuracy']['upper'])
                }
                pass1_results[model]['balanced_accuracy_ci'] = {
                    'lower': float(ci_results['balanced_accuracy']['lower']),
                    'upper': float(ci_results['balanced_accuracy']['upper'])
                }
                for i, cat in enumerate(categories):
                    pass1_results[model]['per_category_metrics'][cat]['accuracy_ci'] = {
                        'lower': float(ci_results['per_category'][cat]['accuracy']['lower']),
                        'upper': float(ci_results['per_category'][cat]['accuracy']['upper'])
                    }
                    pass1_results[model]['per_category_metrics'][cat]['precision_ci'] = {
                        'lower': float(ci_results['per_category'][cat]['precision']['lower']),
                        'upper': float(ci_results['per_category'][cat]['precision']['upper'])
                    }
                    pass1_results[model]['per_category_metrics'][cat]['recall_ci'] = {
                        'lower': float(ci_results['per_category'][cat]['recall']['lower']),
                        'upper': float(ci_results['per_category'][cat]['recall']['upper'])
                    }

        # ====================================================================
        # Consensus@K: Majority vote (if K > 1)
        # ====================================================================
        if k > 1:
            behavior = "original behavior" if args.original_behavior else "new behavior"
            print(f"  Computing consensus@{k} (majority vote - {behavior})...")
            matched_predictions, matched_ground_truth = match_predictions_to_ground_truth(
                ground_truth_df, predictions_df, use_majority_vote=True,
                filter_nones=not args.original_behavior
            )

            if args.original_behavior:
                # Original behavior: Don't filter None values - count them as errors
                n_total = len(matched_predictions)
                n_with_predictions = sum(matched_predictions != None)
                print(f"    Total samples: {n_total}")
                print(f"    Samples with predictions: {n_with_predictions}")
                print(f"    Samples without predictions (None): {n_total - n_with_predictions}")

                # Calculate metrics including None as wrong predictions
                # For metrics calculation, replace None with a dummy value that won't match
                predictions_for_metrics = np.array([p if p is not None else 'NONE_PLACEHOLDER'
                                                   for p in matched_predictions])

                accuracy, precision, recall = calculate_metrics(
                    matched_ground_truth,
                    predictions_for_metrics,
                    categories
                )

                consensus_metrics[model] = [accuracy, precision, recall]

                # Summary statistics - None counts as wrong
                correct = sum((matched_predictions != None) &
                             (matched_predictions == matched_ground_truth))
                bulk_accuracy = correct / n_total
                balanced_accuracy = np.mean(recall)
                print(f"    Overall Accuracy: {bulk_accuracy:.3f}")
                print(f"    Balanced Accuracy: {balanced_accuracy:.3f}")

                consensus_results[model] = {
                    'k_value': int(k),
                    'total_samples': int(n_total),
                    'samples_with_predictions': int(n_with_predictions),
                    'samples_without_predictions': int(n_total - n_with_predictions),
                    'overall_accuracy': float(bulk_accuracy),
                    'balanced_accuracy': float(balanced_accuracy),
                    'per_category_metrics': {
                        cat: {
                            'accuracy': float(accuracy[i]),
                            'precision': float(precision[i]),
                            'recall': float(recall[i])
                        }
                        for i, cat in enumerate(categories)
                    }
                }
            else:
                # New behavior: Filter None values
                valid_mask = matched_predictions != None
                n_valid = sum(valid_mask)
                n_total = len(matched_predictions)
                print(f"    Matched {n_valid}/{n_total} samples")

                if n_valid > 0:
                    # Calculate metrics
                    accuracy, precision, recall = calculate_metrics(
                        matched_ground_truth[valid_mask],
                        matched_predictions[valid_mask],
                        categories
                    )

                    consensus_metrics[model] = [accuracy, precision, recall]

                    # Summary statistics
                    bulk_accuracy = sum(matched_predictions[valid_mask] == matched_ground_truth[valid_mask]) / n_valid
                    balanced_accuracy = np.mean(recall)

                    # Calculate confidence intervals if requested
                    ci_results = None
                    if not args.no_confidence_intervals:
                        print(f"    Computing confidence intervals...")
                        ci_results = bootstrap_confidence_interval(
                            matched_ground_truth[valid_mask],
                            matched_predictions[valid_mask],
                            categories,
                            n_bootstrap=args.n_bootstrap,
                            confidence_level=args.confidence_level
                        )
                        print(f"    Overall Accuracy: {bulk_accuracy:.3f} [{ci_results['overall_accuracy']['lower']:.3f}-{ci_results['overall_accuracy']['upper']:.3f}]")
                        print(f"    Balanced Accuracy: {balanced_accuracy:.3f} [{ci_results['balanced_accuracy']['lower']:.3f}-{ci_results['balanced_accuracy']['upper']:.3f}]")
                    else:
                        print(f"    Overall Accuracy: {bulk_accuracy:.3f}")
                        print(f"    Balanced Accuracy: {balanced_accuracy:.3f}")

                    consensus_results[model] = {
                        'k_value': int(k),
                        'matched_samples': int(n_valid),
                        'total_samples': int(n_total),
                        'overall_accuracy': float(bulk_accuracy),
                        'balanced_accuracy': float(balanced_accuracy),
                        'per_category_metrics': {
                            cat: {
                                'accuracy': float(accuracy[i]),
                                'precision': float(precision[i]),
                                'recall': float(recall[i])
                            }
                            for i, cat in enumerate(categories)
                        }
                    }

                    # Add confidence intervals to results if computed
                    if ci_results:
                        consensus_results[model]['overall_accuracy_ci'] = {
                            'lower': float(ci_results['overall_accuracy']['lower']),
                            'upper': float(ci_results['overall_accuracy']['upper'])
                        }
                        consensus_results[model]['balanced_accuracy_ci'] = {
                            'lower': float(ci_results['balanced_accuracy']['lower']),
                            'upper': float(ci_results['balanced_accuracy']['upper'])
                        }
                        for i, cat in enumerate(categories):
                            consensus_results[model]['per_category_metrics'][cat]['accuracy_ci'] = {
                                'lower': float(ci_results['per_category'][cat]['accuracy']['lower']),
                                'upper': float(ci_results['per_category'][cat]['accuracy']['upper'])
                            }
                            consensus_results[model]['per_category_metrics'][cat]['precision_ci'] = {
                                'lower': float(ci_results['per_category'][cat]['precision']['lower']),
                                'upper': float(ci_results['per_category'][cat]['precision']['upper'])
                            }
                            consensus_results[model]['per_category_metrics'][cat]['recall_ci'] = {
                                'lower': float(ci_results['per_category'][cat]['recall']['lower']),
                                'upper': float(ci_results['per_category'][cat]['recall']['upper'])
                            }

    # Build report
    max_k = max(k_values.values()) if k_values else 1
    report = {
        'config_file': args.config,
        'categories': categories,
        'category_labels': label_names,
        'k_values': k_values,
        'max_k': max_k,
        'ground_truth_sources': config[ground_truth_key],
        'model_sources': {model: config[model] for model in models},
        'pass1_results': pass1_results,
        'consensus_results': consensus_results if consensus_results else None
    }

    # ========================================================================
    # Display Results
    # ========================================================================
    print(f"\n{'='*80}")
    print("PASS@1 RESULTS (Individual Predictions)")
    print(f"{'='*80}")
    for model in models:
        if model in pass1_results:
            res = pass1_results[model]
            print(f"\n{model}:")
            if 'overall_accuracy_ci' in res:
                print(f"  Overall Accuracy: {res['overall_accuracy']:.3f} [{res['overall_accuracy_ci']['lower']:.3f}-{res['overall_accuracy_ci']['upper']:.3f}]")
                print(f"  Balanced Accuracy: {res['balanced_accuracy']:.3f} [{res['balanced_accuracy_ci']['lower']:.3f}-{res['balanced_accuracy_ci']['upper']:.3f}]")
            else:
                print(f"  Overall Accuracy: {res['overall_accuracy']:.3f}")
                print(f"  Balanced Accuracy: {res['balanced_accuracy']:.3f}")
            print(f"  Samples: {res['matched_samples']}")

    if max_k > 1 and consensus_results:
        behavior_label = "Original Behavior" if args.original_behavior else "New Behavior"
        print(f"\n{'='*80}")
        print(f"CONSENSUS@{max_k} RESULTS (Majority Vote - {behavior_label})")
        print(f"{'='*80}")
        if args.original_behavior:
            print("Note: Samples without predictions count as errors")
        for model in models:
            if model in consensus_results:
                res = consensus_results[model]
                print(f"\n{model}:")
                if 'overall_accuracy_ci' in res:
                    print(f"  Overall Accuracy: {res['overall_accuracy']:.3f} [{res['overall_accuracy_ci']['lower']:.3f}-{res['overall_accuracy_ci']['upper']:.3f}]")
                    print(f"  Balanced Accuracy: {res['balanced_accuracy']:.3f} [{res['balanced_accuracy_ci']['lower']:.3f}-{res['balanced_accuracy_ci']['upper']:.3f}]")
                else:
                    print(f"  Overall Accuracy: {res['overall_accuracy']:.3f}")
                    print(f"  Balanced Accuracy: {res['balanced_accuracy']:.3f}")
                if args.original_behavior:
                    print(f"  Total Samples: {res['total_samples']}")
                    print(f"  With Predictions: {res['samples_with_predictions']}")
                    print(f"  Without Predictions: {res['samples_without_predictions']}")
                else:
                    print(f"  Samples: {res['matched_samples']}")

    # ========================================================================
    # Save Outputs
    # ========================================================================
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        # Save JSON report
        report_file = os.path.join(args.output_dir, 'segment_classification_analysis_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n{'='*80}")
        print(f"Report saved to: {report_file}")

    # ========================================================================
    # Plot results
    # ========================================================================
    if pass1_metrics:
        print(f"\n{'='*80}")
        print("Generating plots...")

        # Pass@1 plot
        plot_path_pass1 = None
        if args.save_plot and args.output_dir:
            plot_path_pass1 = os.path.join(args.output_dir, 'segment_classification_pass1.png')
        elif args.save_plot and not args.output_dir:
            print("Warning: --save-plot requires --output-dir, displaying plots instead")

        plot_results(list(pass1_metrics.keys()), pass1_metrics,
                    categories, label_names, f"{args.title} - Pass@1", save_path=plot_path_pass1)

        # Consensus@K plot (if K > 1)
        if max_k > 1 and consensus_metrics:
            plot_path_consensus = None
            if args.save_plot and args.output_dir:
                plot_path_consensus = os.path.join(args.output_dir, f'segment_classification_consensus_{max_k}.png')

            plot_results(list(consensus_metrics.keys()), consensus_metrics,
                        categories, label_names, f"{args.title} - Consensus@{max_k}", save_path=plot_path_consensus)
    else:
        print("\nNo valid results to plot")

    if args.output_dir:
        print(f"\n{'='*80}")
        print("Analysis complete! Results saved to:", args.output_dir)
        print(f"{'='*80}")


if __name__ == '__main__':
    main()
