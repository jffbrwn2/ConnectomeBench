#!/usr/bin/env python3
"""
Script to compare merge identification results across multiple files or configurations.
"""

import pandas as pd
import json
import argparse
import os
from typing import Dict, List
from analyze_merge_performance import calculate_metrics, load_results

def compare_results(file_paths: List[str], labels: List[str] = None) -> Dict:
    """Compare results across multiple files."""
    if labels is None:
        labels = [f"File_{i+1}" for i in range(len(file_paths))]
    
    if len(labels) != len(file_paths):
        raise ValueError("Number of labels must match number of files")
    
    comparison = {}
    all_data = []
    
    for file_path, label in zip(file_paths, labels):
        print(f"Loading {label}: {file_path}")
        df = load_results(file_path)
        
        # Filter for merge identification if needed
        if 'task' in df.columns:
            df = df[df['task'] == 'merge_identification']
        
        if len(df) == 0:
            print(f"Warning: No merge identification data in {label}")
            continue
        
        # Apply majority voting if requested
        if args.majority_vote:
            original_len = len(df)
            df = perform_majority_voting(df)
            print(f"  {label}: Applied majority voting ({original_len} → {len(df)} samples)")
        
        metrics = calculate_metrics(df)
        comparison[label] = metrics
        
        # Add source for combined analysis
        df_copy = df.copy()
        df_copy['source'] = label
        all_data.append(df_copy)
    
    # Combined dataframe for cross-analysis
    combined_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    return comparison, combined_df

def print_comparison_table(comparison: Dict):
    """Print comparison in table format."""
    if not comparison:
        print("No data to compare")
        return
    
    print("\nCOMPARISON TABLE")
    print("=" * 80)
    
    # Headers
    metrics_to_show = ['total_samples', 'accuracy', 'precision', 'recall', 'f1_score', 'true_positives', 'false_positives', 'false_negatives']
    header_names = ['Samples', 'Accuracy', 'Precision', 'Recall', 'F1', 'TP', 'FP', 'FN']
    
    print(f"{'Source':<15} {' '.join(f'{h:>9}' for h in header_names)}")
    print("-" * 80)
    
    for source, metrics in comparison.items():
        values = [str(metrics[m]) if m == 'total_samples' else f"{metrics[m]:.3f}" if isinstance(metrics[m], float) else str(metrics[m]) 
                 for m in metrics_to_show]
        print(f"{source:<15} {' '.join(f'{v:>9}' for v in values)}")

def find_best_performers(comparison: Dict) -> Dict[str, str]:
    """Find best performing configurations for each metric."""
    if not comparison:
        return {}
    
    best = {}
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    for metric in metrics:
        best_score = max(comparison.values(), key=lambda x: x[metric])[metric]
        best_sources = [source for source, data in comparison.items() if data[metric] == best_score]
        best[metric] = best_sources[0] if len(best_sources) == 1 else f"Tie: {', '.join(best_sources)}"
    
    return best

def analyze_heuristic_impact(combined_df: pd.DataFrame) -> Dict:
    """Analyze impact of different heuristics across all data."""
    if 'prompt_mode' not in combined_df.columns:
        return {}
    
    # Extract base mode and heuristics
    combined_df = combined_df.copy()
    combined_df['base_mode'] = combined_df['prompt_mode'].apply(lambda x: x.split('+')[0])
    combined_df['has_heuristics'] = combined_df['prompt_mode'].apply(lambda x: '+' in x)
    combined_df['heuristic_count'] = combined_df['prompt_mode'].apply(lambda x: len([p for p in x.split('+')[1:] if p.startswith('heuristic')]))
    
    analysis = {}
    
    # Compare base modes
    base_mode_comparison = {}
    for base_mode in combined_df['base_mode'].unique():
        base_df = combined_df[combined_df['base_mode'] == base_mode]
        base_mode_comparison[base_mode] = calculate_metrics(base_df)
    analysis['by_base_mode'] = base_mode_comparison
    
    # Compare with vs without heuristics
    heuristic_comparison = {}
    for has_h in [False, True]:
        h_df = combined_df[combined_df['has_heuristics'] == has_h]
        if len(h_df) > 0:
            label = 'with_heuristics' if has_h else 'without_heuristics'
            heuristic_comparison[label] = calculate_metrics(h_df)
    analysis['heuristic_impact'] = heuristic_comparison
    
    # Compare by number of heuristics
    count_comparison = {}
    for count in sorted(combined_df['heuristic_count'].unique()):
        count_df = combined_df[combined_df['heuristic_count'] == count]
        count_comparison[f"{count}_heuristics"] = calculate_metrics(count_df)
    analysis['by_heuristic_count'] = count_comparison
    
    return analysis

def perform_majority_voting(df: pd.DataFrame) -> pd.DataFrame:
    """Perform majority voting for multiple runs of the same prompt."""
    df['unique_id'] = df['operation_id'].astype(str) + '_' + df['id'].astype(str)
    
    majority_results = []
    for unique_id, group in df.groupby('unique_id'):
        predictions = group['model_prediction'].tolist()
        vote_counts = {}
        for pred in predictions:
            vote_counts[pred] = vote_counts.get(pred, 0) + 1
        
        majority_pred = max(vote_counts.keys(), key=lambda x: vote_counts[x])
        confidence = vote_counts[majority_pred] / len(predictions)
        
        majority_row = group.iloc[0].copy()
        majority_row['model_prediction'] = majority_pred
        majority_row['majority_confidence'] = confidence
        majority_row['total_votes'] = len(predictions)
        
        majority_results.append(majority_row)
    
    return pd.DataFrame(majority_results)

def main():
    parser = argparse.ArgumentParser(description="Compare merge identification results across multiple files")
    parser.add_argument("files", nargs='+', help="Paths to result files to compare")
    parser.add_argument("--labels", nargs='+', help="Labels for each file (optional)")
    parser.add_argument("--output", help="Save comparison results to JSON file")
    parser.add_argument("--majority-vote", action="store_true", 
                       help="Apply majority voting to files with K repetitions")
    
    args = parser.parse_args()
    
    # Validate files exist
    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return
    
    # Run comparison
    comparison, combined_df = compare_results(args.files, args.labels)
    
    if not comparison:
        print("No valid data found in any files")
        return
    
    # Print results
    print("MERGE IDENTIFICATION RESULTS COMPARISON")
    print("=" * 50)
    
    print_comparison_table(comparison)
    
    # Best performers
    best = find_best_performers(comparison)
    print(f"\nBEST PERFORMERS:")
    for metric, source in best.items():
        print(f"  {metric.title()}: {source}")
    
    # Heuristic analysis if applicable
    if len(combined_df) > 0:
        heuristic_analysis = analyze_heuristic_impact(combined_df)
        
        if heuristic_analysis.get('heuristic_impact'):
            print(f"\nHEURISTIC IMPACT ANALYSIS:")
            hi = heuristic_analysis['heuristic_impact']
            for condition, metrics in hi.items():
                print(f"  {condition.replace('_', ' ').title()}:")
                print(f"    Samples: {metrics['total_samples']}, Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}")
        
        if heuristic_analysis.get('by_heuristic_count'):
            print(f"\nPERFORMANCE BY NUMBER OF HEURISTICS:")
            hc = heuristic_analysis['by_heuristic_count']
            for count, metrics in hc.items():
                print(f"  {count}: Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f} (n={metrics['total_samples']})")
    
    # Save results
    if args.output:
        output_data = {
            'comparison': comparison,
            'best_performers': best,
            'heuristic_analysis': analyze_heuristic_impact(combined_df) if len(combined_df) > 0 else {}
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nComparison results saved to: {args.output}")

if __name__ == "__main__":
    main()