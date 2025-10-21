#!/usr/bin/env python3
"""
Lightweight script to analyze merge_identification task performance.
No external plotting dependencies required.
"""

import pandas as pd
import numpy as np
import json
import argparse
import os
from typing import Dict, List, Any

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

def calculate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate performance metrics for merge identification."""
    # Convert predictions to binary
    df_copy = df.copy()
    df_copy['model_pred_binary'] = (df_copy['model_prediction'] == '1').astype(int)
    df_copy['ground_truth_binary'] = df_copy['is_correct_merge'].astype(int)
    
    # Calculate confusion matrix values
    tp = ((df_copy['model_pred_binary'] == 1) & (df_copy['ground_truth_binary'] == 1)).sum()
    tn = ((df_copy['model_pred_binary'] == 0) & (df_copy['ground_truth_binary'] == 0)).sum()
    fp = ((df_copy['model_pred_binary'] == 1) & (df_copy['ground_truth_binary'] == 0)).sum()
    fn = ((df_copy['model_pred_binary'] == 0) & (df_copy['ground_truth_binary'] == 1)).sum()
    
    total = len(df_copy)
    
    # Calculate metrics
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'total_samples': total,
        'true_positives': int(tp),
        'true_negatives': int(tn), 
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'positive_rate': df_copy['ground_truth_binary'].mean(),
        'predicted_positive_rate': df_copy['model_pred_binary'].mean()
    }

def analyze_by_group(df: pd.DataFrame, group_col: str) -> Dict[str, Dict]:
    """Analyze metrics grouped by a column."""
    results = {}
    if group_col not in df.columns:
        return results
        
    for group_val in df[group_col].unique():
        if pd.isna(group_val):
            continue
        group_df = df[df[group_col] == group_val]
        results[str(group_val)] = calculate_metrics(group_df)
    
    return results

def extract_heuristics_from_prompt_mode(prompt_mode: str) -> List[str]:
    """Extract heuristics from prompt mode string."""
    if '+' not in prompt_mode:
        return []
    parts = prompt_mode.split('+')
    return [part for part in parts[1:] if part.startswith('heuristic')]

def analyze_heuristic_combinations(df: pd.DataFrame) -> Dict[str, Any]:
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
        by_count[f"{count}_heuristics"] = calculate_metrics(count_df)
    
    heuristic_analysis['by_heuristic_count'] = by_count
    
    # Analyze by specific combinations
    by_combination = {}
    for combination in df_copy['heuristic_str'].unique():
        combo_df = df_copy[df_copy['heuristic_str'] == combination]
        by_combination[combination] = calculate_metrics(combo_df)
    
    heuristic_analysis['by_combination'] = by_combination
    
    return heuristic_analysis

def print_metrics_table(metrics_dict: Dict[str, Dict], title: str):
    """Print metrics in a formatted table."""
    print(f"\n{title}")
    print("=" * len(title))
    
    if not metrics_dict:
        print("No data available")
        return
    
    # Headers
    headers = ['Group', 'Samples', 'Accuracy', 'Precision', 'Recall', 'F1', 'TP', 'FP', 'TN', 'FN']
    print(f"{'Group':<20} {'Samples':>8} {'Accuracy':>8} {'Precision':>9} {'Recall':>8} {'F1':>8} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}")
    print("-" * 90)
    
    for group, metrics in metrics_dict.items():
        print(f"{group:<20} {metrics['total_samples']:>8} {metrics['accuracy']:>8.3f} {metrics['precision']:>9.3f} "
              f"{metrics['recall']:>8.3f} {metrics['f1_score']:>8.3f} {metrics['true_positives']:>4} "
              f"{metrics['false_positives']:>4} {metrics['true_negatives']:>4} {metrics['false_negatives']:>4}")

def find_error_examples(df: pd.DataFrame, error_type: str, n_examples: int = 3) -> List[Dict]:
    """Find examples of specific error types."""
    df_copy = df.copy()
    df_copy['model_pred_binary'] = (df_copy['model_prediction'] == '1').astype(int)
    df_copy['ground_truth_binary'] = df_copy['is_correct_merge'].astype(int)
    
    if error_type == 'false_positive':
        error_df = df_copy[(df_copy['model_pred_binary'] == 1) & (df_copy['ground_truth_binary'] == 0)]
    elif error_type == 'false_negative':
        error_df = df_copy[(df_copy['model_pred_binary'] == 0) & (df_copy['ground_truth_binary'] == 1)]
    else:
        return []
    
    examples = []
    for _, row in error_df.head(n_examples).iterrows():
        example = {
            'operation_id': row.get('operation_id', 'N/A'),
            'segment_id': row.get('id', 'N/A'),
            'model_prediction': row.get('model_prediction', 'N/A'),
            'ground_truth': row.get('is_correct_merge', 'N/A'),
            'analysis_snippet': str(row.get('model_analysis', ''))[:100] + '...' if row.get('model_analysis') else 'N/A'
        }
        examples.append(example)
    
    return examples

def generate_report(df: pd.DataFrame, output_file: str = None):
    """Generate comprehensive performance report."""
    print("MERGE IDENTIFICATION PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Overall metrics
    overall_metrics = calculate_metrics(df)
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Total Samples: {overall_metrics['total_samples']}")
    print(f"  Accuracy: {overall_metrics['accuracy']:.3f}")
    print(f"  Precision: {overall_metrics['precision']:.3f}")
    print(f"  Recall: {overall_metrics['recall']:.3f}")
    print(f"  F1-Score: {overall_metrics['f1_score']:.3f}")
    print(f"  Specificity: {overall_metrics['specificity']:.3f}")
    
    print(f"\nCONFUSION MATRIX:")
    print(f"  True Positives (Correct Merge Identified): {overall_metrics['true_positives']}")
    print(f"  False Positives (Incorrect Merge Identified): {overall_metrics['false_positives']}")
    print(f"  True Negatives (Correct No-Merge Identified): {overall_metrics['true_negatives']}")
    print(f"  False Negatives (Missed Merge): {overall_metrics['false_negatives']}")
    
    print(f"\nCLASS DISTRIBUTION:")
    print(f"  Should Merge (Positive): {int(overall_metrics['positive_rate'] * overall_metrics['total_samples'])} ({overall_metrics['positive_rate']:.1%})")
    print(f"  Should Not Merge (Negative): {int((1-overall_metrics['positive_rate']) * overall_metrics['total_samples'])} ({1-overall_metrics['positive_rate']:.1%})")
    print(f"  Model Predicted Merge: {int(overall_metrics['predicted_positive_rate'] * overall_metrics['total_samples'])} ({overall_metrics['predicted_positive_rate']:.1%})")
    
    # Analysis by groups
    if 'model' in df.columns:
        model_metrics = analyze_by_group(df, 'model')
        print_metrics_table(model_metrics, "PERFORMANCE BY MODEL")
    
    if 'prompt_mode' in df.columns:
        prompt_metrics = analyze_by_group(df, 'prompt_mode')  
        print_metrics_table(prompt_metrics, "PERFORMANCE BY PROMPT MODE")
        
        # Heuristic analysis
        heuristic_analysis = analyze_heuristic_combinations(df)
        if heuristic_analysis:
            print_metrics_table(heuristic_analysis['by_heuristic_count'], "PERFORMANCE BY NUMBER OF HEURISTICS")
            print_metrics_table(heuristic_analysis['by_combination'], "PERFORMANCE BY HEURISTIC COMBINATION")
    
    # Error examples
    print(f"\nERROR EXAMPLES:")
    fp_examples = find_error_examples(df, 'false_positive', 3)
    fn_examples = find_error_examples(df, 'false_negative', 3)
    
    if fp_examples:
        print(f"\nFalse Positives (Model said merge, but shouldn't):")
        for i, ex in enumerate(fp_examples, 1):
            print(f"  {i}. Op: {ex['operation_id']}, Segment: {ex['segment_id']}")
            print(f"     Analysis: {ex['analysis_snippet']}")
    
    if fn_examples:
        print(f"\nFalse Negatives (Model said no merge, but should merge):")
        for i, ex in enumerate(fn_examples, 1):
            print(f"  {i}. Op: {ex['operation_id']}, Segment: {ex['segment_id']}")
            print(f"     Analysis: {ex['analysis_snippet']}")
    
    # Save detailed report
    if output_file:
        report_data = {
            'overall_metrics': overall_metrics,
            'by_model': analyze_by_group(df, 'model') if 'model' in df.columns else {},
            'by_prompt_mode': analyze_by_group(df, 'prompt_mode') if 'prompt_mode' in df.columns else {},
            'heuristic_analysis': analyze_heuristic_combinations(df),
            'error_examples': {
                'false_positives': fp_examples,
                'false_negatives': fn_examples
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        print(f"\nDetailed report saved to: {output_file}")

def perform_majority_voting(df: pd.DataFrame) -> pd.DataFrame:
    """Perform majority voting for multiple runs of the same prompt."""
    # Create unique identifier for each prompt instance
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
        
        majority_results.append(majority_row)
    
    return pd.DataFrame(majority_results)

def main():
    parser = argparse.ArgumentParser(description="Analyze merge identification performance")
    parser.add_argument("results_file", help="Path to results CSV or JSON file")
    parser.add_argument("--output", help="Path to save detailed JSON report", default=None)
    parser.add_argument("--majority-vote", action="store_true", 
                       help="Perform majority voting for multiple runs (K repetitions)")
    
    args = parser.parse_args()
    
    # Load and validate data
    print(f"Loading results from: {args.results_file}")
    df = load_results(args.results_file)
    
    required_cols = ['is_correct_merge', 'model_prediction']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    print(f"Loaded {len(df)} result rows")
    
    # Filter for merge identification task if mixed data
    if 'task' in df.columns:
        merge_df = df[df['task'] == 'merge_identification']
        if len(merge_df) < len(df):
            print(f"Filtered to {len(merge_df)} merge_identification rows")
            df = merge_df
    
    if len(df) == 0:
        print("No merge identification data found")
        return
    
    # Perform majority voting if requested
    if args.majority_vote:
        print(f"Performing majority voting on {len(df)} individual predictions...")
        original_count = len(df)
        df = perform_majority_voting(df)
        
        print(f"Reduced to {len(df)} majority votes (average confidence: {df['majority_confidence'].mean():.3f})")
        
        # Show confidence distribution
        unanimous = (df['majority_confidence'] == 1.0).sum()
        split = (df['majority_confidence'] < 1.0).sum()
        print(f"Unanimous decisions: {unanimous}/{len(df)} ({unanimous/len(df):.1%})")
        print(f"Split decisions: {split}/{len(df)} ({split/len(df):.1%})")
    
    # Generate report
    generate_report(df, args.output)

if __name__ == "__main__":
    main()