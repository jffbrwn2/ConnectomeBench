#!/usr/bin/env python3
"""
Script to visualize merge identification errors with images and model analysis.
Shows front, side, top views for cases where the model prediction was incorrect.
"""

import pandas as pd
import numpy as np
import json
import argparse
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import textwrap

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

def get_error_cases(df: pd.DataFrame, error_type: str = 'all') -> pd.DataFrame:
    """
    Filter for incorrect predictions.
    
    Args:
        df: DataFrame with results
        error_type: 'false_positive', 'false_negative', or 'all'
    """
    # Convert predictions to binary
    df_copy = df.copy()
    df_copy['model_pred_binary'] = (df_copy['model_prediction'] == '1').astype(int)
    df_copy['ground_truth_binary'] = df_copy['is_correct_merge'].astype(int)
    df_copy['is_correct'] = (df_copy['model_pred_binary'] == df_copy['ground_truth_binary'])
    
    # Filter for errors
    error_df = df_copy[~df_copy['is_correct']]
    
    if error_type == 'false_positive':
        # Model said merge, but shouldn't merge
        error_df = error_df[(error_df['model_pred_binary'] == 1) & (error_df['ground_truth_binary'] == 0)]
        error_df['error_type'] = 'False Positive (Said Merge, Should Not)'
    elif error_type == 'false_negative':
        # Model said no merge, but should merge
        error_df = error_df[(error_df['model_pred_binary'] == 0) & (error_df['ground_truth_binary'] == 1)]
        error_df['error_type'] = 'False Negative (Said No Merge, Should Merge)'
    else:
        # All errors
        fp_mask = (error_df['model_pred_binary'] == 1) & (error_df['ground_truth_binary'] == 0)
        fn_mask = (error_df['model_pred_binary'] == 0) & (error_df['ground_truth_binary'] == 1)
        error_df.loc[fp_mask, 'error_type'] = 'False Positive (Said Merge, Should Not)'
        error_df.loc[fn_mask, 'error_type'] = 'False Negative (Said No Merge, Should Merge)'
    
    return error_df

def extract_image_paths(row: pd.Series, image_type: str = 'zoomed') -> Dict[str, str]:
    """
    Extract image paths from the prompt_options field.
    
    Args:
        row: DataFrame row
        image_type: 'zoomed' or 'default'
    """
    paths = {}
    
    try:
        # Try to get paths from prompt_options if available
        if 'prompt_options' in row and pd.notna(row['prompt_options']):
            prompt_options = row['prompt_options']
            if isinstance(prompt_options, str):
                # If it's a JSON string, parse it
                try:
                    prompt_options = json.loads(prompt_options.replace("'", '"'))
                except:
                    return paths
            
            # Extract paths for this segment ID
            segment_id = str(row['id'])
            
            if isinstance(prompt_options, list):
                # Find the option matching this segment
                for option in prompt_options:
                    if str(option.get('id', '')) == segment_id:
                        option_paths = option.get('paths', {})
                        if image_type in option_paths:
                            paths = option_paths[image_type]
                        break
            elif isinstance(prompt_options, dict):
                # Direct path dictionary
                if image_type in prompt_options:
                    paths = prompt_options[image_type]
        
        # Alternative: try to construct paths from operation info
        if not paths and 'operation_id' in row and 'merge_coords' in row:
            # Try to reconstruct likely paths based on file naming patterns
            coords = row.get('merge_coords', [])
            if isinstance(coords, str):
                try:
                    coords = json.loads(coords.replace("'", '"'))
                except:
                    coords = []
            
            if len(coords) >= 3:
                coords_suffix = f"{int(coords[0])}_{int(coords[1])}_{int(coords[2])}"
                base_dir = f"merge_{row['operation_id']}_{coords_suffix}"
                
                # Try common path patterns
                for view in ['front', 'side', 'top']:
                    potential_path = f"output/*/merge_{row['operation_id']}_{coords_suffix}/option_{segment_id}_with_base_{image_type}_{view}.png"
                    # This would need globbing to find actual path
                    paths[view] = potential_path
        
    except Exception as e:
        print(f"Error extracting paths for row {row.name}: {e}")
    
    return paths

def load_image_safely(image_path: str) -> Optional[np.ndarray]:
    """Load image with error handling."""
    try:
        if os.path.exists(image_path):
            img = Image.open(image_path)
            return np.array(img)
        else:
            print(f"Image not found: {image_path}")
            return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def create_error_visualization(row: pd.Series, image_type: str = 'zoomed', save_path: str = None) -> plt.Figure:
    """
    Create visualization for a single error case.
    
    Args:
        row: DataFrame row with error case
        image_type: 'zoomed' or 'default'
        save_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Extract information
    operation_id = row.get('operation_id', 'N/A')
    segment_id = row.get('id', 'N/A')
    error_type = row.get('error_type', 'Unknown Error')
    model_prediction = row.get('model_prediction', 'N/A')
    ground_truth = row.get('is_correct_merge', 'N/A')
    model_analysis = row.get('model_analysis', 'No analysis available')
    prompt_mode = row.get('prompt_mode', 'N/A')
    model_name = row.get('model', 'N/A')
    
    # Create main title
    fig.suptitle(f'{error_type}\nOperation: {operation_id}, Segment: {segment_id}', 
                fontsize=14, fontweight='bold', y=0.95)
    
    # Extract image paths
    image_paths = extract_image_paths(row, image_type)
    
    # Image display area (top 60% of figure)
    views = ['front', 'side', 'top']
    image_axes = []
    
    for i, view in enumerate(views):
        ax = fig.add_subplot(2, 3, i + 1)
        image_axes.append(ax)
        
        if view in image_paths:
            img = load_image_safely(image_paths[view])
            if img is not None:
                ax.imshow(img)
                ax.set_title(f'{view.title()} View', fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f'{view.title()} View\n(Image not found)', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        else:
            ax.text(0.5, 0.5, f'{view.title()} View\n(Path not available)', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Info and analysis area (bottom 40% of figure)
    info_ax = fig.add_subplot(2, 1, 2)
    info_ax.axis('off')
    
    # Create info text
    info_text = f"""PREDICTION DETAILS:
Model: {model_name} | Prompt Mode: {prompt_mode}
Model Prediction: {'Merge' if model_prediction == '1' else 'No Merge'} | Ground Truth: {'Should Merge' if ground_truth else 'Should Not Merge'}

MODEL ANALYSIS:
{textwrap.fill(str(model_analysis), width=120)}
    """
    
    # Color code the error type
    error_color = 'red' if 'False Positive' in error_type else 'orange'
    
    info_ax.text(0.05, 0.95, info_text, transform=info_ax.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=error_color, linewidth=2))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    return fig

def create_error_summary(error_df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Create summary visualization of all errors."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Merge Identification Error Summary', fontsize=16, fontweight='bold')
    
    # Error type distribution
    ax1 = axes[0, 0]
    error_counts = error_df['error_type'].value_counts()
    colors = ['red', 'orange']
    ax1.pie(error_counts.values, labels=error_counts.index, autopct='%1.1f%%', 
           colors=colors[:len(error_counts)], startangle=90)
    ax1.set_title('Error Type Distribution')
    
    # Errors by prompt mode (if available)
    ax2 = axes[0, 1]
    if 'prompt_mode' in error_df.columns:
        prompt_counts = error_df['prompt_mode'].value_counts()
        ax2.bar(range(len(prompt_counts)), prompt_counts.values)
        ax2.set_xticks(range(len(prompt_counts)))
        ax2.set_xticklabels(prompt_counts.index, rotation=45, ha='right')
        ax2.set_title('Errors by Prompt Mode')
        ax2.set_ylabel('Number of Errors')
    else:
        ax2.text(0.5, 0.5, 'Prompt mode data\nnot available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Errors by Prompt Mode')
    
    # Errors by model (if available)
    ax3 = axes[1, 0]
    if 'model' in error_df.columns:
        model_counts = error_df['model'].value_counts()
        ax3.bar(range(len(model_counts)), model_counts.values)
        ax3.set_xticks(range(len(model_counts)))
        ax3.set_xticklabels(model_counts.index, rotation=45, ha='right')
        ax3.set_title('Errors by Model')
        ax3.set_ylabel('Number of Errors')
    else:
        ax3.text(0.5, 0.5, 'Model data\nnot available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Errors by Model')
    
    # Error rate by ground truth class
    ax4 = axes[1, 1]
    error_by_truth = error_df.groupby(['is_correct_merge', 'error_type']).size().unstack(fill_value=0)
    error_by_truth.plot(kind='bar', ax=ax4, color=['red', 'orange'])
    ax4.set_title('Errors by Ground Truth Class')
    ax4.set_xlabel('Ground Truth (Should Merge)')
    ax4.set_ylabel('Number of Errors')
    ax4.legend(title='Error Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.set_xticklabels(['Should Not Merge', 'Should Merge'], rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved error summary to: {save_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Visualize merge identification errors with images")
    parser.add_argument("results_file", help="Path to results CSV or JSON file")
    parser.add_argument("--error-type", choices=['all', 'false_positive', 'false_negative'], 
                       default='all', help="Type of errors to visualize")
    parser.add_argument("--max-examples", type=int, default=10, 
                       help="Maximum number of error examples to visualize")
    parser.add_argument("--image-type", choices=['zoomed', 'default'], default='zoomed',
                       help="Type of images to display")
    parser.add_argument("--output-dir", help="Directory to save visualizations")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--summary-only", action="store_true", help="Only create summary plot")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_file}")
    df = load_results(args.results_file)
    
    # Validate merge identification data
    required_cols = ['is_correct_merge', 'model_prediction']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return
    
    # Filter for merge identification task
    if 'task' in df.columns:
        df = df[df['task'] == 'merge_identification']
    
    if len(df) == 0:
        print("No merge identification data found")
        return
    
    # Get error cases
    error_df = get_error_cases(df, args.error_type)
    
    if len(error_df) == 0:
        print(f"No {args.error_type} errors found")
        return
    
    print(f"Found {len(error_df)} error cases")
    
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Create summary plot
    summary_path = os.path.join(args.output_dir, 'error_summary.png') if args.output_dir else None
    summary_fig = create_error_summary(error_df, summary_path)
    
    if args.show:
        plt.show()
    
    if not args.summary_only:
        # Create individual error visualizations
        n_examples = min(args.max_examples, len(error_df))
        print(f"Creating visualizations for {n_examples} error examples...")
        
        for i, (idx, row) in enumerate(error_df.head(n_examples).iterrows()):
            print(f"Processing example {i+1}/{n_examples}: Op {row.get('operation_id', 'N/A')}, Segment {row.get('id', 'N/A')}")
            
            save_path = None
            if args.output_dir:
                safe_op_id = str(row.get('operation_id', 'unknown')).replace('/', '_')
                safe_seg_id = str(row.get('id', 'unknown')).replace('/', '_')
                filename = f"error_{i+1:02d}_op_{safe_op_id}_seg_{safe_seg_id}.png"
                save_path = os.path.join(args.output_dir, filename)
            
            try:
                fig = create_error_visualization(row, args.image_type, save_path)
                
                if args.show:
                    plt.show()
                else:
                    plt.close(fig)  # Close to save memory
                    
            except Exception as e:
                print(f"Error creating visualization for example {i+1}: {e}")
                continue
    
    print(f"\nVisualization complete!")
    if args.output_dir:
        print(f"Files saved to: {args.output_dir}")
    
    # Print summary statistics
    fp_count = len(error_df[error_df['error_type'].str.contains('False Positive', na=False)])
    fn_count = len(error_df[error_df['error_type'].str.contains('False Negative', na=False)])
    
    print(f"\nError Summary:")
    print(f"  False Positives (said merge, shouldn't): {fp_count}")
    print(f"  False Negatives (said no merge, should): {fn_count}")
    print(f"  Total errors visualized: {len(error_df)}")

if __name__ == "__main__":
    main()