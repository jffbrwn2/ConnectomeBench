#!/usr/bin/env python3
"""
HTML-based visualization of merge identification errors.
Creates an interactive HTML report showing images and model analysis.
"""

import pandas as pd
import json
import argparse
import os
import base64
from typing import Dict, List, Optional
import glob

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
    """Filter for incorrect predictions."""
    df_copy = df.copy()
    df_copy['model_pred_binary'] = (df_copy['model_prediction'] == '1').astype(int)
    df_copy['ground_truth_binary'] = df_copy['is_correct_merge'].astype(int)
    df_copy['is_correct'] = (df_copy['model_pred_binary'] == df_copy['ground_truth_binary'])
    
    # Filter for errors
    error_df = df_copy[~df_copy['is_correct']]
    
    if error_type == 'false_positive':
        error_df = error_df[(error_df['model_pred_binary'] == 1) & (error_df['ground_truth_binary'] == 0)]
        error_df['error_type'] = 'False Positive'
        error_df['error_description'] = 'Model said MERGE, but should NOT merge'
    elif error_type == 'false_negative':
        error_df = error_df[(error_df['model_pred_binary'] == 0) & (error_df['ground_truth_binary'] == 1)]
        error_df['error_type'] = 'False Negative'
        error_df['error_description'] = 'Model said NO MERGE, but should merge'
    else:
        fp_mask = (error_df['model_pred_binary'] == 1) & (error_df['ground_truth_binary'] == 0)
        fn_mask = (error_df['model_pred_binary'] == 0) & (error_df['ground_truth_binary'] == 1)
        error_df.loc[fp_mask, 'error_type'] = 'False Positive'
        error_df.loc[fp_mask, 'error_description'] = 'Model said MERGE, but should NOT merge'
        error_df.loc[fn_mask, 'error_type'] = 'False Negative'
        error_df.loc[fn_mask, 'error_description'] = 'Model said NO MERGE, but should merge'
    
    return error_df

def find_image_paths(operation_id: str, segment_id: str, coords: List, image_type: str = 'zoomed') -> Dict[str, str]:
    """
    Try to find image paths using glob patterns.
    """
    paths = {}
    
    try:
        # Convert coords to string format used in filenames
        if len(coords) >= 3:
            coords_suffix = f"{int(coords[0])}_{int(coords[1])}_{int(coords[2])}"
        else:
            return paths
        
        # Try different possible path patterns
        base_patterns = [
            f"output/*/merge_{operation_id}_{coords_suffix}/option_{segment_id}_with_base_{image_type}_*.png",
            f"merge_{operation_id}_{coords_suffix}/option_{segment_id}_with_base_{image_type}_*.png",
            f"**/merge_{operation_id}_{coords_suffix}/option_{segment_id}_with_base_{image_type}_*.png"
        ]

        for pattern in base_patterns:
            matching_files = glob.glob(pattern, recursive=True)
            
            for file_path in matching_files:
                filename = os.path.basename(file_path)
                
                # Extract view from filename
                if '_front.png' in filename:
                    paths['front'] = file_path
                elif '_side.png' in filename:
                    paths['side'] = file_path
                elif '_top.png' in filename:
                    paths['top'] = file_path
            
            if paths:  # If we found any images, stop looking
                break
    
    except Exception as e:
        print(f"Error finding images for operation {operation_id}, segment {segment_id}: {e}")
    
    return paths

def image_to_base64(image_path: str) -> Optional[str]:
    """Convert image to base64 for HTML embedding."""
    try:
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                img_data = f.read()
            return base64.b64encode(img_data).decode('utf-8')
        else:
            return None
    except Exception as e:
        print(f"Error converting image {image_path} to base64: {e}")
        return None

def create_html_report(error_df: pd.DataFrame, output_path: str, max_examples: int = 20, image_type: str = 'zoomed'):
    """Create HTML report with error visualizations."""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Merge Identification Error Analysis</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
            }}
            .summary {{
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .error-case {{
                background-color: white;
                margin-bottom: 30px;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .error-header {{
                padding: 15px;
                color: white;
                font-weight: bold;
            }}
            .false-positive {{
                background-color: #e74c3c;
            }}
            .false-negative {{
                background-color: #f39c12;
            }}
            .error-content {{
                padding: 20px;
            }}
            .images-container {{
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
                flex-wrap: wrap;
            }}
            .image-box {{
                flex: 1;
                min-width: 200px;
                text-align: center;
            }}
            .image-box img {{
                max-width: 100%;
                height: auto;
                border: 2px solid #bdc3c7;
                border-radius: 4px;
            }}
            .image-title {{
                font-weight: bold;
                margin-bottom: 5px;
                color: #2c3e50;
            }}
            .no-image {{
                background-color: #ecf0f1;
                border: 2px dashed #bdc3c7;
                padding: 40px;
                color: #7f8c8d;
                border-radius: 4px;
            }}
            .analysis-section {{
                background-color: #f8f9fa;
                padding: 15px;
                border-left: 4px solid #3498db;
                border-radius: 4px;
            }}
            .info-table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 15px;
            }}
            .info-table th, .info-table td {{
                padding: 8px 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .info-table th {{
                background-color: #f1f2f6;
                font-weight: bold;
            }}
            .model-analysis {{
                background-color: white;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                white-space: pre-wrap;
                max-height: 200px;
                overflow-y: auto;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 Merge Identification Error Analysis</h1>
            <p>Visual analysis of incorrect predictions with images and model reasoning</p>
        </div>
        
        <div class="summary">
            <h2>📊 Summary Statistics</h2>
            <p><strong>Total Errors:</strong> {len(error_df)}</p>
            <p><strong>False Positives:</strong> {len(error_df[error_df['error_type'] == 'False Positive'])} (Model said merge, but shouldn't)</p>
            <p><strong>False Negatives:</strong> {len(error_df[error_df['error_type'] == 'False Negative'])} (Model said no merge, but should)</p>
            <p><strong>Showing:</strong> First {min(max_examples, len(error_df))} examples</p>
        </div>
    """
    
    # Process each error case
    for i, (idx, row) in enumerate(error_df.head(max_examples).iterrows()):
        operation_id = row.get('operation_id', 'N/A')
        segment_id = row.get('id', 'N/A')
        error_type = row.get('error_type', 'Unknown')
        error_description = row.get('error_description', '')
        model_prediction = row.get('model_prediction', 'N/A')
        ground_truth = row.get('is_correct_merge', 'N/A')
        model_analysis = row.get('model_analysis', 'No analysis available')
        prompt_mode = row.get('prompt_mode', 'N/A')
        model_name = row.get('model', 'N/A')
        
        # Try to find images
        coords = row.get('merge_coords', [])
        if isinstance(coords, str):
            try:
                coords = json.loads(coords.replace("'", '"'))
            except:
                coords = []
        
        # image_paths = find_image_paths(operation_id, segment_id, coords, image_type)
        image_paths = eval(row['image_paths'])['options'][str(segment_id)]['zoomed']

        # Create error case HTML
        error_class = 'false-positive' if error_type == 'False Positive' else 'false-negative'
        
        html_content += f"""
        <div class="error-case">
            <div class="error-header {error_class}">
                🚨 Error Case #{i+1}: {error_type} - {error_description}
            </div>
            <div class="error-content">
                <table class="info-table">
                    <tr><th>Operation ID</th><td>{operation_id}</td></tr>
                    <tr><th>Segment ID</th><td>{segment_id}</td></tr>
                    <tr><th>Model</th><td>{model_name}</td></tr>
                    <tr><th>Prompt Mode</th><td>{prompt_mode}</td></tr>
                    <tr><th>Model Prediction</th><td>{'🔗 Merge' if model_prediction == '1' else '❌ No Merge'}</td></tr>
                    <tr><th>Ground Truth</th><td>{'🔗 Should Merge' if ground_truth else '❌ Should Not Merge'}</td></tr>
                </table>
                
                <div class="images-container">
        """
        
        # Add images
        views = ['front', 'side', 'top']
        for view in views:
            html_content += f'<div class="image-box"><div class="image-title">{view.title()} View</div>'
            
            if view in image_paths:
                img_base64 = image_to_base64(image_paths[view])
                if img_base64:
                    html_content += f'<img src="data:image/png;base64,{img_base64}" alt="{view} view">'
                else:
                    html_content += f'<div class="no-image">Image file not found<br>{view} view</div>'
            else:
                html_content += f'<div class="no-image">Image path not available<br>{view} view</div>'
            
            html_content += '</div>'
        
        html_content += f"""
                </div>
                
                <div class="analysis-section">
                    <h3>🤖 Model Analysis</h3>
                    <div class="model-analysis">{model_analysis}</div>
                </div>
            </div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report created: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create HTML visualization of merge identification errors")
    parser.add_argument("results_file", help="Path to results CSV or JSON file")
    parser.add_argument("--error-type", choices=['all', 'false_positive', 'false_negative'], 
                       default='all', help="Type of errors to visualize")
    parser.add_argument("--max-examples", type=int, default=20, 
                       help="Maximum number of error examples to include")
    parser.add_argument("--image-type", choices=['zoomed', 'default'], default='zoomed',
                       help="Type of images to display")
    parser.add_argument("--output", help="Output HTML file path", 
                       default="merge_identification_errors.html")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_file}")
    df = load_results(args.results_file)
    
    # Validate data
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
    
    # Create HTML report
    create_html_report(error_df, args.output, args.max_examples, args.image_type)
    
    print(f"✅ HTML report created: {args.output}")
    print(f"📊 Included {min(args.max_examples, len(error_df))} error examples")

if __name__ == "__main__":
    main()