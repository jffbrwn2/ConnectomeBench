#!/bin/bash

# Run all analysis scripts for mouse dataset
set -e  # Exit on error

# Fly
echo "Running segment classification analysis..."
python analysis/analyze_segment_classification_results.py analysis/fly_segment_classification_config.txt --output-dir ./reports/fly --categories a,b,c,d,e --labels "single neuron,multiple somas,processes,nucleus,non-neuronal"

echo "Running split comparison analysis..."
python analysis/analyze_split_comparison_results.py --config analysis/fly_split_comparison_configs.txt --output-dir ./reports/fly

echo "Running split identification analysis..."
python analysis/analyze_split_identification_results.py --config analysis/fly_split_identification_configs.txt --output-dir ./reports/fly

echo "Running merge identification analysis..."
python analysis/analyze_merge_identification_results.py --config analysis/fly_merge_identification_configs.txt --output-dir ./reports/fly

echo "Running merge comparison analysis..."
python analysis/analyze_merge_comparison_results.py --config analysis/fly_merge_comparison_configs.txt --output-dir ./reports/fly


# Mouse
echo "Running segment classification analysis..."
python analysis/analyze_segment_classification_results.py analysis/mouse_segment_classification_config.txt --output-dir ./reports/mouse --categories a,b,c,d --labels "single neuron,multiple somas,processes,nucleus"

echo "Running split comparison analysis..."
python analysis/analyze_split_comparison_results.py --config analysis/mouse_split_comparison_configs.txt --output-dir ./reports/mouse

echo "Running split identification analysis..."
python analysis/analyze_split_identification_results.py --config analysis/mouse_split_identification_configs.txt --output-dir ./reports/mouse

echo "Running merge identification analysis..."
python analysis/analyze_merge_identification_results.py --config analysis/mouse_merge_identification_configs.txt --output-dir ./reports/mouse

echo "Running merge comparison analysis..."
python analysis/analyze_merge_comparison_results.py --config analysis/mouse_merge_comparison_configs.txt --output-dir ./reports/mouse

echo "Running human annotation analysis..."
python analysis/analyze_human_annotations.py --config analysis/human_annotation_configs.txt --output-dir reports/human

echo "All analyses complete! Results saved to ./reports/mouse"