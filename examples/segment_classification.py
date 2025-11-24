"""
Tutorial: Segment Classification Task

This script demonstrates how to run the segment classification task using the
ConnectomeBench dataset from HuggingFace.

Task: Given images of a neuron segment, classify it as:
- "undersegmented" - segment is too small, needs merging
- "oversegmented" - segment is too large, needs splitting
- "correct" - segment is properly segmented (a single soma and process(es))

Usage:
    python segment_classification.py --num-samples 10 --model gpt-4o
"""

import os
import sys
import argparse
import asyncio
import pandas as pd
import tempfile
from datasets import load_dataset

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from prompts import create_segment_classification_prompt
from util import LLMProcessor

# Mapping from LLM answer keys to full descriptions in the dataset
CLASS_MAPPING = {
    "a": "a single soma and process(es)",
    "b": "multiple somas (and processes)",
    "c": "Processes without a soma: These can be axons, dendrites, synapses",
    "d": "Nucleus",
    "e": "Non-neuronal types. These can be glial cells, blood vessels.",
    "f": "None of the above.",
    "g": "Unsure"
}


async def run_segment_classification(
    num_samples: int = None,
    model: str = "gpt-4o",
    with_description: bool = True,
    output_dir: str = "output/tutorial_results"
):
    """
    Run segment classification task on ConnectomeBench dataset.

    Args:
        num_samples: Number of samples to evaluate (None = all)
        model: LLM model name to use
        with_description: Include descriptive prompt text
        output_dir: Directory to save results
    """
    print("="*60)
    print("ConnectomeBench Tutorial: Segment Classification")
    print("="*60)

    # Load dataset from HuggingFace
    print("\nLoading dataset from HuggingFace...")
    try:
        ds = load_dataset("jeffbbrown2/ConnectomeBench", "MICrONS, Segment Classification", split="train")
        print(f"Loaded {len(ds)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have authenticated with: huggingface-cli login")
        return

    # Limit number of samples if specified
    if num_samples is not None and num_samples < len(ds):
        ds = ds.select(range(num_samples))
        print(f"Using {num_samples} samples")

    # Initialize LLM processor
    print(f"\nInitializing LLM processor with model: {model}")
    llm_processor = LLMProcessor(model=model, max_tokens=4096, max_concurrent=10)

    # Create temporary directory for images
    temp_dir = tempfile.mkdtemp()
    print(f"Temporary directory for images: {temp_dir}")

    # Create prompts for all samples
    print("\nCreating prompts...")
    prompts = []
    temp_image_paths = []

    for idx, sample in enumerate(ds):
        # Save PIL images to temporary files
        sample_temp_dir = os.path.join(temp_dir, f"sample_{idx}")
        os.makedirs(sample_temp_dir, exist_ok=True)

        front_path = os.path.join(sample_temp_dir, "front.png")
        side_path = os.path.join(sample_temp_dir, "side.png")
        top_path = os.path.join(sample_temp_dir, "top.png")

        sample['option_1_front_image'].save(front_path)
        sample['option_1_side_image'].save(side_path)
        sample['option_1_top_image'].save(top_path)

        segment_images_paths = [front_path, side_path, top_path]
        temp_image_paths.append(segment_images_paths)

        species = sample['species']

        # Get bounding box coordinates (same as original)
        minpt = [sample['xmin'], sample['ymin'], sample['zmin']]
        maxpt = [sample['xmax'], sample['ymax'], sample['zmax']]

        # Create prompt (same as original script)
        segment_classification_prompt = create_segment_classification_prompt(
            segment_images_paths,
            minpt,
            maxpt,
            llm_processor,
            species,
            with_description
        )
        prompts.append(segment_classification_prompt)

    print(f"Created {len(prompts)} prompts")

    # Run LLM evaluation
    print("\nRunning LLM evaluation...")
    llm_responses = await llm_processor.process_batch(prompts)

    # Process results (same as original script)
    print("\nProcessing results...")
    results = []
    for sample, response in zip(ds, llm_responses):
        # Extract Analysis (same as original)
        analysis_start = response.find("<analysis>")
        analysis_end = response.find("</analysis>")
        if analysis_start != -1 and analysis_end != -1:
            analysis = response[analysis_start + len("<analysis>"):analysis_end].strip()
        else:
            print("Warning: Could not find <analysis> tags in the model response.")
            analysis = "Analysis tags not found in response."

        # Extract Answer (same as original)
        answer_start = response.find("<answer>")
        answer_end = response.find("</answer>")
        llm_answer = response[answer_start + len("<answer>"):answer_end].strip()

        # Get ground truth description from dataset
        ground_truth = sample.get('class_description', sample.get('ground_truth', None))

        # Map the LLM answer key to full description
        predicted_description = CLASS_MAPPING.get(llm_answer, None)

        # Determine correctness by comparing descriptions
        correct = None
        if predicted_description and ground_truth:
            correct = (predicted_description == ground_truth)

        # Create result entry
        result = {
            'proofread_root_id': sample.get('proofread_root_id'),
            'current_root_id': sample.get('current_root_id'),
            'option_1_neuron_id': sample.get('option_1_neuron_id'),
            'species': sample.get('species'),
            'model': model,
            'llm_answer': llm_answer,
            'predicted_description': predicted_description,
            'ground_truth': ground_truth,
            'correct': correct,
            'analysis': analysis,
            'full_response': response,
            'with_description': with_description
        }
        results.append(result)

    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary directory")

    # Create DataFrame and save
    results_df = pd.DataFrame(results)

    # Calculate accuracy if ground truth available
    if 'correct' in results_df.columns and results_df['correct'].notna().any():
        accuracy = results_df['correct'].mean()
        print(f"\nAccuracy: {accuracy:.2%}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/segment_classification_{model.replace('/', '_')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\nSummary of predictions:")
    print(results_df['llm_answer'].value_counts())

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Tutorial: Run segment classification on ConnectomeBench dataset"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model to use (default: gpt-4o)"
    )
    parser.add_argument(
        "--with-description",
        action="store_true",
        default=True,
        help="Include descriptive prompt text (default: True)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/tutorial_results",
        help="Directory to save results (default: output/tutorial_results)"
    )

    args = parser.parse_args()

    # Run evaluation
    asyncio.run(run_segment_classification(
        num_samples=args.num_samples,
        model=args.model,
        with_description=args.with_description,
        output_dir=args.output_dir
    ))


if __name__ == "__main__":
    main()
