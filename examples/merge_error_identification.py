"""
Tutorial: Merge Error Identification Task

This script demonstrates how to run the merge error identification task using the
ConnectomeBench dataset from HuggingFace.

Task: Given a base neuron and multiple candidate neurons shown with the base,
identify which candidate should be merged with the base neuron.

Usage:
    python merge_error_identification.py --num-samples 10 --model gpt-4o
"""

import os
import sys
import argparse
import asyncio
import pandas as pd
import tempfile
import shutil
from datasets import load_dataset

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from prompts import create_merge_comparison_prompt
from util import LLMProcessor, evaluate_response


async def run_merge_error_identification(
    num_samples: int = None,
    model: str = "gpt-4o",
    prompt_mode: str = 'informative',
    output_dir: str = "output/tutorial_results"
):
    """
    Run merge error identification task on ConnectomeBench dataset.

    Args:
        num_samples: Number of samples to evaluate (None = all)
        model: LLM model name to use
        prompt_mode: Prompt mode ('informative' or 'minimal')
        output_dir: Directory to save results
    """
    print("="*60)
    print("ConnectomeBench Tutorial: Merge Error Identification")
    print("="*60)

    # Load dataset from HuggingFace
    print("\nLoading dataset from HuggingFace...")
    try:
        ds = load_dataset("jeffbbrown2/ConnectomeBench", "MICrONS, Merge Error Identification", split="train")
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

    for idx, sample in enumerate(ds):
        # Save PIL images to temporary files
        sample_temp_dir = os.path.join(temp_dir, f"sample_{idx}")
        os.makedirs(sample_temp_dir, exist_ok=True)

        # Build prompt options from the sample
        prompt_options = []

        # Determine number of options (check for option_1, option_2, etc.)
        option_num = 1
        while f'option_{option_num}_neuron_id' in sample:
            # Save images for this option
            front_path = os.path.join(sample_temp_dir, f"option_{option_num}_front.png")
            side_path = os.path.join(sample_temp_dir, f"option_{option_num}_side.png")
            top_path = os.path.join(sample_temp_dir, f"option_{option_num}_top.png")

            sample[f'option_{option_num}_front_path'].save(front_path)
            sample[f'option_{option_num}_side_path'].save(side_path)
            sample[f'option_{option_num}_top_path'].save(top_path)

            option_data = {
                'id': sample[f'option_{option_num}_neuron_id'],
                'paths': {
                    'zoomed': {
                        'front': front_path,
                        'side': side_path,
                        'top': top_path
                    }
                }
            }
            prompt_options.append(option_data)
            option_num += 1

        # Create prompt (same as original script)
        prompt = create_merge_comparison_prompt(
            prompt_options,
            use_zoomed_images=True,
            views=['front', 'side', 'top'],
            llm_processor=llm_processor,
            zoom_margin=1024,  # Default zoom margin
            prompt_mode=prompt_mode
        )
        prompts.append(prompt)

    print(f"Created {len(prompts)} prompts")

    # Run LLM evaluation
    print("\nRunning LLM evaluation...")
    llm_responses = await llm_processor.process_batch(prompts)

    # Process results (same as original merge_resolution.py)
    print("\nProcessing results...")
    results = []
    for sample, response in zip(ds, llm_responses):
        # Parse LLM response using evaluate_response (same as original)
        answer_analysis = evaluate_response(response)

        # Get ground truth (the neuron_id that should be merged)
        ground_truth_id = sample.get('ground_truth', None)

        # Build option_id list and index mapping (1-indexed)
        option_ids = []
        option_index_to_id = {}
        option_num = 1
        while f'option_{option_num}_neuron_id' in sample:
            neuron_id = sample[f'option_{option_num}_neuron_id']
            option_ids.append(neuron_id)
            option_index_to_id[option_num] = neuron_id  # 1-indexed
            option_num += 1

        # Determine model chosen ID (same logic as original)
        model_chosen_id = "none"
        error = None
        predicted_index = answer_analysis.get("answer", "none")

        if predicted_index != "none":
            try:
                answer_int = int(predicted_index)
                if answer_int in option_index_to_id:
                    model_chosen_id = str(option_index_to_id[answer_int])
                else:
                    model_chosen_id = "none"
                    error = "Model returned index out of bounds"
            except (ValueError, KeyError):
                model_chosen_id = "none"
                error = "Could not parse model answer"

        # Determine correctness
        correct = None
        if ground_truth_id and model_chosen_id != "none":
            correct = (model_chosen_id == ground_truth_id)

        # Create result entry
        result = {
            'operation_id': sample.get('operation_id'),
            'base_neuron_id': sample.get('base_neuron_id'),
            'species': sample.get('species'),
            'coords': sample.get('coords'),
            'option_ids': option_ids,
            'model': model,
            'model_answer': predicted_index,
            'model_chosen_id': model_chosen_id,
            'model_analysis': answer_analysis.get('analysis', ''),
            'ground_truth_id': ground_truth_id,
            'one_valid_option': sample.get('one_valid_option'),
            'correct': correct,
            'error': error,
            'full_response': response,
            'prompt_mode': prompt_mode
        }
        results.append(result)

    # Clean up temporary directory
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
    output_file = f"{output_dir}/merge_error_identification_{model.replace('/', '_')}_{prompt_mode}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\nSummary of predictions:")
    print(results_df['model_answer'].value_counts())

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Tutorial: Run merge error identification on ConnectomeBench dataset"
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
        "--prompt-mode",
        type=str,
        default="informative",
        choices=['informative', 'minimal'],
        help="Prompt mode (default: informative)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/tutorial_results",
        help="Directory to save results (default: output/tutorial_results)"
    )

    args = parser.parse_args()

    # Run evaluation
    asyncio.run(run_merge_error_identification(
        num_samples=args.num_samples,
        model=args.model,
        prompt_mode=args.prompt_mode,
        output_dir=args.output_dir
    ))


if __name__ == "__main__":
    main()
