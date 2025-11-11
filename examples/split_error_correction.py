"""
Tutorial: Split Error Correction Task

This script demonstrates how to run the split error correction task using the
ConnectomeBench dataset from HuggingFace.

Task: Given images of neuron segments from before and after a split operation,
identify which segment had the split error (was incorrectly split and later merged back).

Usage:
    python split_error_correction.py --num-samples 10 --model gpt-4o
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
from prompts import create_split_identification_prompt
from util import LLMProcessor, evaluate_response


async def run_split_error_correction(
    num_samples: int = None,
    model: str = "gpt-4o",
    prompt_mode: str = 'informative',
    output_dir: str = "output/tutorial_results"
):
    """
    Run split error correction task on ConnectomeBench dataset.

    Args:
        num_samples: Number of samples to evaluate (None = all)
        model: LLM model name to use
        prompt_mode: Prompt mode ('informative' or 'minimal')
        output_dir: Directory to save results
    """
    print("="*60)
    print("ConnectomeBench Tutorial: Split Error Correction")
    print("="*60)

    # Load dataset from HuggingFace
    print("\nLoading dataset from HuggingFace...")
    try:
        ds = load_dataset("jeffbbrown2/ConnectomeBench", "MICrONS, Split Error Correction", split="train")
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
    sample_metadata = []

    for idx, sample in enumerate(ds):
        # Save PIL images to temporary files
        sample_temp_dir = os.path.join(temp_dir, f"sample_{idx}")
        os.makedirs(sample_temp_dir, exist_ok=True)

        # Determine number of options (check for option_1, option_2, etc.)
        option_num = 1
        options_data = []

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
                },
                'merge_coords': str(sample.get('coords', '')),
                'zoom_margin': 1024  # Default zoom margin
            }
            options_data.append(option_data)

            # Create prompt for this specific option (split_identification asks about each option individually)
            prompt = create_split_identification_prompt(
                option_data,
                use_zoomed_images=True,
                views=['front', 'side', 'top'],
                llm_processor=llm_processor,
                zoom_margin=option_data['zoom_margin'],
                prompt_mode=prompt_mode
            )
            prompts.append(prompt)

            # Track which sample and option this prompt corresponds to
            sample_metadata.append({
                'sample_idx': idx,
                'option_num': option_num,
                'option_id': option_data['id']
            })

            option_num += 1

    print(f"Created {len(prompts)} prompts")

    # Run LLM evaluation
    print("\nRunning LLM evaluation...")
    llm_responses = await llm_processor.process_batch(prompts)

    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary directory")

    # Process results (same as original script)
    print("\nProcessing results...")
    results = []

    for prompt_metadata, response in zip(sample_metadata, llm_responses):
        sample = ds[prompt_metadata['sample_idx']]

        # Parse LLM response using evaluate_response (same as original)
        answer_analysis = evaluate_response(response)

        # Get ground truth (the neuron_id that had the split error)
        ground_truth_id = sample.get('ground_truth', None)
        option_id = prompt_metadata['option_id']

        # Answer is "1" if merge error detected, "-1" (converted to "none") if not
        predicted_answer = answer_analysis.get('answer', 'none')

        # Determine correctness:
        # If this option_id == ground_truth_id, model should return "1"
        # If this option_id != ground_truth_id, model should return "-1" (shows as "none")
        correct = None
        if ground_truth_id:
            is_correct_option = (option_id == ground_truth_id)
            if is_correct_option:
                # This IS the option with split error, model should say "1"
                correct = (predicted_answer == "1")
            else:
                # This is NOT the option with split error, model should say "none" (originally "-1")
                correct = (predicted_answer == "none")

        # Create result entry
        result = {
            'operation_id': sample.get('operation_id'),
            'species': sample.get('species'),
            'coords': sample.get('coords'),
            'option_num': prompt_metadata['option_num'],
            'option_id': option_id,
            'model': model,
            'model_answer': predicted_answer,
            'model_analysis': answer_analysis.get('analysis', ''),
            'ground_truth_id': ground_truth_id,
            'is_correct_option': is_correct_option if ground_truth_id else None,
            'one_valid_option': sample.get('one_valid_option'),
            'correct': correct,
            'full_response': response,
            'prompt_mode': prompt_mode
        }
        results.append(result)

    # Create DataFrame and save
    results_df = pd.DataFrame(results)

    # Calculate accuracy if ground truth available
    if 'correct' in results_df.columns and results_df['correct'].notna().any():
        accuracy = results_df['correct'].mean()
        print(f"\nAccuracy: {accuracy:.2%}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/split_error_correction_{model.replace('/', '_')}_{prompt_mode}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\nSummary of predictions:")
    print(results_df['model_answer'].value_counts())

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Tutorial: Run split error correction on ConnectomeBench dataset"
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
    asyncio.run(run_split_error_correction(
        num_samples=args.num_samples,
        model=args.model,
        prompt_mode=args.prompt_mode,
        output_dir=args.output_dir
    ))


if __name__ == "__main__":
    main()
