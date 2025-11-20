"""
Script to merge teacher model responses with the ConnectomeBench HuggingFace dataset.

This script matches rows from CSV files containing teacher model completions
with the corresponding rows in the HuggingFace dataset based on:
- proofread_root_id
- current_root_id
- xmin, ymin, zmin, xmax, ymax, zmax

The augmented dataset can then be used for fine-tuning other models.
Works with any teacher model (Claude, GPT-4, Gemini, etc.)
"""

import pandas as pd
from datasets import load_dataset, Dataset
from pathlib import Path
import argparse
from typing import List, Optional
import json


def load_teacher_responses(csv_paths: List[str]) -> pd.DataFrame:
    """
    Load and combine teacher model response CSV files.

    Args:
        csv_paths: List of paths to CSV files containing teacher responses

    Returns:
        Combined DataFrame with all teacher responses
    """
    dfs = []
    for csv_path in csv_paths:
        print(f"Loading {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} rows")
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal teacher responses: {len(combined_df)}")

    # Normalize column names (remove spaces, lowercase)
    combined_df.columns = [col.strip().replace(' ', '_').lower() for col in combined_df.columns]

    return combined_df


def load_hf_dataset(dataset_name: str, config_name: str, split: str = "train") -> pd.DataFrame:
    """
    Load the HuggingFace dataset and convert to DataFrame.

    Args:
        dataset_name: Name of the HuggingFace dataset
        config_name: Configuration name
        split: Dataset split to load

    Returns:
        DataFrame with the dataset
    """
    print(f"\nLoading HuggingFace dataset: {dataset_name} ({config_name})...")
    ds = load_dataset(dataset_name, config_name, split=split)
    print(f"  Loaded {len(ds)} samples")

    # Convert to DataFrame (excluding image columns for now)
    df = ds.to_pandas()

    return df


def create_matching_key(row):
    """
    Create a unique matching key from identifiers and coordinates.

    Args:
        row: DataFrame row

    Returns:
        Tuple of (proofread_root_id, current_root_id, xmin, ymin, zmin, xmax, ymax, zmax)
    """
    return (
        int(row['proofread_root_id']),
        int(row['current_root_id']),
        float(row['xmin']),
        float(row['ymin']),
        float(row['zmin']),
        float(row['xmax']),
        float(row['ymax']),
        float(row['zmax'])
    )


def merge_responses_with_dataset(
    hf_df: pd.DataFrame,
    teacher_df: pd.DataFrame,
    output_path: Optional[str] = None,
    analysis_column: str = 'analysis',
    answer_column: str = 'llm_answer',
    model_column: str = 'model',
    prompt_column: str = 'prompt'
) -> pd.DataFrame:
    """
    Merge teacher model responses with HuggingFace dataset.

    Args:
        hf_df: DataFrame with HuggingFace dataset
        teacher_df: DataFrame with teacher model responses
        output_path: Optional path to save augmented dataset
        analysis_column: Column name in teacher_df containing analysis text
        answer_column: Column name in teacher_df containing answer
        model_column: Column name in teacher_df containing model name
        prompt_column: Column name in teacher_df containing prompt

    Returns:
        Augmented DataFrame with teacher responses added (uses 'teacher_*' columns)
    """
    print("\nCreating matching keys...")

    # Create matching keys for both datasets
    hf_df['_match_key'] = hf_df.apply(create_matching_key, axis=1)
    teacher_df['_match_key'] = teacher_df.apply(create_matching_key, axis=1)

    # Create a lookup dictionary from teacher responses
    print("Creating teacher response lookup...")
    teacher_lookup = {}

    for idx, row in teacher_df.iterrows():
        key = row['_match_key']
        teacher_lookup[key] = {
            'teacher_analysis': row.get(analysis_column, ''),
            'teacher_answer': row.get(answer_column, ''),
            'teacher_model': row.get(model_column, ''),
            'teacher_prompt': row.get(prompt_column, ''),
            'teacher_add_guidance': row.get('add_guidance', False),
        }

    print(f"Created lookup for {len(teacher_lookup)} unique teacher responses")

    # Match and add teacher responses to HuggingFace dataset
    print("\nMatching responses with dataset...")

    teacher_analysis = []
    teacher_answer = []
    teacher_model = []
    teacher_prompt = []
    teacher_add_guidance = []
    teacher_correct = []

    matched_count = 0
    correct_count = 0

    # Reverse mapping to check correctness
    REVERSE_CLASS_MAPPING = {
        "a single soma and process(es)": "a",
        "multiple somas (and processes)": "b",
        "Processes without a soma: These can be axons, dendrites, synapses": "c",
        "Nucleus": "d",
        "Non-neuronal types. These can be glial cells, blood vessels.": "e",
        "None of the above.": "f",
        "Unsure": "g"
    }

    for idx, row in hf_df.iterrows():
        key = row['_match_key']

        if key in teacher_lookup:
            response = teacher_lookup[key]
            teacher_analysis.append(response['teacher_analysis'])
            teacher_answer.append(response['teacher_answer'])
            teacher_model.append(response['teacher_model'])
            teacher_prompt.append(response['teacher_prompt'])
            teacher_add_guidance.append(response['teacher_add_guidance'])

            # Check if teacher's answer is correct
            expected_answer = REVERSE_CLASS_MAPPING.get(row['ground_truth'], None)
            is_correct = (response['teacher_answer'] == expected_answer)
            teacher_correct.append(is_correct)

            matched_count += 1
            if is_correct:
                correct_count += 1
        else:
            # No match found - add None values
            teacher_analysis.append(None)
            teacher_answer.append(None)
            teacher_model.append(None)
            teacher_prompt.append(None)
            teacher_add_guidance.append(None)
            teacher_correct.append(None)

    # Add columns to dataset with fixed 'teacher_' prefix
    hf_df['teacher_analysis'] = teacher_analysis
    hf_df['teacher_answer'] = teacher_answer
    hf_df['teacher_model'] = teacher_model
    hf_df['teacher_prompt'] = teacher_prompt
    hf_df['teacher_add_guidance'] = teacher_add_guidance
    hf_df['teacher_correct'] = teacher_correct

    # Remove temporary matching key
    hf_df = hf_df.drop(columns=['_match_key'])

    print(f"\nMatching complete:")
    print(f"  Matched: {matched_count}/{len(hf_df)} samples ({matched_count/len(hf_df)*100:.1f}%)")
    print(f"  Unmatched: {len(hf_df) - matched_count}")

    # Print some statistics
    print(f"\nTeacher response statistics:")
    print(f"  Samples with analysis: {hf_df['teacher_analysis'].notna().sum()}")
    print(f"  Samples with answer: {hf_df['teacher_answer'].notna().sum()}")
    print(f"  Correct answers: {correct_count}/{matched_count} ({correct_count/matched_count*100:.1f}%)")

    if output_path:
        print(f"\nSaving augmented dataset to {output_path}...")
        hf_df.to_parquet(output_path, index=False)
        print("  Saved!")

        # Also save metadata
        metadata = {
            'total_samples': len(hf_df),
            'matched_samples': matched_count,
            'unmatched_samples': len(hf_df) - matched_count,
            'match_rate': matched_count / len(hf_df),
            'teacher_csv_files': len(teacher_df),
            'unique_teacher_responses': len(teacher_lookup),
        }

        metadata_path = output_path.replace('.parquet', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata saved to {metadata_path}")

    return hf_df


def main():
    parser = argparse.ArgumentParser(
        description="Merge teacher model responses with ConnectomeBench HuggingFace dataset"
    )
    parser.add_argument(
        '--csv-files',
        nargs='+',
        required=True,
        help='Paths to CSV files with teacher model responses'
    )
    parser.add_argument(
        '--dataset-name',
        default='jeffbbrown2/ConnectomeBench',
        help='HuggingFace dataset name'
    )
    parser.add_argument(
        '--config-name',
        default='MICrONS, Segment Classification',
        help='Dataset configuration name'
    )
    parser.add_argument(
        '--split',
        default='train',
        help='Dataset split to use'
    )
    parser.add_argument(
        '--output',
        default='data/connectomebench_with_teacher_responses.parquet',
        help='Output path for augmented dataset (parquet format)'
    )
    parser.add_argument(
        '--analysis-column',
        default='analysis',
        help='Column name in CSV containing analysis text'
    )
    parser.add_argument(
        '--answer-column',
        default='llm_answer',
        help='Column name in CSV containing answer'
    )
    parser.add_argument(
        '--model-column',
        default='model',
        help='Column name in CSV containing model name'
    )
    parser.add_argument(
        '--prompt-column',
        default='prompt',
        help='Column name in CSV containing prompt'
    )

    args = parser.parse_args()

    # Load teacher responses
    teacher_df = load_teacher_responses(args.csv_files)

    # Load HuggingFace dataset
    hf_df = load_hf_dataset(args.dataset_name, args.config_name, args.split)

    # Merge
    augmented_df = merge_responses_with_dataset(
        hf_df=hf_df,
        teacher_df=teacher_df,
        output_path=None,  # Don't save yet
        analysis_column=args.analysis_column,
        answer_column=args.answer_column,
        model_column=args.model_column,
        prompt_column=args.prompt_column
    )

    # Always filter to only correct teacher responses
    print("\nFiltering to only correct teacher responses...")
    before_count = len(augmented_df)
    augmented_df = augmented_df[augmented_df['teacher_correct'] == True]
    after_count = len(augmented_df)
    print(f"  Filtered from {before_count} to {after_count} samples")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving augmented dataset to {output_path}...")
    augmented_df.to_parquet(str(output_path), index=False)
    print("  Saved!")

    # Save metadata
    matched_count = augmented_df['teacher_analysis'].notna().sum()
    correct_count = augmented_df['teacher_correct'].sum() if 'teacher_correct' in augmented_df.columns else 0
    metadata = {
        'total_samples': len(augmented_df),
        'matched_samples': int(matched_count),
        'correct_samples': int(correct_count),
        'accuracy': float(correct_count / matched_count) if matched_count > 0 else 0,
        'match_rate': float(matched_count / len(augmented_df)) if len(augmented_df) > 0 else 0,
        'teacher_csv_files': args.csv_files,
        'dataset_name': args.dataset_name,
        'config_name': args.config_name,
        'split': args.split,
        'analysis_column': args.analysis_column,
        'answer_column': args.answer_column,
        'model_column': args.model_column,
    }

    metadata_path = str(output_path).replace('.parquet', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {metadata_path}")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)
    print(f"\nAugmented dataset saved to: {output_path}")
    print(f"Total samples: {len(augmented_df)}")
    print(f"Samples with teacher responses: {matched_count}")
    print(f"Correct teacher responses: {correct_count} ({correct_count/matched_count*100:.1f}% accuracy)" if matched_count > 0 else "Correct teacher responses: 0")

    # Show a few examples
    print("\nExample matched samples:")
    matched_samples = augmented_df[augmented_df['teacher_analysis'].notna()].head(3)
    for idx, row in matched_samples.iterrows():
        print(f"\nSample {idx}:")
        print(f"  Species: {row['species']}")
        print(f"  Proofread ID: {row['proofread_root_id']}")
        print(f"  Ground truth: {row['ground_truth']}")
        print(f"  Teacher answer: {row['teacher_answer']}")
        print(f"  Teacher model: {row['teacher_model']}")
        print(f"  Analysis preview: {str(row['teacher_analysis'])[:100]}...")


if __name__ == '__main__':
    main()
