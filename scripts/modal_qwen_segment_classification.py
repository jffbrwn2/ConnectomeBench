import modal
from pathlib import Path

# Paths for model and results storage
MODEL_DIR = Path("/models")
RESULTS_DIR = Path("/results")

# Create volumes
model_volume = modal.Volume.from_name("qwen-models-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("segment-results", create_if_missing=True)

# Define the Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "torchvision",
        "transformers>=4.57.0",
        "qwen-vl-utils",
        "datasets",
        "pandas",
        "Pillow",
        "accelerate",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # Fast Rust-based downloads
)

app = modal.App("qwen-segment-classification", image=image)

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

@app.function(
    gpu="A100",  # Use A100 GPU for the model
    timeout=3600,  # 1 hour timeout
    volumes={
        MODEL_DIR: model_volume,
        RESULTS_DIR: results_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],  # HF token for dataset access
)
def run_segment_classification(
    num_samples: int = 10,
    use_blank_images: bool = False,
    use_simple_prompt: bool = False,
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
):
    """
    Run Qwen-VL on segment classification task.

    Args:
        num_samples: Number of samples to process (default: 10)
        use_blank_images: If True, use blank gray images instead of real ones (sanity check)
        use_simple_prompt: If True, use simple "what do you see" prompt (sanity check)
        model_name: HuggingFace model ID (e.g., "Qwen/Qwen2-VL-7B-Instruct" or "Qwen/Qwen3-VL-8B-Instruct")
    """
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from datasets import load_dataset
    from PIL import Image
    import pandas as pd
    import numpy as np

    print(f"Loading model: {model_name}...")

    # Set cache directory to use the volume
    import os
    cache_dir = MODEL_DIR / model_name.replace("/", "--")
    os.environ["HF_HOME"] = str(MODEL_DIR)

    # Check if model is already cached
    if not cache_dir.exists():
        print(f"Model not cached, downloading to {cache_dir}...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=model_name,
            local_dir=str(cache_dir),
            ignore_patterns=["*.pt", "*.bin", "*.pth"],  # Skip safetensors duplicates
        )
        model_volume.commit()
        print("Model downloaded and cached!")
    else:
        print(f"Using cached model from {cache_dir}")

    # Load model from cache
    model = AutoModelForImageTextToText.from_pretrained(
        str(cache_dir),
        dtype=torch.bfloat16,
        device_map=None,
        local_files_only=True,
    )
    model.to("cuda")
    processor = AutoProcessor.from_pretrained(str(cache_dir), local_files_only=True)

    # Determine if we need qwen_vl_utils (only for Qwen2)
    use_qwen2_processing = "Qwen2" in model_name or "qwen2" in model_name.lower()

    print(f"Loading dataset...")
    ds = load_dataset(
        "jeffbbrown2/ConnectomeBench",
        "MICrONS, Segment Classification",
        split="train"
    )

    # Limit to num_samples
    if num_samples < len(ds):
        ds = ds.select(range(num_samples))

    print(f"Processing {len(ds)} samples...")
    if use_blank_images:
        print("⚠️  SANITY CHECK MODE: Using blank images instead of real ones")
    if use_simple_prompt:
        print("⚠️  SANITY CHECK MODE: Using simple 'what do you see' prompt")

    results = []
    for idx, sample in enumerate(ds):
        print(f"Processing sample {idx + 1}/{len(ds)}...")

        # Extract sample data
        species = sample['species']
        proofread_root_id = sample['proofread_root_id']
        current_root_id = sample['current_root_id']
        ground_truth = sample['ground_truth']

        # Get bounding box dimensions
        xmin, ymin, zmin = sample['xmin'], sample['ymin'], sample['zmin']
        xmax, ymax, zmax = sample['xmax'], sample['ymax'], sample['zmax']
        box_size = np.array([xmax - xmin, ymax - ymin, zmax - zmin])

        # Get the three images (PIL Images from dataset)
        if use_blank_images:
            # Create blank gray images for sanity check
            front_img = Image.new('RGB', (1024, 1024), color=(128, 128, 128))
            side_img = Image.new('RGB', (1024, 1024), color=(128, 128, 128))
            top_img = Image.new('RGB', (1024, 1024), color=(128, 128, 128))
        else:
            front_img = sample['option_1_front_path']
            side_img = sample['option_1_side_path']
            top_img = sample['option_1_top_path']

        # Create the prompt
        if use_simple_prompt:
            prompt = "What do you see in these three images? Describe them in detail."
        else:
            prompt = f"""
You are an expert at analyzing neuronal morphology.

We have the electron microscopy data from the {species} brain.

In the images, we have a selected 3D segmentation that is supposed to correspond to a complete neuronal structure. However, it could have split/merge errors as the segmentation algorithm makes mistakes.

The 3D snapshots are three different views of the same segment. The dimensions of the segment's bounding box are {box_size[0]} x {box_size[1]} x {box_size[2]} nm. Describe in detail what you see using the information in the 3D snapshots. Is the segment a neuron (soma and processes)? Multiple neurons merged together (multiple somas)? Processes like axon and dendrites without a cell body? Non-neuronal structures like glia, astrocytes, or blood vessels? Inspect very closely to avoid making errors, using the 3D views and size of the bounding box in your reasoning.

Choose the best answer:
a) A single soma and process(es).
b) Multiple somas (and processes)
c) Processes without a soma. These can be axons, dendrites, synapses.
d) Nucleus.
e) Non-neuronal types. These can be glial cells, blood vessels.
f) None of the above.
g) Unsure

Surround your analysis with <analysis> and </analysis> tags.
Surround your final answer (the letter a, b, c, d, e, f, or g) with <answer> and </answer> tags.
"""

        # Prepare messages for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": front_img},
                    {"type": "image", "image": side_img},
                    {"type": "image", "image": top_img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template and process
        if use_qwen2_processing:
            # Qwen2-VL uses separate processing with qwen_vl_utils
            from qwen_vl_utils import process_vision_info
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
        else:
            # Qwen3-VL and other models handle everything in apply_chat_template
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to("cuda")

        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

        # Extract analysis and answer
        analysis = ""
        llm_answer = ""

        if use_simple_prompt:
            # For simple prompt, just use the full response
            analysis = output_text
            llm_answer = output_text
        else:
            analysis_start = output_text.find("<analysis>")
            analysis_end = output_text.find("</analysis>")
            if analysis_start != -1 and analysis_end != -1:
                analysis = output_text[analysis_start + len("<analysis>"):analysis_end].strip()

            answer_start = output_text.find("<answer>")
            answer_end = output_text.find("</answer>")
            if answer_start != -1 and answer_end != -1:
                llm_answer = output_text[answer_start + len("<answer>"):answer_end].strip()

        # Evaluate correctness (like in examples/segment_classification.py)
        predicted_description = CLASS_MAPPING.get(llm_answer, None)
        correct = None
        if predicted_description and ground_truth:
            correct = (predicted_description == ground_truth)

        # Store result
        results.append({
            'species': species,
            'proofread_root_id': proofread_root_id,
            'current_root_id': current_root_id,
            'ground_truth': ground_truth,
            'xmin': xmin, 'ymin': ymin, 'zmin': zmin,
            'xmax': xmax, 'ymax': ymax, 'zmax': zmax,
            'unit': sample['unit'],
            'analysis': analysis,
            'llm_answer': llm_answer,
            'predicted_description': predicted_description,
            'correct': correct,
            'full_response': output_text,
            'model': model_name,
            'used_blank_images': use_blank_images,
            'used_simple_prompt': use_simple_prompt
        })

        print(f"  Answer: {llm_answer}")

    # Save results
    df = pd.DataFrame(results)

    # Calculate accuracy if ground truth available
    if 'correct' in df.columns and df['correct'].notna().any():
        accuracy = df['correct'].mean()
        print(f"\n{'='*60}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Correct: {df['correct'].sum()}/{len(df)}")
        print(f"{'='*60}")

        # Print breakdown by answer
        print("\nPrediction breakdown:")
        for answer_key in sorted(df['llm_answer'].unique()):
            subset = df[df['llm_answer'] == answer_key]
            correct_count = subset['correct'].sum() if 'correct' in subset.columns else 0
            total_count = len(subset)
            print(f"  {answer_key}: {correct_count}/{total_count} correct - {CLASS_MAPPING.get(answer_key, 'Unknown')}")

    # Create a clean filename from model name
    model_suffix = model_name.split("/")[-1].lower().replace("-", "_")
    blank_suffix = "_blank" if use_blank_images else ""
    simple_suffix = "_simple" if use_simple_prompt else ""
    output_path = RESULTS_DIR / f"{model_suffix}_results_{num_samples}samples{blank_suffix}{simple_suffix}.csv"
    df.to_csv(str(output_path), index=False)
    results_volume.commit()

    print(f"\nResults saved to: {output_path}")
    print(f"Processed {len(results)} samples")

    return df


@app.local_entrypoint()
def main(
    num_samples: int = 10,
    blank_images: bool = False,
    simple_prompt: bool = False,
    model: str = "Qwen/Qwen2-VL-7B-Instruct"
):
    """
    Local entry point to run the segment classification.

    Usage:
        # Qwen2-VL-7B (default)
        modal run scripts/modal_qwen_segment_classification.py --num-samples 10

        # Qwen3-VL-8B
        modal run scripts/modal_qwen_segment_classification.py --num-samples 10 --model "Qwen/Qwen3-VL-235B-A22B-Thinking"

        # Sanity checks
        modal run scripts/modal_qwen_segment_classification.py --num-samples 10 --blank-images --simple-prompt
        modal run scripts/modal_qwen_segment_classification.py --num-samples 5 --model "Qwen/Qwen3-VL-235B-A22B-Thinking" --simple-prompt
    """
    print(f"Starting segment classification with {model} on {num_samples} samples...")
    result_df = run_segment_classification.remote(
        num_samples,
        use_blank_images=blank_images,
        use_simple_prompt=simple_prompt,
        model_name=model
    )
    print("\nCompleted!")
    print(result_df[['species', 'current_root_id', 'llm_answer', 'ground_truth']].head())
