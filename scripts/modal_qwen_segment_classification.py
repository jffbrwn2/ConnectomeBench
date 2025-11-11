import modal
import os
from pathlib import Path

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
    )
)

app = modal.App("qwen-segment-classification", image=image)

# Create a volume to store results
volume = modal.Volume.from_name("segment-results", create_if_missing=True)

@app.function(
    gpu="A100",  # Use A100 GPU for the 7B model
    timeout=3600,  # 1 hour timeout
    volumes={"/results": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],  # HF token for dataset access
)
def run_segment_classification(num_samples: int = 10):
    """
    Run Qwen2-VL-7B on segment classification task.

    Args:
        num_samples: Number of samples to process (default: 10)
    """
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    from datasets import load_dataset
    import pandas as pd
    import numpy as np

    print(f"Loading model: Qwen/Qwen2-VL-7B-Instruct...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    print(f"Loading dataset...")
    ds = load_dataset(
        "jeffbbrown2/ConnectomeBench",
        "MICrONS, Segment Classifications",
        split="train"
    )

    # Limit to num_samples
    if num_samples < len(ds):
        ds = ds.select(range(num_samples))

    print(f"Processing {len(ds)} samples...")

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
        front_img = sample['option_1_front_path']
        side_img = sample['option_1_side_path']
        top_img = sample['option_1_top_path']

        # Create the prompt based on the original prompts.py
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

        analysis_start = output_text.find("<analysis>")
        analysis_end = output_text.find("</analysis>")
        if analysis_start != -1 and analysis_end != -1:
            analysis = output_text[analysis_start + len("<analysis>"):analysis_end].strip()

        answer_start = output_text.find("<answer>")
        answer_end = output_text.find("</answer>")
        if answer_start != -1 and answer_end != -1:
            llm_answer = output_text[answer_start + len("<answer>"):answer_end].strip()

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
            'full_response': output_text,
            'model': 'Qwen/Qwen2-VL-7B-Instruct'
        })

        print(f"  Answer: {llm_answer}")

    # Save results
    df = pd.DataFrame(results)
    output_path = f"/results/qwen2_vl_7b_results_{num_samples}samples.csv"
    df.to_csv(output_path, index=False)
    volume.commit()

    print(f"\nResults saved to: {output_path}")
    print(f"Processed {len(results)} samples")

    return df


@app.local_entrypoint()
def main(num_samples: int = 10):
    """
    Local entry point to run the segment classification.

    Usage:
        modal run scripts/modal_qwen_segment_classification.py --num-samples 10
    """
    print(f"Starting Qwen2-VL segment classification with {num_samples} samples...")
    result_df = run_segment_classification.remote(num_samples)
    print("\nCompleted!")
    print(result_df[['species', 'current_root_id', 'llm_answer', 'ground_truth']].head())
