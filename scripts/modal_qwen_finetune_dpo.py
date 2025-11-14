import modal
from pathlib import Path
from dataclasses import dataclass

# Paths for storage
MODEL_DIR = Path("/models")
DATASET_DIR = Path("/datasets")
CHECKPOINT_DIR = Path("/checkpoints")
RESULTS_DIR = Path("/results")

# Create volumes
model_volume = modal.Volume.from_name("qwen-finetune-models", create_if_missing=True)
dataset_volume = modal.Volume.from_name("qwen-finetune-datasets", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("qwen-finetune-checkpoints", create_if_missing=True)
results_volume = modal.Volume.from_name("qwen-finetune-results", create_if_missing=True)

# Define the Modal image with all dependencies for DPO
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # Install unsloth for efficient training
    .pip_install(
        "unsloth",
        "unsloth_zoo",
    )
    # Install TRL with DPO support
    .pip_install(
        "trl>=0.12.0",
        "datasets",
        "pandas",
        "Pillow",
        "huggingface_hub[hf_transfer]",
        "wandb",
        "accelerate",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("qwen-segment-finetune-dpo", image=image)


@dataclass
class DPOTrainingConfig:
    """Configuration for DPO fine-tuning"""
    # Model settings
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    max_seq_length: int = 2048

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Quantization
    load_in_4bit: bool = True

    # DPO-specific hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-7  # Lower LR for DPO (typically 10x smaller than SFT)

    # DPO loss settings
    beta: float = 0.1  # DPO temperature parameter (controls how much to penalize rejected responses)
    loss_type: str = "sigmoid"  # Options: "sigmoid", "hinge", "ipo", "kto_pair"

    # Optimization
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    warmup_steps: int = 5

    # Logging
    logging_steps: int = 1
    save_steps: int = 100
    save_total_limit: int = 3

    # Precision
    fp16: bool = False
    bf16: bool = True
    seed: int = 42

    # Dataset settings
    num_samples: int = None

    # W&B settings
    use_wandb: bool = False
    wandb_project: str = "qwen-segment-dpo"
    wandb_run_name: str = None


# Class mapping for segment classification
CLASS_MAPPING = {
    "a": "a single soma and process(es)",
    "b": "multiple somas (and processes)",
    "c": "Processes without a soma: These can be axons, dendrites, synapses",
    "d": "Nucleus",
    "e": "Non-neuronal types. These can be glial cells, blood vessels.",
    "f": "None of the above.",
    "g": "Unsure"
}

REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}


def prepare_dataset_for_dpo(dataset, tokenizer):
    """
    Prepare dataset for DPO training.

    DPO requires pairs of (chosen, rejected) responses for each prompt.
    We'll create synthetic pairs based on ground truth:
    - Chosen: Correct answer with analysis
    - Rejected: Incorrect answer (from other classes)

    Args:
        dataset: HuggingFace dataset with ground truth labels
        tokenizer: Model tokenizer

    Returns:
        Dataset with 'prompt', 'chosen', 'rejected' columns
    """
    import numpy as np
    import random

    def create_dpo_pairs(sample):
        """Convert a single sample to DPO format with chosen/rejected pairs"""
        # Get sample metadata
        species = sample['species']
        ground_truth = sample['ground_truth']
        ground_truth_key = REVERSE_CLASS_MAPPING.get(ground_truth, "g")

        # Get bounding box dimensions
        xmin, ymin, zmin = sample['xmin'], sample['ymin'], sample['zmin']
        xmax, ymax, zmax = sample['xmax'], sample['ymax'], sample['zmax']
        box_size = np.array([xmax - xmin, ymax - ymin, zmax - zmin])

        # Get the three images
        front_img = sample['option_1_front_path']
        side_img = sample['option_1_side_path']
        top_img = sample['option_1_top_path']

        # Create the instruction prompt (same as SFT)
        prompt_text = f"""You are an expert at analyzing neuronal morphology.

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
Surround your final answer (the letter a, b, c, d, e, f, or g) with <answer> and </answer> tags."""

        # Create user prompt messages
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": front_img},
                    {"type": "image", "image": side_img},
                    {"type": "image", "image": top_img},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Create CHOSEN response (correct answer)
        chosen_response = f"""<analysis>
Based on the three orthogonal views of this segment, I can analyze its structure.

The bounding box dimensions of {box_size[0]} x {box_size[1]} x {box_size[2]} nm provide important scale information.

Looking at the morphology across all three views, this segment appears to be: {ground_truth}
</analysis>

<answer>{ground_truth_key}</answer>"""

        # Create REJECTED response (incorrect answer)
        # Pick a random wrong answer
        all_keys = list(CLASS_MAPPING.keys())
        wrong_keys = [k for k in all_keys if CLASS_MAPPING[k] != ground_truth]
        rejected_key = random.choice(wrong_keys)
        rejected_description = CLASS_MAPPING[rejected_key]

        rejected_response = f"""<analysis>
Based on the three orthogonal views of this segment, I can analyze its structure.

The bounding box dimensions of {box_size[0]} x {box_size[1]} x {box_size[2]} nm provide important scale information.

Looking at the morphology across all three views, this segment appears to be: {rejected_description}
</analysis>

<answer>{rejected_key}</answer>"""

        # For DPO, we need full conversations
        chosen_messages = prompt_messages + [
            {"role": "assistant", "content": chosen_response}
        ]

        rejected_messages = prompt_messages + [
            {"role": "assistant", "content": rejected_response}
        ]

        return {
            "prompt": prompt_messages,
            "chosen": chosen_messages,
            "rejected": rejected_messages,
        }

    # Set random seed for reproducibility
    random.seed(42)

    # Map dataset
    dpo_dataset = dataset.map(create_dpo_pairs, remove_columns=dataset.column_names)

    return dpo_dataset


@app.function(
    gpu="H100",  # H100 recommended for large vision models
    timeout=7200,  # 2 hours
    volumes={
        MODEL_DIR: model_volume,
        DATASET_DIR: dataset_volume,
        CHECKPOINT_DIR: checkpoint_volume,
        RESULTS_DIR: results_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def finetune_qwen_dpo(config: DPOTrainingConfig = DPOTrainingConfig()):
    """
    Fine-tune Qwen-VL model on segment classification using DPO.

    Args:
        config: DPOTrainingConfig object with all training parameters
    """
    import torch
    import os
    from unsloth import FastVisionModel, is_bf16_supported
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    print("="*60)
    print("Starting Qwen-VL DPO Fine-tuning")
    print("="*60)

    # Set cache directories
    os.environ["HF_HOME"] = str(MODEL_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_DIR)

    # Initialize W&B if requested
    if config.use_wandb:
        import wandb
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"{config.model_name.split('/')[-1]}-dpo",
            config=config.__dict__
        )

    # 1. Load model with Unsloth
    print(f"\nLoading model: {config.model_name}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=config.model_name,
        load_in_4bit=config.load_in_4bit,
        max_seq_length=config.max_seq_length,
    )

    # 2. Add LoRA adapters
    print("\nAdding LoRA adapters...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        random_state=config.seed,
        use_rslora=False,
        loftq_config=None,
    )

    # 3. Load dataset
    print("\nLoading ConnectomeBench dataset...")
    dataset = load_dataset(
        "jeffbbrown2/ConnectomeBench",
        "MICrONS, Segment Classification",
        split="train",
        cache_dir=str(DATASET_DIR)
    )

    # Limit samples if specified
    if config.num_samples is not None and config.num_samples < len(dataset):
        dataset = dataset.select(range(config.num_samples))
        print(f"Using {config.num_samples} samples for training")
    else:
        print(f"Using all {len(dataset)} samples for training")

    # 4. Prepare dataset for DPO (create chosen/rejected pairs)
    print("\nPreparing dataset for DPO...")
    print("Creating synthetic preference pairs (correct vs incorrect answers)...")
    train_dataset = prepare_dataset_for_dpo(dataset, tokenizer)
    print(f"Dataset ready with {len(train_dataset)} preference pairs")

    # 5. Setup DPO training arguments
    print("\nSetting up DPO training...")
    training_args = DPOConfig(
        output_dir=str(CHECKPOINT_DIR / config.model_name.replace("/", "_")),

        # Training hyperparameters
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,

        # DPO-specific
        beta=config.beta,
        loss_type=config.loss_type,

        # Optimization
        optim=config.optim,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,

        # Precision
        fp16=config.fp16 and not is_bf16_supported(),
        bf16=config.bf16 and is_bf16_supported(),

        # Logging and saving
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,

        # Other
        seed=config.seed,
        report_to="wandb" if config.use_wandb else "none",
    )

    # 6. Create DPO trainer
    print("\nCreating DPO trainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # 7. Train!
    print("\n" + "="*60)
    print("Starting DPO training...")
    print("="*60)

    trainer.train()

    # 8. Save the LoRA adapters with unique identifier
    print("\nSaving LoRA adapters...")

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = config.model_name.split('/')[-1]

    run_name_parts = [
        model_short_name,
        "dpo",  # Indicate this is DPO training
        timestamp,
        f"samples{config.num_samples if config.num_samples else 'all'}",
        f"epochs{config.num_train_epochs}",
        f"lr{config.learning_rate}",
        f"r{config.lora_r}",
        f"beta{config.beta}",
    ]

    if config.wandb_run_name:
        run_name_parts.insert(1, config.wandb_run_name)

    run_folder_name = "_".join(run_name_parts)
    final_model_path = CHECKPOINT_DIR / run_folder_name

    model.save_pretrained(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    # Save training config
    import json
    config_dict = {
        "model_name": config.model_name,
        "training_method": "dpo",
        "num_samples": config.num_samples,
        "num_train_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "beta": config.beta,
        "loss_type": config.loss_type,
        "batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "timestamp": timestamp,
    }
    with open(str(final_model_path / "training_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    checkpoint_volume.commit()

    print("\n" + "="*60)
    print("DPO Training completed!")
    print(f"LoRA adapters saved to: {final_model_path}")
    print(f"\nTo evaluate this model, use the evaluation script:")
    print(f"  modal run scripts/modal_qwen_evaluate_adapter.py \\")
    print(f"    --adapter-path \"{run_folder_name}\" \\")
    print(f"    --num-samples 100")
    print("="*60)

    if config.use_wandb:
        wandb.finish()

    return str(final_model_path)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    num_samples: int = None,
    epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation: int = 4,
    learning_rate: float = 5e-7,
    lora_r: int = 16,
    beta: float = 0.1,
    loss_type: str = "sigmoid",
    use_4bit: bool = True,
    use_wandb: bool = False,
    run_name: str = None,
):
    """
    Local entry point to start DPO fine-tuning.

    DPO (Direct Preference Optimization) is simpler than GRPO and works by:
    1. Creating pairs of correct (chosen) vs incorrect (rejected) responses
    2. Training the model to prefer correct responses over incorrect ones
    3. No need for online generation during training (more stable)

    Usage:
        # Basic DPO fine-tuning with Qwen3-VL-8B on small subset
        modal run scripts/modal_qwen_finetune_dpo.py --num-samples 100 --epochs 3

        # DPO with custom beta (higher = stronger preference for chosen)
        modal run scripts/modal_qwen_finetune_dpo.py \\
            --num-samples 500 \\
            --epochs 5 \\
            --beta 0.2 \\
            --run-name "high_beta"

        # Full DPO training with W&B tracking
        modal run scripts/modal_qwen_finetune_dpo.py \\
            --epochs 3 \\
            --use-wandb \\
            --run-name "dpo_full_v1"

        # Try different loss types
        modal run scripts/modal_qwen_finetune_dpo.py \\
            --num-samples 200 \\
            --loss-type "ipo" \\
            --run-name "ipo_loss"
    """
    print(f"Starting DPO fine-tuning with {model}...")
    print(f"Samples: {num_samples if num_samples else 'all'}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size} (effective: {batch_size * gradient_accumulation})")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA rank: {lora_r}")
    print(f"Beta (preference strength): {beta}")
    print(f"Loss type: {loss_type}")
    print(f"4-bit quantization: {use_4bit}")

    # Create config
    config = DPOTrainingConfig(
        model_name=model,
        num_samples=num_samples,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        lora_r=lora_r,
        beta=beta,
        loss_type=loss_type,
        load_in_4bit=use_4bit,
        use_wandb=use_wandb,
        wandb_run_name=run_name,
    )

    # Run DPO fine-tuning
    final_path = finetune_qwen_dpo.remote(config)

    print("\n" + "="*60)
    print("DPO fine-tuning job completed!")
    print(f"Model saved at: {final_path}")
    print("\nNext steps:")
    print("1. Evaluate the model using modal_qwen_evaluate_adapter.py")
    print("2. Compare performance with supervised fine-tuning")
    print("3. Try different hyperparameters (beta, loss_type)")
    print("="*60)
