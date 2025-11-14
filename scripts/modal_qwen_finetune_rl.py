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

# Define the Modal image with all dependencies for GRPO
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # Install unsloth for efficient training
    .pip_install(
        "unsloth",
        "unsloth_zoo",
    )
    # Install TRL with GRPO support
    .pip_install(
        "trl>=0.12.0",  # GRPO support added in 0.12.0
        "datasets",
        "pandas",
        "Pillow",
        "huggingface_hub[hf_transfer]",
        "wandb",
        "accelerate",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("qwen-segment-finetune-rl", image=image)


@dataclass
class GRPOTrainingConfig:
    """Configuration for GRPO fine-tuning"""
    # Model settings
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    max_seq_length: int = 2048

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Quantization
    load_in_4bit: bool = True

    # GRPO-specific hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4  # Should be multiple of num_generations
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4

    # GRPO generation settings
    num_generations: int = 4  # Generate multiple responses per prompt (batch_size must be multiple of this)
    max_prompt_length: int = 512  # Max length for prompt
    max_completion_length: int = 1024  # Max length for generated completion
    temperature: float = 0.7  # For exploration during training

    # Reward settings
    beta: float = 0.1  # KL divergence coefficient (prevents drift from base model)

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

    # Vision-specific settings (for Qwen-VL)
    max_pixels: int = None  # Maximum image pixels (None = use model default)
    min_pixels: int = None  # Minimum image pixels (None = use model default)

    # W&B settings
    use_wandb: bool = False
    wandb_project: str = "qwen-segment-rl"
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


def create_reward_function():
    """
    Create a reward function for GRPO that evaluates model responses.

    Returns:
        A function that takes completions and ground_truths and returns rewards
    """
    def reward_fn(completions, ground_truth=None, **kwargs):
        """
        Evaluate model completions and return rewards.

        Args:
            completions: List of model-generated completion strings
            ground_truth: List of ground truth class descriptions (from dataset column)
            **kwargs: Other dataset columns (ignored)

        Returns:
            List of reward scores (floats between 0 and 1)
        """
        import re

        # Handle case where ground_truth is not provided
        if ground_truth is None:
            print("Warning: No ground_truth provided to reward function")
            return [0.0] * len(completions)

        rewards = []
        for idx, (completion, gt) in enumerate(zip(completions, ground_truth)):
            reward = 0.0

            # Handle different completion formats
            # For vision models, completion might be a list of message dicts or a string
            if isinstance(completion, list):
                # Extract text from list of messages
                completion_text = ""
                for msg in completion:
                    if isinstance(msg, dict) and "content" in msg:
                        completion_text += str(msg["content"]) + " "
                    elif isinstance(msg, str):
                        completion_text += msg + " "
                completion_text = completion_text.strip()
            else:
                completion_text = str(completion)

            # Extract the answer from <answer> tags
            answer_match = re.search(r'<answer>\s*([a-g])\s*</answer>', completion_text.lower())
            predicted_letter = None
            predicted_description = None

            if answer_match:
                predicted_letter = answer_match.group(1)
                predicted_description = CLASS_MAPPING.get(predicted_letter)

                # Full reward if correct
                if predicted_description == gt:
                    reward = 1.0
                else:
                    # No reward for incorrect answer
                    reward = 0.0
            else:
                # Penalty for not following format
                reward = 0.0

            # Optional: Add small bonus for including analysis tags (encourages reasoning)
            if '<analysis>' in completion_text.lower() and '</analysis>' in completion_text.lower():
                reward += 0.1

            # Clip reward to [0, 1]
            reward = min(1.0, max(0.0, reward))
            rewards.append(reward)

            # Print first 3 completions of each batch for debugging
            if idx < 3:
                gt_key = REVERSE_CLASS_MAPPING.get(gt, "?")
                print(f"\n{'='*60}")
                print(f"Sample {idx + 1}:")
                print(f"Ground Truth: {gt_key} - {gt}")
                print(f"Predicted: {predicted_letter if predicted_letter else 'NONE'} - {predicted_description if predicted_description else 'NO ANSWER FOUND'}")
                print(f"Reward: {reward:.2f}")
                print(f"\nCompletion (first 500 chars):")
                print(completion_text[:500])
                print(f"{'='*60}\n")

        return rewards

    return reward_fn


def prepare_dataset_for_grpo(dataset, tokenizer):
    """
    Prepare dataset for GRPO training with vision models.

    CRITICAL: GRPO expects simple string content in messages, NOT pre-structured
    with image dictionaries. The trainer's prepare_multimodal_messages() function
    automatically converts string content and injects images from the "images" column.

    Args:
        dataset: HuggingFace dataset
        tokenizer: Model tokenizer

    Returns:
        Dataset with:
        - 'prompt': List of messages with simple string content
        - 'images': List of PIL images
        - 'ground_truth': Ground truth labels for reward function
    """
    import numpy as np

    def format_sample(sample):
        """Convert a single sample to GRPO format for vision models"""
        # Get sample metadata
        species = sample['species']
        ground_truth = sample['ground_truth']

        # Get bounding box dimensions
        xmin, ymin, zmin = sample['xmin'], sample['ymin'], sample['zmin']
        xmax, ymax, zmax = sample['xmax'], sample['ymax'], sample['zmax']
        box_size = np.array([xmax - xmin, ymax - ymin, zmax - zmin])

        # Get the three images (PIL Images from dataset)
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

        # Format as conversational messages with SIMPLE STRING CONTENT
        # GRPO's prepare_multimodal_messages() will automatically:
        # 1. Convert string to [{"type": "text", "text": ...}]
        # 2. Add {"type": "image"} placeholders at the beginning
        # 3. Inject actual PIL images from the "images" column
        prompt_messages = [
            {
                "role": "user",
                "content": prompt_text,  # Simple string, NOT pre-structured!
            }
        ]

        return {
            "prompt": prompt_messages,  # Simple string content
            "images": [front_img, side_img, top_img],  # Separate PIL images list
            "ground_truth": ground_truth,  # Will be passed to reward function
        }

    # Map to add prompts and images
    # PIL images in "images" column will be handled properly by GRPO
    formatted = dataset.map(format_sample)

    return formatted


@app.function(
    gpu="H100",  # H100 recommended for GRPO due to memory requirements
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
def finetune_qwen_grpo(config: GRPOTrainingConfig = GRPOTrainingConfig()):
    """
    Fine-tune Qwen-VL model on segment classification using GRPO (RL).

    Args:
        config: GRPOTrainingConfig object with all training parameters
    """
    import torch
    import os
    from unsloth import FastVisionModel, is_bf16_supported
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    print("="*60)
    print("Starting Qwen-VL RL Fine-tuning with GRPO")
    print("="*60)

    # Set cache directories
    os.environ["HF_HOME"] = str(MODEL_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_DIR)

    # Initialize W&B if requested
    if config.use_wandb:
        import wandb
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"{config.model_name.split('/')[-1]}-grpo",
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

    # 4. Prepare dataset for GRPO (prompts only, no assistant responses)
    print("\nPreparing dataset for GRPO...")
    train_dataset = prepare_dataset_for_grpo(dataset, tokenizer)
    print(f"Dataset ready with {len(train_dataset)} samples")

    # Debug: Check what the first sample looks like
    print("\nDEBUG: Inspecting first sample...")
    first_sample = train_dataset[0]
    print(f"Keys: {first_sample.keys()}")
    print(f"Ground truth: {first_sample['ground_truth']}")

    # Check prompt format
    print(f"\nPrompt type: {type(first_sample['prompt'])}")
    if isinstance(first_sample['prompt'], list) and len(first_sample['prompt']) > 0:
        print(f"First message keys: {first_sample['prompt'][0].keys()}")
        if 'content' in first_sample['prompt'][0]:
            content = first_sample['prompt'][0]['content']
            print(f"Content type: {type(content)}")
            if isinstance(content, str):
                print(f"Content length: {len(content)} chars")
                print(f"Content preview (first 200 chars): {content[:200]}")
            else:
                print(f"WARNING: Content is not a string! Type: {type(content)}")

    # Check images column
    if 'images' in first_sample:
        images = first_sample['images']
        print(f"\nImages type: {type(images)}")
        print(f"Number of images: {len(images) if images else 0}")
        if images and len(images) > 0:
            for idx, img in enumerate(images):
                print(f"  Image[{idx}] type: {type(img)}")
                if hasattr(img, 'size'):
                    print(f"    PIL Image size: {img.size}")
                elif isinstance(img, dict):
                    print(f"    WARNING: Image is a dict (serialized)!")
    else:
        print("\nWARNING: No 'images' column found!")

    # 5. Create reward function
    print("\nSetting up reward function...")
    reward_fn = create_reward_function()

    # 6. Setup GRPO training arguments
    print("\nSetting up GRPO training...")
    training_args = GRPOConfig(
        output_dir=str(CHECKPOINT_DIR / config.model_name.replace("/", "_")),

        # Training hyperparameters
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,

        # GRPO-specific: Data preprocessing and generation
        num_generations=config.num_generations,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        beta=config.beta,  # KL divergence weight

        # Disable automatic dataset processing - we provide pre-formatted messages
        remove_unused_columns=False,  # Keep all columns including images

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
        log_completions=True,  # Log generated completions for debugging

        # Other
        seed=config.seed,
        report_to="wandb" if config.use_wandb else "none",
    )

    # 7. Create GRPO trainer
    print("\nCreating GRPO trainer...")

    # GRPO has built-in multimodal support
    # For vision models, use processing_class (includes tokenizer + image processor)
    trainer_kwargs = {
        "model": model,
        "processing_class": tokenizer,  # For vision models, this includes image processing
        "args": training_args,
        "train_dataset": train_dataset,
        "reward_funcs": [reward_fn],  # Unsloth expects a list of reward functions
    }

    # Add vision-specific parameters if specified
    if config.max_pixels is not None:
        trainer_kwargs["max_pixels"] = config.max_pixels
    if config.min_pixels is not None:
        trainer_kwargs["min_pixels"] = config.min_pixels

    trainer = GRPOTrainer(**trainer_kwargs)

    # 8. Train!
    print("\n" + "="*60)
    print("Starting GRPO training...")
    print("="*60)

    trainer.train()

    # 9. Save the LoRA adapters with unique identifier
    print("\nSaving LoRA adapters...")

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = config.model_name.split('/')[-1]

    run_name_parts = [
        model_short_name,
        "grpo",  # Indicate this is RL training
        timestamp,
        f"samples{config.num_samples if config.num_samples else 'all'}",
        f"epochs{config.num_train_epochs}",
        f"lr{config.learning_rate}",
        f"r{config.lora_r}",
        f"numgen{config.num_generations}",
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
        "training_method": "grpo",
        "num_samples": config.num_samples,
        "num_train_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "num_generations": config.num_generations,
        "max_prompt_length": config.max_prompt_length,
        "max_completion_length": config.max_completion_length,
        "temperature": config.temperature,
        "beta": config.beta,
        "batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "timestamp": timestamp,
    }
    with open(str(final_model_path / "training_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    checkpoint_volume.commit()

    print("\n" + "="*60)
    print("GRPO Training completed!")
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
    model: str = "Qwen/Qwen3-VL-8B-Thinking",
    num_samples: int = None,
    epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation: int = 2,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    num_generations: int = 4,
    max_prompt_length: int = 512,
    max_completion_length: int = 1024,
    temperature: float = 0.7,
    beta: float = 0.1,
    max_pixels: int = None,
    min_pixels: int = None,
    use_4bit: bool = True,
    use_wandb: bool = False,
    run_name: str = None,
):
    """
    Local entry point to start GRPO fine-tuning.

    Usage:
        # Basic GRPO fine-tuning with Qwen3-VL-8B on small subset
        modal run scripts/modal_qwen_finetune_rl.py --num-samples 100 --epochs 3

        # GRPO with more exploration (higher temperature and generations)
        modal run scripts/modal_qwen_finetune_rl.py \\
            --num-samples 1000 \\
            --epochs 5 \\
            --num-generations 8 \\
            --temperature 0.9 \\
            --beta 0.05 \\
            --run-name "high_exploration"

        # Full GRPO training with W&B tracking
        modal run scripts/modal_qwen_finetune_rl.py \\
            --epochs 3 \\
            --use-wandb \\
            --run-name "grpo_full_v1"
    """
    print(f"Starting GRPO fine-tuning with {model}...")
    print(f"Samples: {num_samples if num_samples else 'all'}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size} (effective: {batch_size * gradient_accumulation})")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA rank: {lora_r}")
    print(f"Num generations per prompt: {num_generations}")
    print(f"Max prompt length: {max_prompt_length}")
    print(f"Max completion length: {max_completion_length}")
    print(f"Temperature: {temperature}")
    print(f"Beta (KL coefficient): {beta}")
    print(f"4-bit quantization: {use_4bit}")

    # Create config
    config = GRPOTrainingConfig(
        model_name=model,
        num_samples=num_samples,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        lora_r=lora_r,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        temperature=temperature,
        beta=beta,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        load_in_4bit=use_4bit,
        use_wandb=use_wandb,
        wandb_run_name=run_name,
    )

    # Run GRPO fine-tuning
    final_path = finetune_qwen_grpo.remote(config)

    print("\n" + "="*60)
    print("GRPO fine-tuning job completed!")
    print(f"Model saved at: {final_path}")
    print("\nNext steps:")
    print("1. Evaluate the model using modal_qwen_evaluate_adapter.py")
    print("2. Compare performance with supervised fine-tuning")
    print("3. Try different hyperparameters (num_generations, temperature, kl_coef)")
    print("="*60)
