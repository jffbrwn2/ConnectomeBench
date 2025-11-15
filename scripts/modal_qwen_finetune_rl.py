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
    # Install core dependencies for training
    .pip_install(
        "torch",  # PyTorch
        "torchvision",  # Required for Qwen-VL video/image processor
        "transformers",  # HuggingFace transformers
        "peft",  # Parameter-Efficient Fine-Tuning (LoRA)
        "bitsandbytes",  # 8-bit optimizers
        "trl==0.25.1",  # Latest TRL with GRPO support
        "datasets",
        "pandas",
        "Pillow",
        "huggingface_hub[hf_transfer]",
        "wandb",
        "accelerate",
    )
    # Install vLLM separately with TRL-compatible version (Modal's mirror has old versions)
    .pip_install(
        "vllm==0.10.2",  # High-throughput inference engine (TRL-recommended version)
        extra_index_url="https://pypi.org/simple",  # Use main PyPI for this version
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

    # Note: 4-bit quantization not needed for 8B model on H100

    # GRPO-specific hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4  # Should be multiple of num_generations
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4

    # GRPO generation settings
    num_generations: int = 4  # Generate multiple responses per prompt (batch_size must be multiple of this)
    max_prompt_length: int = None  # None to avoid truncating image tokens!
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
    balanced_sampling: bool = False  # Use stratified sampling for balanced class distribution

    # Checkpoint settings
    resume_from_checkpoint: str = None  # Path to checkpoint to resume from (e.g., "checkpoint-100")

    # Vision-specific settings (for Qwen-VL)
    max_pixels: int = None  # Maximum image pixels (None = use model default)
    min_pixels: int = None  # Minimum image pixels (None = use model default)

    # W&B settings
    use_wandb: bool = False
    wandb_project: str = "qwen-segment-rl"
    wandb_run_name: str = None

    # vLLM settings (for faster generation)
    use_vllm: bool = False  # Enable vLLM for high-throughput generation
    vllm_mode: str = "colocate"  # "colocate" or "server" - colocate shares GPU with training
    vllm_gpu_memory_utilization: float = 0.3  # GPU memory reserved for vLLM (colocate mode)
    vllm_tensor_parallel_size: int = 1  # Number of GPUs for tensor parallelism

    # Debug settings
    debug_simple_prompt: bool = False  # Use simple "What do you see?" prompt for debugging


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


def stratified_sample(dataset, num_samples, label_column="ground_truth"):
    """
    Perform stratified sampling to get balanced class distribution.

    Args:
        dataset: HuggingFace dataset
        num_samples: Total number of samples to select
        label_column: Column name containing class labels

    Returns:
        Dataset with balanced sampling across classes
    """
    from collections import Counter
    import numpy as np

    # Get all labels
    all_labels = dataset[label_column]
    label_counts = Counter(all_labels)
    unique_labels = list(label_counts.keys())

    print(f"\nClass distribution (before sampling):")
    for label, count in sorted(label_counts.items()):
        label_key = REVERSE_CLASS_MAPPING.get(label, "?")
        print(f"  {label_key}: {count} samples")

    # Calculate samples per class
    num_classes = len(unique_labels)
    samples_per_class = num_samples // num_classes
    remainder = num_samples % num_classes

    # Group indices by label
    label_to_indices = {label: [] for label in unique_labels}
    for idx, label in enumerate(all_labels):
        label_to_indices[label].append(idx)

    # Sample from each class
    selected_indices = []
    for i, label in enumerate(unique_labels):
        indices = label_to_indices[label]
        # Add 1 extra sample to first 'remainder' classes to reach exact num_samples
        n_to_sample = samples_per_class + (1 if i < remainder else 0)
        n_to_sample = min(n_to_sample, len(indices))  # Don't sample more than available

        sampled = np.random.choice(indices, size=n_to_sample, replace=False).tolist()
        selected_indices.extend(sampled)

    # Shuffle the combined indices
    np.random.shuffle(selected_indices)

    # Select the samples
    sampled_dataset = dataset.select(selected_indices)

    # Print new distribution
    sampled_labels = sampled_dataset[label_column]
    sampled_counts = Counter(sampled_labels)
    print(f"\nClass distribution (after balanced sampling):")
    for label, count in sorted(sampled_counts.items()):
        label_key = REVERSE_CLASS_MAPPING.get(label, "?")
        print(f"  {label_key}: {count} samples")

    return sampled_dataset


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


def prepare_dataset_for_grpo(dataset, processor, debug_simple_prompt=False):
    """
    Prepare dataset for GRPO training with vision models.

    GRPOTrainer expects:
    - 'prompt': List of messages with role/content (content can be simple strings)
    - 'images': List of PIL images (passed to prepare_multimodal_messages)
    - Additional columns for reward function (e.g., 'ground_truth')

    Args:
        dataset: HuggingFace dataset
        processor: Model processor (not used, kept for API compatibility)
        debug_simple_prompt: If True, use simple "What do you see?" prompt for debugging

    Returns:
        Dataset with 'prompt', 'images', and 'ground_truth' columns
    """
    import numpy as np

    def format_sample(sample):
        """Convert a single sample to GRPO format"""
        species = sample['species']
        ground_truth = sample['ground_truth']

        # Get bounding box dimensions
        xmin, ymin, zmin = sample['xmin'], sample['ymin'], sample['zmin']
        xmax, ymax, zmax = sample['xmax'], sample['ymax'], sample['zmax']
        box_size = np.array([xmax - xmin, ymax - ymin, zmax - zmin])

        # Get the three images (PIL Images)
        images = [
            sample['option_1_front_path'],
            sample['option_1_side_path'],
            sample['option_1_top_path']
        ]

        # Create instruction prompt
        if debug_simple_prompt:
            prompt_text = "What do you see in the images?"
        else:
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

        # Simple message format - GRPOTrainer calls prepare_multimodal_messages internally
        prompt_messages = [{"role": "user", "content": prompt_text}]

        return {
            "prompt": prompt_messages,
            "images": images,  # Use 'images' (plural) for multiple images
            "ground_truth": ground_truth,
        }

    # Use set_transform to format on-the-fly (keeps PIL Images in memory)
    def transform_fn(batch):
        batch_size = len(batch[next(iter(batch.keys()))])

        prompts = []
        images_list = []
        ground_truths = []

        for i in range(batch_size):
            sample = {k: batch[k][i] for k in batch.keys()}
            formatted = format_sample(sample)

            prompts.append(formatted["prompt"])
            images_list.append(formatted["images"])
            ground_truths.append(formatted["ground_truth"])

        return {
            "prompt": prompts,
            "images": images_list,
            "ground_truth": ground_truths,
        }

    dataset.set_transform(transform_fn)
    return dataset


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
    from datasets import load_dataset

    # Load model with transformers + PEFT for vision support
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import GRPOConfig, GRPOTrainer

    print("\n" + "="*60)
    print("Qwen-VL GRPO Fine-tuning")
    print("="*60)

    # Set cache directories
    os.environ["HF_HOME"] = str(MODEL_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_DIR)

    # Configure wandb environment
    if config.use_wandb:
        os.environ["WANDB_PROJECT"] = config.wandb_project
        os.environ["WANDB_LOG_MODEL"] = "false"  # Don't auto-upload model checkpoints

    # Initialize W&B if requested
    if config.use_wandb:
        import wandb
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"{config.model_name.split('/')[-1]}-grpo",
            config=config.__dict__,
            resume="allow",  # Allow resuming runs
        )
        print(f"Weights & Biases run: {wandb.run.name} (ID: {wandb.run.id})")
        print(f"Dashboard: {wandb.run.url}")

    # 1. Load processor (tokenizer + image processor)
    print(f"\nLoading processor...")
    processor = AutoProcessor.from_pretrained(config.model_name)

    # Configure image processor settings if specified
    if hasattr(processor, 'image_processor'):
        if config.max_pixels is not None:
            processor.image_processor.max_pixels = config.max_pixels
        if config.min_pixels is not None:
            processor.image_processor.min_pixels = config.min_pixels

    # Set padding side for decoder-only models
    processor.tokenizer.padding_side = 'left'
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    # 2. Load model in bfloat16
    print(f"\nLoading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16,
    )
    model = model.to("cuda")
    model.gradient_checkpointing_enable()

    # 3. Add LoRA adapters with PEFT

    # Configure LoRA - inference_mode=False ensures adapters are trainable
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,  # Explicitly set for training
        # Target all attention and MLP modules for comprehensive fine-tuning
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",      # MLP
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(
        "jeffbbrown2/ConnectomeBench",
        "MICrONS, Segment Classification",
        split="train",
        cache_dir=str(DATASET_DIR)
    )

    # Filter out unwanted classes (e: non-neuronal, f: none of the above)
    print(f"Original dataset size: {len(dataset)}")
    dataset = dataset.filter(lambda x: x['ground_truth'] not in [
        CLASS_MAPPING['e'],  # Non-neuronal types
        CLASS_MAPPING['f'],   # None of the above
        CLASS_MAPPING['g'],   # Unsure
    ])
    print(f"Filtered dataset size: {len(dataset)} (excluded classes e, f)")

    # Limit samples if specified
    if config.num_samples is not None and config.num_samples < len(dataset):
        if config.balanced_sampling:
            print(f"Using stratified sampling for {config.num_samples} balanced samples")
            dataset = stratified_sample(dataset, config.num_samples)
        else:
            dataset = dataset.select(range(config.num_samples))
            print(f"Using {config.num_samples} samples (first N)")
    else:
        print(f"Using all {len(dataset)} samples")

    # 4. Prepare dataset for GRPO
    train_dataset = prepare_dataset_for_grpo(dataset, processor, config.debug_simple_prompt)
    if config.debug_simple_prompt:
        print("DEBUG MODE: Simple prompt enabled")

    # 5. Create reward function
    reward_fn = create_reward_function()

    # 6. Setup GRPO training arguments
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

        # Precision - use bf16 if available (H100 supports it)
        fp16=False,  # Don't use fp16 with bfloat16 hardware
        bf16=True,   # H100 supports bfloat16

        # Logging and saving
        logging_steps=config.logging_steps,
        logging_first_step=True,  # Log metrics on first step
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        log_completions=True,  # Log generated completions for debugging
        log_level="info",  # Enable info-level logging

        # vLLM settings for faster generation
        use_vllm=config.use_vllm,
        vllm_mode=config.vllm_mode,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=config.vllm_tensor_parallel_size,

        # Other
        seed=config.seed,
        report_to="wandb" if config.use_wandb else "none",
    )

    if config.use_vllm:
        print(f"vLLM enabled ({config.vllm_mode}, {config.vllm_gpu_memory_utilization:.0%} GPU)")

    # 7. Create GRPO trainer with checkpoint callback
    from transformers import TrainerCallback

    class VolumeCommitCallback(TrainerCallback):
        """Commit volume after each checkpoint save"""
        def on_save(self, args, state, control, **kwargs):
            print("Committing checkpoint to volume...")
            checkpoint_volume.commit()

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=[reward_fn],
        callbacks=[VolumeCommitCallback()],
    )

    # 8. Train!
    if config.resume_from_checkpoint:
        checkpoint_path = CHECKPOINT_DIR / config.model_name.replace("/", "_") / config.resume_from_checkpoint
        print(f"\nResuming training from checkpoint: {checkpoint_path}")
        trainer.train(resume_from_checkpoint=str(checkpoint_path))
    else:
        print("\nStarting training...")
        trainer.train()

    # 9. Save the LoRA adapters
    print("\nSaving model...")

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
    processor.save_pretrained(str(final_model_path))

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

    print(f"\nTraining complete!")
    print(f"Model saved: {run_folder_name}")

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
    max_prompt_length: int = None,  # None to avoid truncating image tokens
    max_completion_length: int = 1024,
    temperature: float = 0.7,
    beta: float = 0.1,
    max_pixels: int = None,
    min_pixels: int = None,
    use_wandb: bool = False,
    run_name: str = None,
    use_vllm: bool = False,
    vllm_mode: str = "colocate",
    vllm_gpu_memory: float = 0.3,
    balanced_sampling: bool = False,
    resume_from_checkpoint: str = None,
    debug_simple_prompt: bool = False,
):
    """
    Local entry point to start GRPO fine-tuning.

    Usage:
        # Debug mode with simple prompt (test if images are working)
        modal run scripts/modal_qwen_finetune_rl.py --num-samples 10 --epochs 1 --debug-simple-prompt

        # Basic GRPO fine-tuning with balanced class sampling
        modal run scripts/modal_qwen_finetune_rl.py --num-samples 100 --epochs 3 --balanced-sampling

        # With vLLM for faster generation (recommended!)
        modal run scripts/modal_qwen_finetune_rl.py \\
            --num-samples 100 \\
            --epochs 3 \\
            --balanced-sampling \\
            --use-vllm

        # GRPO with more exploration (higher temperature and generations)
        modal run scripts/modal_qwen_finetune_rl.py \\
            --num-samples 1000 \\
            --epochs 5 \\
            --num-generations 8 \\
            --temperature 0.9 \\
            --beta 0.05 \\
            --balanced-sampling \\
            --use-vllm \\
            --run-name "high_exploration"

        # Full GRPO training with W&B tracking
        modal run scripts/modal_qwen_finetune_rl.py \\
            --epochs 3 \\
            --use-vllm \\
            --use-wandb \\
            --run-name "grpo_full_v1"

        # Resume from checkpoint
        modal run scripts/modal_qwen_finetune_rl.py \\
            --resume-from-checkpoint "checkpoint-500"
    """
    print(f"Starting GRPO training: {model}")
    print(f"Samples: {num_samples or 'all'}{' (balanced)' if balanced_sampling else ''} | Epochs: {epochs} | Batch: {batch_size}x{gradient_accumulation}")
    print(f"LR: {learning_rate} | LoRA-r: {lora_r} | Generations: {num_generations}")
    if use_vllm:
        print(f"vLLM: {vllm_mode} ({vllm_gpu_memory:.0%})")
    if debug_simple_prompt:
        print("DEBUG: Simple prompt mode")

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
        use_wandb=use_wandb,
        wandb_run_name=run_name,
        use_vllm=use_vllm,
        vllm_mode=vllm_mode,
        vllm_gpu_memory_utilization=vllm_gpu_memory,
        balanced_sampling=balanced_sampling,
        resume_from_checkpoint=resume_from_checkpoint,
        debug_simple_prompt=debug_simple_prompt,
    )

    # Run GRPO fine-tuning
    final_path = finetune_qwen_grpo.remote(config)
    print(f"\nJob complete: {final_path}")
