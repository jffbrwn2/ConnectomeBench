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

# Define the Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # Required for installing unsloth from GitHub
    # Install unsloth - simple installation as recommended by unsloth docs
    .pip_install(
        "unsloth",
        "unsloth_zoo",
    )
    # Then install additional dependencies
    .pip_install(
        "datasets",
        "pandas",
        "Pillow",
        "huggingface_hub[hf_transfer]",
        "wandb",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("qwen-segment-finetune", image=image)

# Also make the app available for the evaluation script to import
__all__ = ["app", "image"]


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning"""
    # Model settings
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    max_seq_length: int = 2048

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list = None  # None = auto-detect

    # Quantization
    load_in_4bit: bool = True  # Use QLoRA for memory efficiency

    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    gpu_count: int = 2  # Number of GPUs to use
    warmup_steps: int = 5
    max_steps: int = -1  # -1 means use num_train_epochs

    # Optimization
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"

    # Other settings
    logging_steps: int = 1
    save_steps: int = 100
    save_total_limit: int = 3
    fp16: bool = False
    bf16: bool = True
    seed: int = 42

    # Dataset settings
    num_samples: int = None  # None = use all data
    test_split_ratio: float = 0.2  # Fraction of data to hold out for evaluation

    # Evaluation settings
    eval_strategy: str = "steps"  # "steps", "epoch", or "no"
    eval_steps: int = 50  # How often to run evaluation (if strategy is "steps")

    # W&B settings
    use_wandb: bool = False
    wandb_project: str = "qwen-segment-classification"
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

# Reverse mapping for creating ground truth answers
REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}


def convert_sample_to_conversation(sample, use_teacher_responses=False, teacher_response_column='teacher_analysis'):
    """
    Convert a ConnectomeBench sample to the conversation format expected by Unsloth.

    Args:
        sample: A sample from the ConnectomeBench dataset
        use_teacher_responses: If True and sample has teacher response, use that response
        teacher_response_column: Column name containing teacher model's response

    Returns:
        dict with 'messages' key containing conversation in ChatML format
    """
    import numpy as np
    import pandas as pd

    # Get sample metadata
    species = sample['species']
    ground_truth = sample['ground_truth']

    # Get bounding box dimensions
    xmin, ymin, zmin = sample['xmin'], sample['ymin'], sample['zmin']
    xmax, ymax, zmax = sample['xmax'], sample['ymax'], sample['zmax']
    box_size = np.array([xmax - xmin, ymax - ymin, zmax - zmin])

    # Get the three images (PIL Images)
    front_img = sample['option_1_front_path']
    side_img = sample['option_1_side_path']
    top_img = sample['option_1_top_path']

    # Create the instruction prompt (with multiple choice and tags)
    prompt = f"""You are an expert at analyzing neuronal morphology.

We have the electron microscopy data from the {species} brain.

In the images, we have a selected 3D segmentation that is supposed to correspond to a complete neuronal structure. However, it could have split/merge errors as the segmentation algorithm makes mistakes.

The 3D snapshots are three different views of the same segment. The dimensions of the segment's bounding box are {box_size[0]} x {box_size[1]} x {box_size[2]} nm. Describe in detail what you see using the information in the 3D snapshots. Is the segment a neuron (soma and processes)? Multiple neurons merged together (multiple somas)? Processes like axon and dendrites without a cell body? Non-neuronal structures like glia, astrocytes, or blood vessels? Inspect very closely to avoid making errors, using the 3D views and size of the bounding box in your reasoning.

For {species} neurons, the somas tend to be round and generally {'a single process extends' if species == 'fly' else 'multiple processes extend'} from them {'before it branches into many processes' if species == 'fly' else 'outwards'}. Processes can be axons or dendrites, long and often branching. Synapses can also be considered as a part of processes, and these are often small segments (often smaller than a cubic micron). The nucleuses are round and do not have any processes extending from them. Blood vessels are tubular and obviously do not have any processes extending from them. Glial cells lack the branching processes of neurons, and instead appear like jagged masses.

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

    # Get the correct answer letter
    answer_key = REVERSE_CLASS_MAPPING.get(ground_truth, "g")

    # Determine response based on mode
    if use_teacher_responses and teacher_response_column in sample and sample[teacher_response_column] is not None and pd.notna(sample[teacher_response_column]):
        # Use teacher model's analysis and format with correct answer
        teacher_analysis = sample[teacher_response_column]
        assistant_response = f"""<analysis>
{teacher_analysis}
</analysis>

<answer>{answer_key}</answer>"""
    else:
        # Create synthetic response with ground truth
        assistant_response = f"""<analysis>
Based on the three orthogonal views of this segment, I can analyze its structure.

The bounding box dimensions of {box_size[0]} x {box_size[1]} x {box_size[2]} nm provide important scale information.

Looking at the morphology across all three views, this segment appears to be: {ground_truth}
</analysis>

<answer>{answer_key}</answer>"""

    # Format as conversation (ChatML format for Qwen)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": front_img},
                {"type": "image", "image": side_img},
                {"type": "image", "image": top_img},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": assistant_response,
        }
    ]

    return {"messages": messages}


class AccuracyCallback:
    """Custom callback to compute classification accuracy during training evaluation."""

    def __init__(self, eval_dataset, model, tokenizer, class_mapping, max_eval_samples=50):
        self.eval_dataset = eval_dataset
        self.model = model
        self.tokenizer = tokenizer
        self.class_mapping = class_mapping
        self.max_eval_samples = max_eval_samples

    # Implement all 15 TrainerCallback methods
    def on_init_end(self, args, state, control, **kwargs):
        return control
    def on_train_begin(self, args, state, control, **kwargs):
        return control
    def on_train_end(self, args, state, control, **kwargs):
        return control
    def on_epoch_begin(self, args, state, control, **kwargs):
        return control
    def on_epoch_end(self, args, state, control, **kwargs):
        return control
    def on_step_begin(self, args, state, control, **kwargs):
        return control
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        return control
    def on_optimizer_step(self, args, state, control, **kwargs):
        return control
    def on_substep_end(self, args, state, control, **kwargs):
        return control
    def on_step_end(self, args, state, control, **kwargs):
        return control
    def on_predict(self, args, state, control, **kwargs):
        return control
    def on_save(self, args, state, control, **kwargs):
        return control
    def on_log(self, args, state, control, **kwargs):
        return control
    def on_prediction_step(self, args, state, control, **kwargs):
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Compute classification accuracy on eval set."""
        import torch
        import numpy as np

        print(f"\n{'='*60}")
        print("Computing classification accuracy...")

        num_samples = min(len(self.eval_dataset), self.max_eval_samples)
        eval_subset = self.eval_dataset.select(range(num_samples))
        correct = 0
        total = 0
        self.model.eval()

        # Process in batches (copied from evaluate_adapter logic)
        for batch_start in range(0, num_samples, 2):
            batch_end = min(batch_start + 2, num_samples)
            batch = eval_subset.select(range(batch_start, batch_end))

            batch_texts = []
            batch_images = []
            ground_truths = []

            try:
                # Iterate directly over batch samples
                for sample in batch:
                    # Get images (they're PIL Image objects in the dataset)
                    front_img = sample['option_1_front_path']
                    side_img = sample['option_1_side_path']
                    top_img = sample['option_1_top_path']
                    ground_truth = sample['ground_truth']
                    ground_truths.append(ground_truth)

                    species = sample['species']
                    xmin, ymin, zmin = sample['xmin'], sample['ymin'], sample['zmin']
                    xmax, ymax, zmax = sample['xmax'], sample['ymax'], sample['zmax']
                    box_size = np.array([xmax - xmin, ymax - ymin, zmax - zmin])

                    prompt = f"""You are an expert at analyzing neuronal morphology.

We have the electron microscopy data from the {species} brain.

In the images, we have a selected 3D segmentation that is supposed to correspond to a complete neuronal structure. However, it could have split/merge errors as the segmentation algorithm makes mistakes.

The 3D snapshots are three different views of the same segment. The dimensions of the segment's bounding box are {box_size[0]} x {box_size[1]} x {box_size[2]} nm. Describe in detail what you see using the information in the 3D snapshots. Is the segment a neuron (soma and processes)? Multiple neurons merged together (multiple somas)? Processes like axon and dendrites without a cell body? Non-neuronal structures like glia, astrocytes, or blood vessels? Inspect very closely to avoid making errors, using the 3D views and size of the bounding box in your reasoning.

For {species} neurons, the somas tend to be round and generally {'a single process extends' if species == 'fly' else 'multiple processes extend'} from them {'before it branches into many processes' if species == 'fly' else 'outwards'}. Processes can be axons or dendrites, long and often branching. Synapses can also be considered as a part of processes, and these are often small segments (often smaller than a cubic micron). The nucleuses are round and do not have any processes extending from them. Blood vessels are tubular and obviously do not have any processes extending from them. Glial cells lack the branching processes of neurons, and instead appear like jagged masses.

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

                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": front_img},
                            {"type": "image", "image": side_img},
                            {"type": "image", "image": top_img},
                            {"type": "text", "text": prompt},
                        ],
                    }]

                    text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    batch_texts.append(text)
                    batch_images.extend([front_img, side_img, top_img])

                # Generate prediction
                inputs = self.tokenizer(text=batch_texts, images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
                    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
                    output_texts = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                # Extract answer
                for output_text, ground_truth in zip(output_texts, ground_truths):
                    answer_start = output_text.find("<answer>")
                    answer_end = output_text.find("</answer>")
                    if answer_start != -1 and answer_end != -1:
                        llm_answer = output_text[answer_start + len("<answer>"):answer_end].strip()
                        predicted_description = self.class_mapping.get(llm_answer, None)
                        if predicted_description == ground_truth:
                            correct += 1
                    total += 1
            except Exception as e:
                # Print error details instead of silently skipping
                print(f"Error processing batch: {e}")
                import traceback
                traceback.print_exc()
                continue

        accuracy = correct / total if total > 0 else 0
        if metrics is not None:
            metrics['eval_classification_accuracy'] = accuracy

        print(f"Classification Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"{'='*60}\n")
        self.model.train()
        return control


@app.function(
    gpu="A100-40GB",  # ⚠️ MUST match config.gpu_count parameter
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
def finetune_qwen(
    config: TrainingConfig = TrainingConfig(),
    augmented_dataset_path: str = None,
    use_teacher_responses: bool = False,
    teacher_response_column: str = 'teacher_analysis',
    teacher_model_name: str = None
):
    """
    Fine-tune Qwen-VL model on segment classification task.

    Args:
        config: TrainingConfig object with all training parameters
        augmented_dataset_path: Optional path to parquet file with teacher responses
        use_teacher_responses: If True, use teacher model responses from augmented dataset
        teacher_response_column: Column name containing teacher model's responses
        teacher_model_name: Optional name of teacher model for tracking
    """
    import torch
    import os
    import pandas as pd
    from unsloth import FastVisionModel
    from datasets import load_dataset, Dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth import is_bf16_supported

    # Assert GPU count matches decorator configuration
    num_gpus_available = torch.cuda.device_count()
    assert num_gpus_available == config.gpu_count, (
        f"GPU count mismatch! Detected {num_gpus_available} GPU(s) but config.gpu_count={config.gpu_count}. "
        f"Update @app.function decorator to: gpu=modal.gpu.A100(count={config.gpu_count})"
    )

    print("="*60)
    if use_teacher_responses:
        print(f"Starting Qwen-VL Fine-tuning with Teacher Responses")
        if teacher_model_name:
            print(f"Teacher Model: {teacher_model_name}")
    else:
        print("Starting Qwen-VL Fine-tuning")
    print(f"GPU Configuration: {config.gpu_count}x A100")
    effective_batch_size = config.per_device_train_batch_size * config.gpu_count * config.gradient_accumulation_steps
    print(f"Effective batch size: {effective_batch_size} ({config.per_device_train_batch_size} per device * {config.gpu_count} GPUs * {config.gradient_accumulation_steps} grad accum)")
    print("="*60)

    # Set cache directories
    os.environ["HF_HOME"] = str(MODEL_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_DIR)

    # Initialize W&B if requested
    if config.use_wandb:
        import wandb
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"{config.model_name.split('/')[-1]}-finetune",
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
        finetune_vision_layers=True,  # Fine-tune vision encoder
        finetune_language_layers=True,  # Fine-tune language model
        finetune_attention_modules=True,  # Fine-tune attention
        finetune_mlp_modules=True,  # Fine-tune MLPs

        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        random_state=config.seed,
        use_rslora=False,
        loftq_config=None,
    )

    # 3. Load dataset
    if augmented_dataset_path and use_teacher_responses:
        print(f"\nLoading augmented dataset from: {augmented_dataset_path}")
        df = pd.read_parquet(augmented_dataset_path)
        print(f"  Loaded {len(df)} samples")

        # Filter to only correct responses if correctness column exists
        # The merge script always creates 'teacher_correct' column
        if 'teacher_correct' in df.columns:
            before_count = len(df)
            df = df[df['teacher_correct'] == True].copy()
            print(f"  Filtered to {len(df)} samples with correct responses (removed {before_count - len(df)} incorrect)")
        else:
            print(f"  Note: 'teacher_correct' column not found, using all samples")

        # Verify samples have teacher responses
        has_responses = df[teacher_response_column].notna().sum()
        print(f"  Samples with teacher responses: {has_responses}/{len(df)}")

        # Show teacher model breakdown if available
        if 'teacher_model' in df.columns:
            print(f"\n  Teacher model breakdown:")
            for model_name, count in df['teacher_model'].value_counts().items():
                print(f"    {model_name}: {count} samples")

        # Convert DataFrame to HuggingFace Dataset
        dataset = Dataset.from_pandas(df)
    else:
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
        print(f"Limited to {config.num_samples} samples")
    else:
        print(f"Using all {len(dataset)} samples")

    # Split into train and test sets
    if config.test_split_ratio > 0:
        print(f"\nSplitting dataset with test_split_ratio={config.test_split_ratio}...")
        split_dataset = dataset.train_test_split(
            test_size=config.test_split_ratio,
            seed=config.seed,
            shuffle=True
        )
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Eval samples: {len(eval_dataset)}")
    else:
        print("\nNo train/test split (test_split_ratio=0)")
        train_dataset = dataset
        eval_dataset = None
        print(f"  Train samples: {len(train_dataset)}")

    # 4. Define formatting function that converts raw dataset samples to chat format
    def formatting_func(examples):
        """
        Format sample(s) from the dataset into the chat template format.
        This handles both single samples (during testing) and batches (during training).
        Uses teacher responses if use_teacher_responses is True.
        """
        from PIL import Image

        # Check if this is a single sample or a batch by checking if 'species' is a list
        is_batch = isinstance(examples['species'], list)

        if not is_batch:
            # Single sample case - wrap in lists to process uniformly
            examples = {key: [value] for key, value in examples.items()}

        texts = []
        batch_size = len(examples['species'])

        # Process each sample in the batch
        for i in range(batch_size):
            # Extract single sample from batch
            sample = {key: examples[key][i] for key in examples.keys()}

            # Convert the raw sample to conversation format
            conversation = convert_sample_to_conversation(
                sample,
                use_teacher_responses=use_teacher_responses,
                teacher_response_column=teacher_response_column
            )

            # Apply chat template to convert messages to text
            text = tokenizer.apply_chat_template(
                conversation["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        # Return list of formatted texts (or single item list for single sample)
        return texts

    # 5. Setup training arguments
    print("\nSetting up training...")
    training_args = SFTConfig(
        output_dir=str(CHECKPOINT_DIR / config.model_name.replace("/", "_")),

        # Training hyperparameters
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,

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

        # Evaluation
        eval_strategy=config.eval_strategy if eval_dataset is not None else "no",
        eval_steps=config.eval_steps if config.eval_strategy == "steps" else None,
        per_device_eval_batch_size=config.per_device_train_batch_size,

        # Other
        seed=config.seed,
        report_to="wandb" if config.use_wandb else "none",
    )

    # 6. Create trainer with accuracy callback
    callbacks = []
    if eval_dataset is not None:
        accuracy_callback = AccuracyCallback(
            eval_dataset=eval_dataset,
            model=model,
            tokenizer=tokenizer,
            class_mapping=CLASS_MAPPING,
            max_eval_samples=50
        )
        callbacks.append(accuracy_callback)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Will be None if test_split_ratio=0
        formatting_func=formatting_func,  # Required by Unsloth
        max_seq_length=config.max_seq_length,
        callbacks=callbacks,
    )

    # 7. Train!
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    trainer.train()

    # 8. Save the LoRA adapters with unique identifier
    print("\nSaving LoRA adapters...")

    # Create unique save path with timestamp and training info
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = config.model_name.split('/')[-1]  # e.g., "Qwen3-VL-8B-Instruct"

    # Build descriptive folder name
    run_name_parts = [
        model_short_name,
        timestamp,
        f"samples{config.num_samples if config.num_samples else 'all'}",
        f"epochs{config.num_train_epochs}",
        f"lr{config.learning_rate}",
        f"r{config.lora_r}",
    ]

    # Add W&B run name if specified
    if config.wandb_run_name:
        run_name_parts.insert(1, config.wandb_run_name)

    run_folder_name = "_".join(run_name_parts)
    final_model_path = CHECKPOINT_DIR / run_folder_name

    model.save_pretrained(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    # Save training config for reference
    import json
    config_dict = {
        "model_name": config.model_name,
        "num_samples": config.num_samples,
        "num_train_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "timestamp": timestamp,
        "use_teacher_responses": use_teacher_responses,
    }
    if augmented_dataset_path:
        config_dict["augmented_dataset_path"] = augmented_dataset_path
        config_dict["teacher_response_column"] = teacher_response_column
    if teacher_model_name:
        config_dict["teacher_model_name"] = teacher_model_name
    with open(str(final_model_path / "training_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    checkpoint_volume.commit()

    print("\n" + "="*60)
    print("Training completed!")
    print(f"LoRA adapters saved to: {final_model_path}")
    print(f"Size: ~50-200 MB (adapters only)")
    print(f"\nTo use for inference:")
    print(f"  1. Load base model: FastVisionModel.from_pretrained('{config.model_name}')")
    print(f"  2. Load adapters: FastVisionModel.load_adapter(model, '{final_model_path}')")
    print("="*60)

    if config.use_wandb:
        wandb.finish()

    return str(final_model_path)


@app.function(
    gpu="A100",
    timeout=3600,  # 1 hour
    volumes={
        MODEL_DIR: model_volume,
        DATASET_DIR: dataset_volume,
        CHECKPOINT_DIR: checkpoint_volume,
        RESULTS_DIR: results_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def evaluate_adapter(
    adapter_path: str,
    num_samples: int = None,
    batch_size: int = 4,
    use_blank_images: bool = False,
    use_simple_prompt: bool = False,
):
    """
    Evaluate a fine-tuned LoRA adapter on segment classification task.

    Args:
        adapter_path: Path to the LoRA adapter folder
        num_samples: Number of samples to evaluate (None = all)
        batch_size: Batch size for inference
        use_blank_images: Use blank images for sanity check
        use_simple_prompt: Use simple prompt for sanity check
    """
    import torch
    import pandas as pd
    import numpy as np
    import json
    from datasets import load_dataset
    from PIL import Image
    from unsloth import FastVisionModel
    from peft import PeftModel

    print("="*60)
    print("Evaluating Fine-tuned LoRA Adapter")
    print("="*60)

    # Set cache directories
    import os
    os.environ["HF_HOME"] = str(MODEL_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_DIR)

    # Load training config to get base model name
    full_adapter_path = CHECKPOINT_DIR / adapter_path
    config_path = full_adapter_path / "training_config.json"

    if config_path.exists():
        with open(str(config_path), "r") as f:
            training_config = json.load(f)
        base_model_name = training_config["model_name"]
        print(f"\nTraining config found:")
        print(f"  Base model: {base_model_name}")
        print(f"  Trained on: {training_config.get('num_samples', 'all')} samples")
        print(f"  Epochs: {training_config.get('num_train_epochs')}")
    else:
        print(f"\nWarning: No training_config.json found")
        base_model_name = "Qwen/Qwen3-VL-8B-Instruct"

    # Load base model + adapters
    print(f"\nLoading base model: {base_model_name}...")
    from transformers import AutoProcessor

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=base_model_name,
        load_in_4bit=True,
        max_seq_length=2048,
    )

    # Also load the processor for proper image handling
    processor = AutoProcessor.from_pretrained(base_model_name)

    print(f"Loading LoRA adapters from: {full_adapter_path}...")
    model = PeftModel.from_pretrained(model, str(full_adapter_path))
    FastVisionModel.for_inference(model)

    # Load dataset
    print("\nLoading ConnectomeBench dataset...")
    ds = load_dataset(
        "jeffbbrown2/ConnectomeBench",
        "MICrONS, Segment Classification",
        split="train"
    )

    if num_samples is not None and num_samples < len(ds):
        ds = ds.select(range(num_samples))
        print(f"Evaluating on {num_samples} samples")
    else:
        print(f"Evaluating on all {len(ds)} samples")

    results = []

    # Process in batches for speed
    print(f"Processing with batch_size={batch_size}...")
    for batch_start in range(0, len(ds), batch_size):
        batch_end = min(batch_start + batch_size, len(ds))
        batch = ds.select(range(batch_start, batch_end))

        print(f"Processing batch {batch_start//batch_size + 1}/{(len(ds) + batch_size - 1)//batch_size} (samples {batch_start + 1}-{batch_end})...")

        # Prepare batch data
        batch_texts = []
        batch_images = []
        batch_metadata = []

        for sample in batch:
            species = sample['species']
            ground_truth = sample['ground_truth']
            xmin, ymin, zmin = sample['xmin'], sample['ymin'], sample['zmin']
            xmax, ymax, zmax = sample['xmax'], sample['ymax'], sample['zmax']
            box_size = np.array([xmax - xmin, ymax - ymin, zmax - zmin])

            # Get images
            if use_blank_images:
                front_img = Image.new('RGB', (1024, 1024), color=(128, 128, 128))
                side_img = Image.new('RGB', (1024, 1024), color=(128, 128, 128))
                top_img = Image.new('RGB', (1024, 1024), color=(128, 128, 128))
            else:
                front_img = sample['option_1_front_path']
                side_img = sample['option_1_side_path']
                top_img = sample['option_1_top_path']

            # Create prompt
            if use_simple_prompt:
                prompt = "What do you see in these three images?"
            else:
                prompt = f"""You are an expert at analyzing neuronal morphology.

We have the electron microscopy data from the {species} brain.

In the images, we have a selected 3D segmentation that is supposed to correspond to a complete neuronal structure. However, it could have split/merge errors as the segmentation algorithm makes mistakes.

The 3D snapshots are three different views of the same segment. The dimensions of the segment's bounding box are {box_size[0]} x {box_size[1]} x {box_size[2]} nm. Describe in detail what you see using the information in the 3D snapshots. Is the segment a neuron (soma and processes)? Multiple neurons merged together (multiple somas)? Processes like axon and dendrites without a cell body? Non-neuronal structures like glia, astrocytes, or blood vessels? Inspect very closely to avoid making errors, using the 3D views and size of the bounding box in your reasoning.

For {species} neurons, the somas tend to be round and generally {'a single process extends' if species == 'fly' else 'multiple processes extend'} from them {'before it branches into many processes' if species == 'fly' else 'outwards'}. Processes can be axons or dendrites, long and often branching. Synapses can also be considered as a part of processes, and these are often small segments (often smaller than a cubic micron). The nucleuses are round and do not have any processes extending from them. Blood vessels are tubular and obviously do not have any processes extending from them. Glial cells lack the branching processes of neurons, and instead appear like jagged masses.

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

            # Prepare messages for this sample
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": front_img},
                    {"type": "image", "image": side_img},
                    {"type": "image", "image": top_img},
                    {"type": "text", "text": prompt},
                ],
            }]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_texts.append(text)
            batch_images.extend([front_img, side_img, top_img])

            # Store metadata
            batch_metadata.append({
                'species': species,
                'proofread_root_id': sample['proofread_root_id'],
                'current_root_id': sample['current_root_id'],
                'ground_truth': ground_truth,
            })

        # Process batch through model
        inputs = processor(
            text=batch_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

        # Process outputs
        for output_text, metadata in zip(output_texts, batch_metadata):
            # Extract answer from tags
            llm_answer = ""
            analysis = ""

            answer_start = output_text.find("<answer>")
            answer_end = output_text.find("</answer>")
            if answer_start != -1 and answer_end != -1:
                llm_answer = output_text[answer_start + len("<answer>"):answer_end].strip()

            analysis_start = output_text.find("<analysis>")
            analysis_end = output_text.find("</analysis>")
            if analysis_start != -1 and analysis_end != -1:
                analysis = output_text[analysis_start + len("<analysis>"):analysis_end].strip()
            elif answer_start != -1:
                analysis = output_text[:answer_start].strip()

            # Map letter to description
            predicted_description = CLASS_MAPPING.get(llm_answer, None)

            # Check if correct
            correct = (predicted_description == metadata['ground_truth']) if predicted_description and metadata['ground_truth'] else None

            results.append({
                **metadata,
                'llm_answer': llm_answer,
                'predicted_description': predicted_description,
                'correct': correct,
                'analysis': analysis,
                'full_response': output_text,
            })

            print(f"  Sample answer: {llm_answer} -> {predicted_description}")

    # Save results
    df = pd.DataFrame(results)

    if 'correct' in df.columns and df['correct'].notna().any():
        accuracy = df['correct'].mean()
        print(f"\n{'='*60}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Correct: {df['correct'].sum()}/{len(df)}")
        print(f"{'='*60}")

        print("\nPrediction breakdown:")
        for answer_key in sorted(df['llm_answer'].unique()):
            subset = df[df['llm_answer'] == answer_key]
            correct_count = subset['correct'].sum()
            total_count = len(subset)
            description = CLASS_MAPPING.get(answer_key, 'Unknown')
            print(f"  {answer_key}: {correct_count}/{total_count} correct - {description}")

    # Save to file
    adapter_name = adapter_path.replace("/", "_")
    num_samples_str = f"{num_samples}samples" if num_samples else "all_samples"
    output_path = RESULTS_DIR / f"{adapter_name}_eval_{num_samples_str}.csv"
    df.to_csv(str(output_path), index=False)
    results_volume.commit()

    print(f"\nResults saved to: {output_path}")
    return df


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    num_samples: int = None,
    epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    gpu_count: int = 1,  # Number of GPUs to use
    use_4bit: bool = True,
    use_wandb: bool = False,
    run_name: str = None,
    augmented_dataset: str = None,  # Path to parquet file with teacher responses
    use_teacher: bool = False,  # Whether to use teacher model responses
    teacher_column: str = 'teacher_analysis',  # Column name (default works with merge script output)
    teacher_name: str = None,  # Optional: Teacher model name for W&B tracking
    test_split: float = 0.2,  # Fraction of data to hold out for evaluation
    eval_steps: int = 50,  # How often to evaluate (if eval_strategy="steps")
    eval_strategy: str = "steps",  # "steps", "epoch", or "no"
):
    """
    Local entry point to start fine-tuning.

    ⚠️ IMPORTANT: When changing --gpu-count, you MUST also update the decorator:
       @app.function(gpu=modal.gpu.A100(count=N)) to match, or you'll get an assertion error.

    Default: 2x A100 GPUs with distributed data parallel (DDP).
    Default effective batch size: 2 per GPU * 2 GPUs * 4 grad accum = 16.

    Usage:
        # Basic fine-tuning with Qwen3-VL-8B on small subset (default)
        modal run scripts/modal_qwen_finetune.py --num-samples 100 --epochs 3

        # Fine-tune with more data and custom name
        modal run scripts/modal_qwen_finetune.py \\
            --num-samples 1000 \\
            --epochs 5 \\
            --batch-size 4 \\
            --run-name "experiment1"

        # Full fine-tuning with W&B tracking
        modal run scripts/modal_qwen_finetune.py \\
            --epochs 3 \\
            --use-wandb \\
            --run-name "full_training_v1"

        # Fine-tune with teacher model responses
        # (parquet files from merge_claude_responses.py use unified 'teacher_*' columns)
        modal run scripts/modal_qwen_finetune.py \\
            --augmented-dataset "data/with_teacher_responses.parquet" \\
            --use-teacher \\
            --epochs 3 \\
            --run-name "teacher_distill_v1"

        # Fine-tune Qwen2-VL-7B instead
        modal run scripts/modal_qwen_finetune.py \\
            --model "Qwen/Qwen2-VL-7B-Instruct" \\
            --num-samples 100 \\
            --epochs 3

        # Train with custom train/test split and evaluation
        modal run scripts/modal_qwen_finetune.py \\
            --test-split 0.15 \\
            --eval-steps 25 \\
            --eval-strategy "steps" \\
            --epochs 5

        # Train without evaluation (use full dataset)
        modal run scripts/modal_qwen_finetune.py \\
            --test-split 0 \\
            --epochs 3

        # Use different number of GPUs (update decorator first: gpu=modal.gpu.A100(count=4))
        modal run scripts/modal_qwen_finetune.py \\
            --gpu-count 4 \\
            --epochs 3
    """
    print(f"Starting fine-tuning with {model}...")
    if use_teacher and augmented_dataset:
        print(f"Mode: Teacher distillation")
        print(f"Teacher data: {augmented_dataset}")
        print(f"Teacher column: {teacher_column}")
        if teacher_name:
            print(f"Teacher model (for tracking): {teacher_name}")
    else:
        print(f"Mode: Standard fine-tuning with ground truth labels")
    print(f"GPUs: {gpu_count}x A100")
    print(f"Samples: {num_samples if num_samples else 'all'}")
    print(f"Epochs: {epochs}")
    effective_batch = batch_size * gpu_count * gradient_accumulation
    print(f"Batch size: {batch_size} per GPU (effective: {effective_batch} with {gpu_count} GPUs)")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA rank: {lora_r}")
    print(f"4-bit quantization: {use_4bit}")
    print(f"Train/test split: {test_split*100:.0f}% test set" if test_split > 0 else "No train/test split")
    if test_split > 0:
        print(f"Eval strategy: {eval_strategy}")
        if eval_strategy == "steps":
            print(f"Eval frequency: every {eval_steps} steps")

    # Create config
    config = TrainingConfig(
        model_name=model,
        num_samples=num_samples,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        lora_r=lora_r,
        gpu_count=gpu_count,
        load_in_4bit=use_4bit,
        use_wandb=use_wandb,
        wandb_run_name=run_name,
        test_split_ratio=test_split,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
    )

    # Run fine-tuning
    final_path = finetune_qwen.remote(
        config=config,
        augmented_dataset_path=augmented_dataset,
        use_teacher_responses=use_teacher,
        teacher_response_column=teacher_column,
        teacher_model_name=teacher_name
    )

    print("\n" + "="*60)
    print("Fine-tuning job completed!")
    print(f"Model saved at: {final_path}")
    print("\nTo use the fine-tuned model, you can:")
    print("1. Use the merged model for inference (no LoRA adapters needed)")
    print("2. Use the checkpoint with LoRA adapters (smaller, requires loading adapters)")
    print("="*60)


@app.function(
    volumes={DATASET_DIR: dataset_volume},
    timeout=600
)
def _write_file_to_volume(path: str, data: bytes):
    """Helper function to write data to volume."""
    with open(path, 'wb') as f:
        f.write(data)
    dataset_volume.commit()
    return path


@app.local_entrypoint()
def upload_teacher_data(
    local_path: str,
    remote_name: str = None,
):
    """
    Upload teacher response data to Modal volume.

    Usage:
        modal run scripts/modal_qwen_finetune.py::upload_teacher_data \\
            --local-path "data/claude_37_teacher_responses.parquet" \\
            --remote-name "claude_37_teacher_responses.parquet"
    """
    from pathlib import Path

    local_file = Path(local_path)
    if not local_file.exists():
        print(f"Error: File not found: {local_path}")
        return

    # Use filename if remote name not specified
    if remote_name is None:
        remote_name = local_file.name

    remote_path = DATASET_DIR / remote_name

    print(f"Uploading {local_path} to volume...")
    print(f"  Remote path: {remote_path}")
    print(f"  File size: {local_file.stat().st_size / 1024 / 1024:.2f} MB")

    # Read local file
    with open(local_path, 'rb') as f:
        data = f.read()

    # Write to volume
    _write_file_to_volume.remote(str(remote_path), data)

    print(f"✓ Uploaded successfully!")
    print(f"\nUse in training with:")
    print(f"  --augmented-dataset \"{remote_path}\"")


@app.local_entrypoint()
def evaluate(
    adapter_path: str,
    num_samples: int = None,
    batch_size: int = 4,
    blank_images: bool = False,
    simple_prompt: bool = False,
):
    """
    Evaluate a fine-tuned LoRA adapter.

    Usage:
        modal run scripts/modal_qwen_finetune.py::evaluate \\
            --adapter-path "Qwen3-VL-8B-Instruct_20251112_191538_samplesall_epochs1_lr0.0002_r16" \\
            --num-samples 100 \\
            --batch-size 4
    """
    print(f"Evaluating adapter: {adapter_path}")
    print(f"Samples: {num_samples if num_samples else 'all'}")
    print(f"Batch size: {batch_size}")

    result_df = evaluate_adapter.remote(
        adapter_path=adapter_path,
        num_samples=num_samples,
        batch_size=batch_size,
        use_blank_images=blank_images,
        use_simple_prompt=simple_prompt,
    )

    print("\nCompleted!")
    if len(result_df) > 0:
        print(result_df[['species', 'current_root_id', 'llm_answer', 'ground_truth', 'correct']].head(10))
