# Fine-tuning with Claude 3.7 Sonnet Responses

This guide explains how to use high-quality responses from Claude 3.7 Sonnet to fine-tune Qwen-VL models.

## Overview

Instead of using synthetic responses, we can leverage the superior reasoning and analysis from Claude 3.7 Sonnet as training data for smaller, more efficient models like Qwen-VL.

## Workflow

### Step 1: Merge Claude Responses with Dataset

First, match Claude's responses with the HuggingFace dataset and filter to only correct responses:

```bash
python scripts/merge_claude_responses.py \
  --csv-files \
    "output/mouse_segment_classification/claude-3-7-sonnet-20250219_analysis_results_K5_v2.csv" \
    "output/mouse_segment_classification/claude-3-7-sonnet-20250219_analysis_results_K5.csv" \
  --output "data/connectomebench_with_claude_responses_correct_only.parquet" \
  --filter-correct-only
```

This will:
- Load both CSV files with Claude responses
- Match them with the HuggingFace dataset using proofread_root_id, current_root_id, and coordinates
- Check correctness by comparing Claude's answers with ground truth
- Filter to only samples where Claude answered correctly
- Create an augmented dataset with Claude's analysis and answers
- Save as parquet format for efficient loading

**Results:**
- Total samples in dataset: 377
- Matched samples: 377 (100% match rate)
- Claude accuracy: **84.4%** (318/377 correct)
- Final training dataset: **318 correct responses**
- Output: `data/connectomebench_with_claude_responses_correct_only.parquet`

**Note:** We only use correct responses for training to ensure high-quality knowledge distillation. Training on incorrect responses would teach the model to make the same mistakes.

### Step 2: Fine-tune with Claude Responses

Now use the augmented dataset to fine-tune Qwen-VL:

```bash
# Basic fine-tuning with all samples
modal run scripts/modal_qwen_finetune_with_claude.py \
  --epochs 3 \
  --batch-size 2 \
  --gradient-accumulation 4

# With W&B tracking and custom name
modal run scripts/modal_qwen_finetune_with_claude.py \
  --epochs 5 \
  --use-wandb \
  --run-name "claude_teacher_v1"

# Test run with fewer samples
modal run scripts/modal_qwen_finetune_with_claude.py \
  --num-samples 50 \
  --epochs 2 \
  --run-name "test_run"
```

### Step 3: Evaluate the Fine-tuned Model

Use the evaluation function from the original script:

```bash
modal run scripts/modal_qwen_finetune.py::evaluate \
  --adapter-path "Qwen3-VL-8B-Instruct_claude_20251115_..._samplesall_epochs3_lr0.0002_r16" \
  --num-samples 100 \
  --batch-size 4
```

## Script Details

### `merge_claude_responses.py`

**Purpose:** Match Claude responses with HuggingFace dataset

**Key features:**
- Matches based on proofread_root_id, current_root_id, and x/y/z coordinates
- Handles multiple CSV files
- Saves augmented dataset in parquet format
- Generates metadata JSON with statistics
- Optional filtering to only include matched samples

**Arguments:**
- `--csv-files`: One or more CSV files with Claude responses
- `--dataset-name`: HuggingFace dataset name (default: jeffbbrown2/ConnectomeBench)
- `--config-name`: Dataset configuration (default: "MICrONS, Segment Classification")
- `--output`: Output path for augmented dataset
- `--filter-matched-only`: Only save samples with Claude responses

### `modal_qwen_finetune_with_claude.py`

**Purpose:** Fine-tune Qwen-VL using Claude responses

**Key differences from base script:**
- Uses `convert_sample_to_conversation_with_claude()` which prioritizes Claude's actual responses
- Adds "claude" tag to saved model names for easy identification
- Saves training data source in config for tracking

**Arguments:**
- `--augmented-dataset`: Path to augmented dataset parquet file
- `--model`: Qwen model to fine-tune (default: Qwen/Qwen3-VL-8B-Instruct)
- `--num-samples`: Number of samples to use (None = all)
- `--epochs`: Number of training epochs
- `--batch-size`: Per-device batch size
- `--gradient-accumulation`: Gradient accumulation steps
- `--learning-rate`: Learning rate (default: 2e-4)
- `--lora-r`: LoRA rank (default: 16)
- `--use-wandb`: Enable W&B logging
- `--run-name`: Custom run name for W&B and model folder

## Benefits of This Approach

1. **Higher Quality Training Data:** Claude 3.7 Sonnet provides detailed, accurate analysis
2. **Knowledge Distillation:** Transfer Claude's reasoning to smaller, faster models
3. **Cost Efficiency:** Once fine-tuned, Qwen-VL is cheaper to run than Claude
4. **Customization:** Can fine-tune for specific domains or tasks

## Expected Results

- Training on 377 high-quality Claude responses
- Models saved with "claude" tag for identification
- Can evaluate against original ground truth to measure knowledge transfer
- Compare with models trained on synthetic data to see improvement

## Next Steps

1. Run evaluation on multiple checkpoints to find best model
2. Compare performance with base Qwen-VL and synthetic-trained models
3. Consider adding more Claude responses for better coverage
4. Experiment with different LoRA configurations and learning rates
