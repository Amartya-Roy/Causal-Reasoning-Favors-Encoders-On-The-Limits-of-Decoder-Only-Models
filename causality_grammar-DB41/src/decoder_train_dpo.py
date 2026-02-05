s"""
DPO (Direct Preference Optimization) Finetuning Script
For training models to prefer correct reasoning chains over corrupted ones.

Data format expected:
- Question: The logic problem with facts, rules, and query
- Complex_CoT: The correct reasoning chain (chosen response)
- Non_Valid_Complex_CoT: The corrupted reasoning chain (rejected response)
- Response: The final label (0 or 1)
- Depth: The depth of the problem

Multi-GPU Training:
    Using torchrun (recommended):
        torchrun --nproc_per_node=4 decoder_train_dpo.py
    
    Using accelerate:
        accelerate launch --multi_gpu --num_processes=4 decoder_train_dpo.py
    
    Using DeepSpeed:
        accelerate launch --config_file deepspeed_config.yaml decoder_train_dpo.py
"""

import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import torch
import torch.distributed as dist
from datasets import load_dataset, Dataset, Features, Value
from torch.utils.data import DataLoader

import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from trl import DPOTrainer, DPOConfig
from transformers import DataCollatorForSeq2Seq

# Check if running in distributed mode
def is_distributed():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    if is_distributed():
        return dist.get_rank()
    return int(os.environ.get("LOCAL_RANK", 0))

def is_main_process():
    return get_rank() == 0

def print_main(*args, **kwargs):
    """Print only on main process."""
    if is_main_process():
        print(*args, **kwargs)


def set_env(base_path):
    """Set environment variables for HuggingFace cache directories."""
    os.environ["HF_HOME"] = base_path
    os.environ["HF_TRANSFORMERS_CACHE"] = base_path + "/hf_cache"
    os.environ["HF_HUB_CACHE"] = base_path + "/hf_hub"
    os.environ["HF_XET_CACHE"] = base_path + "/hf_xet"
    os.environ["HF_DATASETS"] = base_path + "/datasets"


def load_data_local(dataset_path):
    """
    Load dataset from local parquet file for DPO training.
    
    Expected columns:
    - Question: The logic problem
    - Complex_CoT: Correct reasoning (chosen)
    - Non_Valid_Complex_CoT: Corrupted reasoning (rejected)
    - Response: Final label
    - Depth: Problem depth
    """
    
    df = pd.read_parquet(dataset_path)
    
    print(f"Loaded {len(df)} examples from local file")
    print(f"Dataset columns: {df.columns.tolist()}")
    
    # Verify required columns exist
    required_cols = ["Question", "Complex_CoT", "Non_Valid_Complex_CoT", "Response"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def load_model(cache_dir, model_name):
    """Load model and tokenizer using unsloth for efficient training."""
    
    # Get local rank for multi-GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f"unsloth/{model_name}",
        max_seq_length=4096,    # Context length
        load_in_4bit=True,      # 4bit uses much less memory
        load_in_8bit=False,
        full_finetuning=False,  # Use LoRA for DPO (more stable)
        # token = "hf_...",     # Use if using gated models
        cache_dir=cache_dir,
        # Don't specify device_map for multi-GPU - let distributed handle it
    )
    
    # Add LoRA adapters for DPO training
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,                    # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,          # Optimized to 0
        bias="none",
        use_gradient_checkpointing="unsloth",  # Memory efficient
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Print trainable parameters only on main process
    if is_main_process():
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


def create_prompt(question):
    """Create the user prompt for the logic reasoning task."""
    
    prompt = f"""You are evaluating a subset of first-order logic. 
In this subset, conjunctions are given by [AND], implications by [IMPLY], and separations between clauses as [PERIOD]
You will be given Facts, and Rules. Based on these, generate the correct proof chain to determine the truth value of the Query.

Provide the reasoning chain that logically derives the answer from the given facts and rules.

{question}
"""
    return prompt


def prepare_dpo_dataset(df, tokenizer, model_type="Qwen"):
    """
    Prepare dataset for DPO training.
    
    DPO requires:
    - prompt: The input prompt
    - chosen: The preferred response (correct proof chain)
    - rejected: The non-preferred response (corrupted proof chain)
    
    Note: We only train on proof chains, not the final label.
    """
    
    dpo_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing DPO dataset"):
        question = row["Question"]
        chosen_cot = row["Complex_CoT"]
        rejected_cot = row["Non_Valid_Complex_CoT"]
        
        # Skip if either CoT is missing or empty
        if pd.isna(chosen_cot) or pd.isna(rejected_cot):
            continue
        if not str(chosen_cot).strip() or not str(rejected_cot).strip():
            continue
            
        # Create the prompt
        prompt = create_prompt(question)
        
        # Create chosen and rejected responses (proof chains only)
        chosen_response = str(chosen_cot)
        rejected_response = str(rejected_cot)
        
        dpo_data.append({
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
        })
    
    print(f"Created {len(dpo_data)} DPO training examples")
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(pd.DataFrame(dpo_data))
    
    return dataset


def prepare_dpo_dataset_with_chat_template(df, tokenizer, model_type="Qwen"):
    """
    Prepare dataset for DPO training using chat templates.
    
    This version applies the chat template to format prompts properly.
    Note: We only train on proof chains, not the final label.
    """
    
    dpo_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing DPO dataset"):
        question = row["Question"]
        chosen_cot = row["Complex_CoT"]
        rejected_cot = row["Non_Valid_Complex_CoT"]
        
        # Skip if either CoT is missing or empty
        if pd.isna(chosen_cot) or pd.isna(rejected_cot):
            continue
        if not str(chosen_cot).strip() or not str(rejected_cot).strip():
            continue
        
        # Create the user message
        user_message = create_prompt(question)
        
        # Create chosen and rejected responses (proof chains only)
        chosen_response = str(chosen_cot)
        rejected_response = str(rejected_cot)
        
        # Format as chat conversations
        prompt_messages = [{"role": "user", "content": user_message}]
        
        # Apply chat template for the prompt
        formatted_prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True  # Add the assistant prefix
        )
        
        dpo_data.append({
            "prompt": formatted_prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
        })
    
    print(f"Created {len(dpo_data)} DPO training examples with chat template")
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(pd.DataFrame(dpo_data))
    
    return dataset


def setup_dpo_trainer(model, tokenizer, train_dataset, model_output_dir, ref_model=None, num_gpus=1):
    """
    Set up DPO trainer with recommended hyperparameters.
    
    DPO directly optimizes for the preference without needing a reward model.
    Supports multi-GPU training via DDP/FSDP/DeepSpeed.
    """
    
    # Detect number of GPUs if not specified
    if num_gpus == 1 and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    
    # Adjust batch size based on number of GPUs
    # Effective batch size = per_device_batch * gradient_accum * num_gpus
    per_device_batch = 2
    gradient_accum = max(1, 8 // num_gpus)  # Target effective batch of ~16
    
    if is_main_process():
        print(f"Training on {num_gpus} GPU(s)")
        print(f"Per-device batch size: {per_device_batch}")
        print(f"Gradient accumulation steps: {gradient_accum}")
        print(f"Effective batch size: {per_device_batch * gradient_accum * num_gpus}")
    
    dpo_config = DPOConfig(
        output_dir=model_output_dir,
        
        # Batch size settings
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=gradient_accum,
        
        # Learning rate settings
        learning_rate=5e-6,  # Lower LR for DPO (important!)
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        
        # Training duration
        num_train_epochs=3,
        max_steps=-1,
        
        # DPO specific
        beta=0.1,  # KL penalty coefficient (0.1-0.5 typical)
        loss_type="sigmoid",  # Standard DPO loss
        
        # Memory optimization
        gradient_checkpointing=True,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        
        # Logging
        logging_steps=50,
        logging_first_step=True,
        
        # Saving
        save_strategy="epoch",
        save_total_limit=2,
        
        # Misc
        seed=3407,
        report_to="none",
        
        # Sequence lengths
        max_length=4096,
        max_prompt_length=2048,
        
        # Multi-GPU settings
        ddp_find_unused_parameters=False,  # More efficient for DDP
        dataloader_num_workers=4,  # Parallel data loading
        dataloader_pin_memory=True,
        
        # For DeepSpeed ZeRO (optional - uncomment if using DeepSpeed)
        # deepspeed="deepspeed_config.json",
    )
    
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,  # If None, uses implicit reference
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    return trainer


def save_stats(history, save_path):
    """Save training statistics to CSV (only on main process)."""
    if not is_main_process():
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame(history)
    df.to_csv(save_path, index=False)
    print(f"Training stats saved to {save_path}")


def save_model(model, tokenizer, save_path):
    """Save the trained model and tokenizer (only on main process)."""
    if not is_main_process():
        return
    os.makedirs(save_path, exist_ok=True)
    
    # Save with unsloth's efficient method
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"Model saved to {save_path}")


def main(base_path, model_name, dataset_path, use_chat_template=True):
    """
    Main training function for DPO finetuning.
    
    Args:
        base_path: Base directory for models and cache
        model_name: Name of the model (e.g., "Qwen3-1.7B", "Llama-3.2-1B-Instruct")
        dataset_path: Path to the parquet file with corruptions
        use_chat_template: Whether to use chat templates for formatting
    
    Multi-GPU Usage:
        torchrun --nproc_per_node=NUM_GPUS decoder_train_dpo.py
        OR
        accelerate launch --multi_gpu --num_processes=NUM_GPUS decoder_train_dpo.py
    """
    
    set_env(base_path)
    
    # Get number of GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # ==================== Load Model ====================
    print_main("=" * 60)
    print_main("Loading Model...")
    print_main(f"Detected {num_gpus} GPU(s)")
    print_main("=" * 60)
    
    model, tokenizer = load_model(base_path + "/models", model_name)
    
    # Set up chat template for specific models
    if "Llama" in model_name:
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="llama-3.1",
        )
    elif "Qwen" in model_name:
        # Qwen models typically have built-in chat template
        pass
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print_main("Model ready!")
    
    # ==================== Load Data ====================
    print_main("=" * 60)
    print_main("Loading Data...")
    print_main("=" * 60)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    
    df = load_data_local(dataset_path)
    print_main(f"Dataset shape: {df.shape}")
    
    # ==================== Prepare DPO Dataset ====================
    print_main("=" * 60)
    print_main("Preparing DPO Dataset...")
    print_main("=" * 60)
    
    if use_chat_template:
        train_dataset = prepare_dpo_dataset_with_chat_template(df, tokenizer, model_type=model_name)
    else:
        train_dataset = prepare_dpo_dataset(df, tokenizer, model_type=model_name)
    
    print_main(f"Training dataset size: {len(train_dataset)}")
    
    # Show a sample (only on main process)
    if is_main_process():
        print("\n--- Sample DPO Example ---")
        sample = train_dataset[0]
        print(f"Prompt (first 500 chars):\n{sample['prompt'][:500]}...")
        print(f"\nChosen (first 300 chars):\n{sample['chosen'][:300]}...")
        print(f"\nRejected (first 300 chars):\n{sample['rejected'][:300]}...")
        print("-" * 40)
    
    # ==================== Train ====================
    print_main("=" * 60)
    print_main("Starting DPO Training...")
    print_main("=" * 60)
    
    model_output_dir = os.path.join(base_path, "models", f"{model_name}-dpo-finetuned-temp")
    
    trainer = setup_dpo_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        model_output_dir=model_output_dir,
        ref_model=None,  # DPO will create implicit reference
        num_gpus=num_gpus,
    )
    
    # Train
    trainer_stats = trainer.train()
    history = trainer.state.log_history
    
    print_main("Training complete!")
    
    # ==================== Save ====================
    print_main("=" * 60)
    print_main("Saving Model and Stats...")
    print_main("=" * 60)
    
    # Save training logs
    stats_save_path = os.path.join(base_path, "output_logs", f"{model_name}_dpo_training_logs.csv")
    save_stats(history, stats_save_path)
    
    # Save final model
    model_save_path = os.path.join(base_path, "models", f"{model_name}-dpo-finetuned")
    save_model(model, tokenizer, model_save_path)
    
    print_main("=" * 60)
    print_main("DPO Training Complete!")
    print_main("=" * 60)
    
    return model, tokenizer, trainer, history


if __name__ == "__main__":
    
    # ==================== Configuration ====================
    
    # Base path for models, cache, and outputs
    base_path = "/scratch/...."  # Update with your path
    
    # Model to finetune
    model_name = "Qwen3-1.7B"
    # model_name = "Llama-3.2-1B-Instruct"
    
    # Path to training data with corruptions
    dataset_path = "/Users/amartyaroy/Causal-Reasoning-Favors-Encoders-On-The-Limits-of-Decoder-Only-Models/causality_grammar-DB41/data/train_data_with_corruptions.parquet"
    
    # Whether to use chat templates
    use_chat_template = True
    
    # ==================== Run Training ====================
    
    main(
        base_path=base_path,
        model_name=model_name,
        dataset_path=dataset_path,
        use_chat_template=use_chat_template
    )
