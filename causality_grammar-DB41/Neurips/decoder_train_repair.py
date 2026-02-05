import os, re

# Set required environment variables BEFORE importing unsloth
os.environ["UNSLOTH_COMPILE"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import torch
from datasets import load_dataset, Dataset, Features, Value
from torch.utils.data import DataLoader

import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq


def set_env(base_path):
    os.environ["HF_HOME"] = base_path
    os.environ["HF_TRANSFORMERS_CACHE"] = base_path + "/hf_cache"
    os.environ["HF_HUB_CACHE"] = base_path + "/hf_hub"
    os.environ["HF_XET_CACHE"] = base_path + "/hf_xet"
    os.environ["HF_DATSETS"] = base_path + "/datasets"

def load_data_from_hf(cache_dir):
    """
    Load dataset from Hugging Face with repair task:
    Input: Question + Non_Valid_Complex_CoT
    Output: Complex_CoT (correct reasoning)
    """
    
    # Define features that match the actual parquet file
    custom_features = Features({
        "Question": Value("string"),
        "Complex_CoT": Value("string"),
        "Non_Valid_Complex_CoT": Value("string"),
        "Response": Value("int64"),
        "Depth": Value("int64")
    })

    dataset = load_dataset(
        "Amartya77/LogicBench",
        split="train",  # Load the train split directly
        features=custom_features,  # Override the default features with the actual ones
        cache_dir=cache_dir,
        verification_mode="no_checks"
    )
    
    print(f"Loaded {len(dataset)} examples from Hugging Face")
    print(f"Dataset columns: {dataset.column_names}")
    
    return dataset

def load_data_local(dataset_path):
    """
    Load dataset from local parquet file with repair task:
    Input: Question + Non_Valid_Complex_CoT
    Output: Complex_CoT (correct reasoning)
    """
    
    # Define features that match the actual parquet file
    custom_features = Features({
        "Question": Value("string"),
        "Complex_CoT": Value("string"),
        "Non_Valid_Complex_CoT": Value("string"),
        "Response": Value("int64"),
        "Depth": Value("int64")
    })

    dataset = load_dataset(
        "parquet",
        data_files={"train": dataset_path},
        split="train",
        features=custom_features,
        cache_dir=None,
        verification_mode="no_checks"
    )
    
    print(f"Loaded {len(dataset)} examples from local file")
    print(f"Dataset columns: {dataset.column_names}")
    
    return dataset

def load_model(cache_dir, model_name):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = f"unsloth/{model_name}",
        max_seq_length = 4096,   # Context length - can be longer, but uses more memory
        load_in_4bit = True,     # 4bit uses much less memory
        load_in_8bit = False,    # A bit more accurate, uses 2x memory
        full_finetuning = True,  # We have full finetuning now!
        # token = "hf_...",      # use one if using gated models
        cache_dir=cache_dir
        # Don't use device_map for DDP - let torch distributed handle it
    )

    # Check trainable parameters
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"{100*trainable/total:.2f}% trainable")

    return model, tokenizer

def generate_conversation_repair(examples):
    """
    Generate conversation for repair task:
    π_θ(Complex_CoT | Question, Non_Valid_Complex_CoT)
    
    The model receives:
    1. The question (facts, rules, query)
    2. An incorrect reasoning chain (Non_Valid_Complex_CoT)
    
    And must predict:
    - The correct reasoning chain (Complex_CoT)
    """

    questions = examples["Question"]
    non_valid_cot = examples["Non_Valid_Complex_CoT"]
    correct_cot = examples["Complex_CoT"]

    user = f"""You are a logical reasoning repair system. You will be given:
1. A logic problem with Facts, Rules, and a Query
2. An incorrect reasoning chain that contains errors

Your task is to identify the errors and provide the CORRECT reasoning chain.

Use symbolic notation:
- Use ∧ for conjunction (AND)
- Use ⇒ for implication (IMPLY)
- Format: "Proof chain:" followed by the inference steps

Problem:
{questions}

Incorrect Reasoning:
{non_valid_cot}

Provide the corrected reasoning chain:"""

    assistant = f"{correct_cot}"
    
    conversation = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    return {"conversation": conversation}

def process_data(dataset, tokenizer):
    """Process dataset for Qwen models"""
    # Map the conversation generation function
    dataset_with_convos = dataset.map(generate_conversation_repair, batched=False)
    
    # Apply chat template to each conversation individually
    conversations = []
    for example in dataset_with_convos:
        formatted = tokenizer.apply_chat_template(
            example["conversation"],
            tokenize=False,
            add_generation_prompt=False
        )
        conversations.append(formatted)

    data = pd.DataFrame(conversations)
    train_dataset = Dataset.from_pandas(data)

    return train_dataset

def llama_process_data(dataset, tokenizer):
    """Process dataset for Llama models"""

    def formatting_prompts_func(examples):
        convos = examples["conversation"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}
    
    dataset = dataset.map(generate_conversation_repair)
    dataset = standardize_sharegpt(dataset)
    dataset = dataset.map(formatting_prompts_func, batched=True)

    return dataset

def llama_setup_and_train(model, tokenizer, train_dataset, model_output_dir):
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=4096,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        packing=False,  # Can make training 5x faster for short sequences.
        args=SFTConfig(
            output_dir=model_output_dir, 
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,  # Use GA to mimic batch size!
            num_train_epochs=3,  # Set this for 1 full training run.
            max_steps=-1,  # Set to -1 to run for num_train_epochs
            learning_rate=5e-5,  # Reduce to 2e-5 for long training runs
            logging_steps=100,
            seed=3407,
            save_strategy="no",       
            save_total_limit=1,  
            report_to="none",  # Use this for WandB etc
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    
    trainer_stats = trainer.train()
    history = trainer.state.log_history

    return model, tokenizer, trainer, history

def setup_and_train(model, tokenizer, train_dataset, model_output_dir):
    """Training setup for Qwen models"""

    def formatting_prompts_func(examples):
        texts = []
        # Corrected to access column '0' instead of 'text'
        for text in examples["0"]:
            texts.append(text)
        return texts

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,  # Can set up evaluation!
        args=SFTConfig(
            output_dir=model_output_dir, 
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,  # Use GA to mimic batch size!
            num_train_epochs=3,  # Set to 3 epochs
            max_steps=-1,  # Set to -1 to run for num_train_epochs
            learning_rate=4e-5,  # Slightly reduced from 5e-5 for stability
            logging_steps=100,
            seed=3407,
            save_strategy="epoch",  # Save checkpoints at each epoch      
            save_total_limit=2,  # Keep last 2 checkpoints
            report_to="none",  # Use this for WandB etc
            ddp_find_unused_parameters=False,  # For multi-GPU
        ),
        formatting_func=formatting_prompts_func,
    )

    trainer.train()

    return model, tokenizer, trainer

def save_stats(history, save_dir_path):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_dir_path), exist_ok=True)
    df = pd.DataFrame(history)
    df.to_csv(save_dir_path, index=False)

def save_model(model, tokenizer, trainer, save_dir_path):
    # Create directory if it doesn't exist
    os.makedirs(save_dir_path, exist_ok=True)
    trainer.save_model(save_dir_path)
    tokenizer.save_pretrained(save_dir_path)

def main(base_path, model_name, use_hf=False):

    set_env(base_path)

    print("Loading Model......")
    model, tokenizer = load_model(base_path + "/models", model_name)
    if "Llama" in model_name:
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="llama-3.1",
        )
    print("Model ready!")

    print("Loading Data......")
    if use_hf:
        dataset = load_data_from_hf(base_path + "/datasets")
    else:
        # Use training data with proper corruptions
        dataset_path = "/home/omy1cob/Causal_LLM/causality_grammar-DB41/data/train_data_proper_corruptions.parquet"
        
        import os
        if not os.path.exists(dataset_path):
            print(f"ERROR: Training data not found at {dataset_path}")
            print("Please run: python -c 'import pandas as pd; ...' to generate corruptions")
            raise FileNotFoundError(dataset_path)
        
        print(f"Using training data: {dataset_path}")
        dataset = load_data_local(dataset_path)
    
    # Optional: use a subset for testing
    # dataset = dataset.shuffle().select(range(1000))
    
    if "Qwen" in model_name:
        train_dataset = process_data(dataset, tokenizer)
    elif "Llama" in model_name:
        train_dataset = llama_process_data(dataset, tokenizer)
    else:
        raise ValueError("Enter Valid model name")
    print("Dataset ready!")

    print("Starting training......")
    model_output_dir = base_path + f"/models/{model_name}-repair-v2-fullfinetuned-temp"

    if "Qwen" in model_name:
        model, tokenizer, trainer = setup_and_train(model, tokenizer, train_dataset, model_output_dir)
        history = trainer.state.log_history
    elif "Llama" in model_name:
        model, tokenizer, trainer, history = llama_setup_and_train(model, tokenizer, train_dataset, model_output_dir)
    else:
        raise ValueError("Enter Valid model name")
    print("Training complete!")

    print("Saving Model and Stats......")
    stats_save_dir_path = f"/home/omy1cob/Causal_LLM/causality_grammar-DB41/output_logs/{model_name}_repair_v2_training_logs.csv"
    save_stats(history, stats_save_dir_path)
    model_save_dir_path = base_path + f"/models/{model_name}-repair-v2-fullfinetuned"
    save_model(model, tokenizer, trainer, model_save_dir_path)
    print("Complete!")

if __name__ == "__main__":

    base_path = "/home/omy1cob/Causal_LLM/causality_grammar-DB41"  # Update with your path
    model_name = "Qwen3-1.7B"
    # model_name = "Llama-3.2-1B-Instruct"
    
    # Set to True to load from Hugging Face, False to load from local parquet
    use_hf = False
    
    main(base_path, model_name, use_hf=use_hf)
