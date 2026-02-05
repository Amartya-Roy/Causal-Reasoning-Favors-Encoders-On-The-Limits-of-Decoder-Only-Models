import os, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import torch
from datasets import load_dataset, Dataset
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

def load_data(dataset_path):
    """
    Load training data from local parquet file.
    Original task: Question → Complex_CoT + Response
    """
    dataset = load_dataset(
        "parquet",
        data_files=dataset_path,
        split="train",
        verification_mode="no_checks"
    )

    # temp_dataset = dataset.shuffle().select(range(100))
    # return temp_dataset
    return dataset

def load_model(cache_dir, model_name):

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = f"unsloth/{model_name}",
        max_seq_length = 4096,   # Context length - can be longer, but uses more memory
        load_in_4bit = True,     # 4bit uses much less memory
        load_in_8bit = False,    # A bit more accurate, uses 2x memory
        full_finetuning = True, # We have full finetuning now!
        # token = "hf_...",      # use one if using gated models
        cache_dir=cache_dir
    )

    return model, tokenizer

def generate_conversation(example):
  """
  Generate conversation for standard training with mixed inputs:
  - 50% of examples: Question only → Valid proof
  - 50% of examples: Question + Invalid proof (as context) → Valid proof
  
  This is NOT repair - model learns to generate valid proofs with or without invalid context.
  """
  question = example["Question"]
  proof_chain = example["Complex_CoT"]
  invalid_proof = example.get("Non_Valid_Complex_CoT", "")
  
  # Randomly decide whether to include invalid proof as context (50%)
  use_invalid_context = random.random() < 0.5
  
  if use_invalid_context and invalid_proof and str(invalid_proof).strip():
    # Include invalid proof as context (not explicit repair task)
    user = f"""You are a logical reasoning system. Solve this first-order logic problem.

Given: Facts, Rules, and a Query.
Task: Generate a proof chain to determine if the query is true.

Use symbolic notation:
- ∧ for AND, ⇒ for IMPLY
- Format: "Proof chain:" followed by the inference steps

Problem:
{question}

Previous attempt (may contain errors):
{invalid_proof}

Generate the correct proof chain:"""
  else:
    # Standard: question only
    user = f"""You are a logical reasoning system. Solve this first-order logic problem.

Given: Facts, Rules, and a Query.
Task: Generate a proof chain to determine if the query is true.

Use symbolic notation:
- ∧ for AND, ⇒ for IMPLY
- Format: "Proof chain:" followed by the inference steps

{question}

OUTPUT:"""
  
  # Only output the proof chain, no label/response
  assistant = f"{proof_chain}"
  
  conversation = [
      {"role" : "user",      "content" : user},
      {"role" : "assistant", "content" : assistant},
  ]
  return {"conversation": conversation}

def process_data(dataset, tokenizer):
    # Map to create conversations first
    dataset_with_convos = dataset.map(generate_conversation, batched=False)
    
    # Apply chat template to each conversation
    def apply_template(example):
        text = tokenizer.apply_chat_template(
            example["conversation"],
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}
    
    train_dataset = dataset_with_convos.map(apply_template, batched=False)
    return train_dataset

def llama_process_data(dataset, tokenizer):

    def formatting_prompts_func(examples):
        convos = examples["conversation"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }
    
    dataset = dataset.map(generate_conversation)
    dataset = standardize_sharegpt(dataset)
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    return dataset

def llama_setup_and_train(model, tokenizer, train_dataset, model_output_dir):
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "text",
        max_seq_length = 4096,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        packing = False, # Can make training 5x faster for short sequences.
        args = SFTConfig(
            output_dir= model_output_dir, 
            # dataset_text_field is not needed when using formatting_func
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4, # Use GA to mimic batch size!
            num_train_epochs = 3, # Set this for 1 full training run.
            max_steps = -1, # Set to -1 to run for num_train_epochs
            learning_rate = 5e-5, # Reduce to 2e-5 for long training runs
            logging_steps = 100,
            seed = 3407,
            save_strategy="no",       
            save_total_limit=1,  
            report_to = "none", # Use this for WandB etc
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    
    trainer_stats = trainer.train()
    history = trainer.state.log_history

    return model, tokenizer, trainer, history

def setup_and_train(model, tokenizer, train_dataset, model_output_dir):

    def formatting_prompts_func(examples):
        texts = []
        # Corrected to access column '0' instead of 'text'
        for text in examples["0"]:
            texts.append(text)
        return texts

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = None, # Can set up evaluation!
        args = SFTConfig(
            output_dir= model_output_dir, 
            # dataset_text_field is not needed when using formatting_func
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4, # Use GA to mimic batch size!
            # warmup_steps = 5,
            num_train_epochs = 3, # Set this for 1 full training run.
            max_steps = -1, # Set to -1 to run for num_train_epochs
            learning_rate = 5e-5, # Reduce to 2e-5 for long training runs
            logging_steps = 100,
            # optim = "adamw_8bit", # Using default optimizer
            # weight_decay = 0.01,
            # lr_scheduler_type = "linear",
            seed = 3407,
            save_strategy="no",       
            save_total_limit=1,      
            report_to = "none", # Use this for WandB etc
        ),
        formatting_func = formatting_prompts_func,
    )

    trainer.train()

    return model, tokenizer, trainer

def save_stats(history, save_dir_path):

    df = pd.DataFrame(history)
    df.to_csv(save_dir_path, index=False)


def save_model(model, tokenizer, trainer, save_dir_path):

    trainer.save_model(save_dir_path)
    tokenizer.save_pretrained(save_dir_path)

def main(base_path, model_name):

    set_env(base_path)

    print("Loading Model......")
    model, tokenizer = load_model(base_path + "/models", model_name)
    if "Llama" in model_name:
        tokenizer = get_chat_template(
            tokenizer,
            chat_template = "llama-3.1",
        )
    print("Model ready!")


    print("Loading Data......")
    dataset_path = base_path + "/data/train_data_proper_corruptions.parquet"
    dataset = load_data(dataset_path)
    if "Qwen" in model_name:
        train_dataset = process_data(dataset, tokenizer)
    elif "Llama" in model_name:
        train_dataset = llama_process_data(dataset, tokenizer)
    else:
        raise ValueError("Enter Valid model name")
    print("Dataset ready!")

    print("Starting training......")
    model_output_dir = base_path + f"/models/{model_name}-standard-v2-fullfinetuned-temp"

    history = None  # Initialize history
    if "Qwen" in model_name:
        model, tokenizer, trainer = setup_and_train(model, tokenizer, train_dataset, model_output_dir)
        # Extract training history from trainer if available
        if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
            history = trainer.state.log_history
    elif "Llama" in model_name:
        model, tokenizer, trainer, history = llama_setup_and_train(model, tokenizer, train_dataset, model_output_dir)
    else:
        raise ValueError("Enter Valid model name")
    print("Training complete!")

    print("Saving Model and Stats......")
    if history:
        stats_save_dir_path = base_path + f"/output_logs/{model_name}_standard_v2_training_logs.csv"
        save_stats(history, stats_save_dir_path)
    model_save_dir_path = base_path + f"/models/{model_name}-standard-v2-fullfinetuned"
    save_model(model, tokenizer, trainer, model_save_dir_path)
    print("Complete!")



if __name__ == "__main__":

    base_path = "/home/amartya/Causal_LLM/causality_grammar-DB41"
    model_name = "Qwen3-1.7B"
    # model_name = "Llama-3.2-1B-Instruct"
    main(base_path, model_name)

