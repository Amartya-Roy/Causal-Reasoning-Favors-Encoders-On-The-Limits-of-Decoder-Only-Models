import os, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

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

    dataset = load_dataset(
        "causality-grammar/logic_explanations",
        data_files={"train": "deepseek_5k_RP.parquet"},
        cache_dir= dataset_path,
        verification_mode="no_checks"
    )["train"]

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

def generate_conversation(examples):

  questions  = examples["Question"]
  proof_chains = examples["Complex_CoT"]
  final_answers = examples["Response"]

  user = f"""You are evaluating a subset of first-order logic. 
In this subset, conjunctions are given by [AND], implications by [IMPLY], and separations between clauses as [PERIOD]
You will be given Facts, and Rules. Based on these, determine the truth value of the Query.
Your final answer should be 0 (if the Query is false) or 1 (if true).

Give your final answer in the following format:
Label: The label from the criterion. Only use the numbers 0 or 1.

{questions}
"""
#   assistant = f"<think>\n{proof_chains}\n</think>\nLabel: {final_answers} "
  assistant = f"{proof_chains}\nLabel: {final_answers} "
  conversation = [
      {"role" : "user",      "content" : user},
      {"role" : "assistant", "content" : assistant},
  ]
  return {"conversation": conversation}

def process_data(dataset, tokenizer):
    conversations = tokenizer.apply_chat_template(
        dataset.map(generate_conversation, batched=False)["conversation"],
        tokenize = False,
    )

    data = pd.DataFrame(conversations)
    train_dataset = Dataset.from_pandas(data)

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
    dataset = load_data(base_path + "/datasets")
    if "Qwen" in model_name:
        train_dataset = process_data(dataset, tokenizer)
    elif "Llama" in model_name:
        train_dataset = llama_process_data(dataset, tokenizer)
    else:
        raise ValueErrors("Enter Valid model name")
    print("Dataset ready!")

    print("Starting training......")
    model_output_dir = base_path + f"/models/{model_name}-depth8-fullfinetuned-version2-temp"

    if "Qwen" in model_name:
        model, tokenizer, trainer = setup_and_train(model, tokenizer, train_dataset, model_output_dir)
    elif "Llama" in model_name:
        model, tokenizer, trainer, history = llama_setup_and_train(model, tokenizer, train_dataset, model_output_dir)
    else:
        raise ValueError("Enter Valid model name")
    print("Training complete!")

    print("Saving Model and Stats......")
    stats_save_dir_path =  f"/home2/...../causality_grammar/output_logs/{model_name}_version2_training_logs.csv"
    save_stats(history, stats_save_dir_path)
    model_save_dir_path = base_path + f"/models/{model_name}-depth8-fullfinetuned-version2"
    save_model(model, tokenizer, trainer, model_save_dir_path)
    print("Complete!")



if __name__ == "__main__":

    base_path = "/scratch/...."
    # model_name = "Qwen3-1.7B"
    model_name = "Llama-3.2-1B-Instruct"
    main(base_path, model_name)

