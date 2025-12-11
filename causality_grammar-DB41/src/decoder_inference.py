import os, re
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer
from datasets import load_dataset

def set_env(base_path):
    models_cache_path = base_path + "/models"
    datasets_cache_path = base_path + "/datasets"

    os.environ["HF_HOME"] = base_path
    os.environ["HF_TRANSFORMERS_CACHE"] = base_path + "/hf_cache"
    os.environ["HF_HUB_CACHE"] = base_path + "/hf_hub"
    os.environ["HF_XET_CACHE"] = base_path + "/hf_xet"
    os.environ["HF_DATSETS"] = datasets_cache_path

def create_prompt(question_text):
    """Creates a standardized prompt for the models."""
    source = f"""You are evaluating a subset of first-order logic. 
In this subset, conjunctions are given by [AND], implications by [IMPLY], and separations between clauses as [PERIOD]
You will be given Facts, and Rules. Based on these, determine the truth value of the Query.
Your final answer should be 0 (if the Query is false) or 1 (if true).

Give your final answer in the following format:
Label: The label from the criterion. Only use the numbers 0 or 1.

{question_text}
"""
    return source

# def create_prompt(question_text):
#     """Creates a standardized prompt for the models."""
#     return f"Answer the question based on the provided context.\n\nYour final answer should be 0 (if False) and 1 (if True)\n\nQuestion: {question_text}\n\nAnswer:"

def parse_final_answer(text):
    """Extracts the final integer answer from the generated text."""
    match = re.search(r"Label: (\d+)", text, re.IGNORECASE)
    if match:
        try: return int(match.group(1))
        except (ValueError, IndexError): return -1
    return -1

# def parse_final_answer(text):
#     """Extracts the final integer answer from the generated text."""
#     match = re.search(r"Final Answer: (\d+)", text, re.IGNORECASE)
#     if match:
#         try: return int(match.group(1))
#         except (ValueError, IndexError): return -1
#     return -1

def get_batch_predictions(model, tokenizer, prompts):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")

    with torch.no_grad():
        output_sequences = model.generate(
            **inputs,
            max_new_tokens=1024, # changed for qwen3 trained on depth 8 dataset
        )
        outputs = model(**inputs)
    return tokenizer.batch_decode(output_sequences, skip_special_tokens=True), outputs.logits[:, -1, :].cpu()

def load_model(model_path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-1.7B",
        max_seq_length = 4096,                      # match what you trained with
        dtype = None,                               # auto-detect (fp16/bf16)
        load_in_4bit = True,                       # True if you want quantized loading
        cache_dir="/scratch/..../models"
    )

    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name = model_path, # YOUR MODEL YOU USED FOR TRAINING
    #     max_seq_length = 4096,
    #     load_in_4bit = True,
    #     rope_scaling = None,
    #     trust_remote_code=True
    # )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def load_data(data_path: str):

    dataset = load_dataset("parquet", data_files=data_path)["train"]

    return dataset

def prepare_data(dataset, tokenizer):
    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": create_prompt(ex["Question"])}],
            tokenize=False,
            add_generation_prompt=True
        )
        for ex in dataset
    ]
    all_true_labels = [ex['Response'] for ex in dataset]
    all_depths = [ex['Depth'] for ex in dataset]
    return chat_prompts, all_true_labels, all_depths

def llama_prepare_data(dataset, tokenizer):

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )

    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": create_prompt(ex["Question"])}],
            tokenize=False,
            add_generation_prompt=True
        )
        for ex in dataset
    ]
    all_true_labels = [ex['Response'] for ex in dataset]
    all_depths = [ex['Depth'] for ex in dataset]
    return chat_prompts, all_true_labels, all_depths

def save_outputs(output_file_path: str, data: list):

    with open(output_file_path, "w") as file:
        json.dump(data, file)

    return None


def main(hf_home_path, output_base_path, dataset_path, finetuned_model_path):

    set_env(hf_home_path)
    model, tokenizer = load_model(finetuned_model_path)
    print(f"Model device {model.device}")
    test_dataset = load_data(dataset_path)
    # chat_prompts, all_true_labels, all_depths = llama_prepare_data(test_dataset, tokenizer)
    chat_prompts, all_true_labels, all_depths = prepare_data(test_dataset, tokenizer)

    BATCH_SIZE = 8 
    prompt_loader = DataLoader(chat_prompts, batch_size=BATCH_SIZE)

    save_outputs(output_base_path + "/labels-and-depths.json", [all_true_labels, all_depths])
    model_proof_chains = list()
    model_preds = list()
    model_logits = list()

    for batch in (pbar := tqdm(prompt_loader, desc="Model Batches")):
        outputs, logits = get_batch_predictions(model, tokenizer, batch)
        model_proof_chains.extend(outputs)
        model_preds.extend([parse_final_answer(out) for out in outputs])
        model_logits.append(logits)

        if pbar.n < 5:
            save_outputs(output_base_path + "/model-proof-chains.json", model_proof_chains)
            save_outputs(output_base_path + "/model-preds.json", model_preds)
            all_logits = torch.cat(model_logits, dim=0)
            torch.save(all_logits, output_base_path + "/non_nl_depth_12.pt")

        if pbar.n % 100 == 0:
            # Save your data here
            save_outputs(output_base_path + "/model-proof-chains.json", model_proof_chains)
            save_outputs(output_base_path + "/model-preds.json", model_preds)
            all_logits = torch.cat(model_logits, dim=0)
            torch.save(all_logits, output_base_path + "/non_nl_depth_12.pt")


    all_logits = torch.cat(model_logits, dim=0)
    print(all_logits.shape)
    torch.save(all_logits, output_base_path + "/non_nl_depth_12.pt")

    print('Loaded Tensor')
    loaded_tensor = torch.load(output_base_path + "/non_nl_depth_12.pt")
    print('shape', loaded_tensor.shape)
    print('equality test:', torch.equal(all_logits, loaded_tensor))
    save_outputs(output_base_path + "/model-proof-chains.json", model_proof_chains)
    save_outputs(output_base_path + "/model-preds.json", model_preds)


if __name__ == "__main__":

    hf_home_path = "/scratch/..."
    output_base_path = "/home2/...../causality_grammar/model_predictions/qwen3_nonfinetuned/non-nl-depth12"
    dataset_path = "/home2/...../causality_grammar/data/datasets/non-nl-depth12.parquet"
    # dataset_path = "/home2/...../causality_grammar/data/datasets/non-nl-depth12.parquet"
    # dataset_path = "/home2/...../causality_grammar/data/datasets/non-nl-depth12.parquet"
    # finetuned_model_path = "/scratch/...../models/Llama-3.2-1B-Instruct-depth8-fullfinetuned"
    finetuned_model_path = " "
    main(hf_home_path, output_base_path, dataset_path, finetuned_model_path)



