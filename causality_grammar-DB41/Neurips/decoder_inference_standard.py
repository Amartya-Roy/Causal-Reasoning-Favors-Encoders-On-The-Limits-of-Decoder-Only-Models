"""
End-to-end inference script for Standard Model: Direct Generation

Pipeline:
  Single Stage: Given Question only -> generate proof chain + extract label

This is for the STANDARD model trained on Question → Complex_CoT (no repair task).
The model generates proof chains directly, then we extract labels using rule-based methods.

Outputs:
  - model-proof-chains.json   (generated proof chains)
  - model-preds.json          (extracted labels from proofs)
  - labels-and-depths.json    (ground truth labels + depths)
  - evaluation_metrics.json   (accuracy overall and by depth)

Notes:
  - Dataset should have columns: Question, Response, Depth
  - Uses Unsloth FastLanguageModel for Qwen
  - Multi-GPU: Set use_multi_gpu=True for DataParallel
"""

import os
import re
import json
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from unsloth import FastLanguageModel
from extract_label_from_proof import extract_label_from_symbolic_proof

# Set environment to avoid compile issues
os.environ["UNSLOTH_COMPILE"] = "0"


# ----------------------------
# Environment / Paths
# ----------------------------
def set_env(base_path: str):
    os.environ["HF_HOME"] = base_path
    os.environ["HF_TRANSFORMERS_CACHE"] = os.path.join(base_path, "hf_cache")
    os.environ["HF_HUB_CACHE"] = os.path.join(base_path, "hf_hub")
    os.environ["HF_XET_CACHE"] = os.path.join(base_path, "hf_xet")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(base_path, "datasets")


# ----------------------------
# Prompting
# ----------------------------
def create_generation_prompt(question_text: str) -> str:
    """
    Create prompt for standard model to generate proof chain directly from question.
    This matches the training prompt format.
    """
    prompt = f"""You are a logical reasoning system. Solve this first-order logic problem.

Given: Facts, Rules, and a Query.
Task: Generate a proof chain to determine if the query is true.

Use symbolic notation:
- ∧ for AND, ⇒ for IMPLY
- Format: "Proof chain:" followed by the inference steps

{question_text}

OUTPUT:"""
    return prompt


# Use the fixed label extraction function
extract_label_from_proof_chain = extract_label_from_symbolic_proof


# ----------------------------
# Model / Data
# ----------------------------
def load_model(model_path: str, cache_dir: str, use_multi_gpu: bool = False):
    """
    Load the standard finetuned model.
    use_multi_gpu: If True, wrap model in DataParallel for multi-GPU inference
    """
    model_name = model_path.strip() if model_path and model_path.strip() else "unsloth/Qwen3-1.7B"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
        cache_dir=cache_dir,
    )
    FastLanguageModel.for_inference(model)
    
    # Multi-GPU support with DataParallel
    if use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference")
        model = torch.nn.DataParallel(model)
    
    return model, tokenizer


def load_data(data_path: str):
    """Load test dataset from parquet file."""
    dataset = load_dataset("parquet", data_files=data_path)["train"]
    return dataset


def save_outputs(path: str, data):
    """Save outputs to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_text(model, tokenizer, prompts, max_new_tokens=1024, temperature=1.0, do_sample=True):
    """
    Generate text from prompts using the model.
    Handles multi-GPU models wrapped in DataParallel.
    """
    # Truncate to leave room for generation (4096 - max_new_tokens)
    max_input_length = 4096 - max_new_tokens
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length).to("cuda")
    input_lengths = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        # Access the underlying model if wrapped in DataParallel
        gen_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        output_sequences = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_k=50 if do_sample else None,
        )
    
    # Decode only the generated tokens (excluding input prompt)
    generated_only = output_sequences[:, input_lengths:]
    return tokenizer.batch_decode(generated_only, skip_special_tokens=True)


def compute_metrics(all_preds, true_labels, depths):
    """
    Compute accuracy overall and by depth.
    Returns dict with metrics.
    """
    # Filter valid predictions
    valid_indices = [i for i, p in enumerate(all_preds) if p in (0, 1)]
    
    if not valid_indices:
        return {"error": "No valid predictions"}
    
    valid_preds = [all_preds[i] for i in valid_indices]
    valid_labels = [true_labels[i] for i in valid_indices]
    valid_depths = [depths[i] for i in valid_indices]
    
    # Overall accuracy
    correct = sum(p == y for p, y in zip(valid_preds, valid_labels))
    overall_acc = correct / len(valid_preds)
    
    # Accuracy by depth
    depth_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for pred, label, depth in zip(valid_preds, valid_labels, valid_depths):
        depth_stats[depth]["total"] += 1
        if pred == label:
            depth_stats[depth]["correct"] += 1
    
    depth_accuracies = {
        depth: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        for depth, stats in depth_stats.items()
    }
    
    metrics = {
        "overall_accuracy": overall_acc,
        "total_examples": len(true_labels),
        "valid_predictions": len(valid_preds),
        "accuracy_by_depth": dict(sorted(depth_accuracies.items())),
        "samples_by_depth": {depth: stats["total"] for depth, stats in depth_stats.items()}
    }
    
    return metrics


# ----------------------------
# Main
# ----------------------------
def main(
    hf_home_path: str,
    output_base_path: str,
    dataset_path: str,
    finetuned_model_path: str,
    cache_dir: str,
    batch_size: int = 8,
    max_new_tokens: int = 1024,
    use_multi_gpu: bool = False,
):
    """
    Run inference on test dataset using standard model.
    
    Args:
        hf_home_path: Base path for HuggingFace cache
        output_base_path: Where to save results
        dataset_path: Path to test parquet file
        finetuned_model_path: Path to trained standard model
        cache_dir: Model cache directory
        batch_size: Batch size for inference
        max_new_tokens: Max tokens to generate
        use_multi_gpu: Use DataParallel for multi-GPU
    """
    set_env(hf_home_path)

    # Load model
    model, tokenizer = load_model(finetuned_model_path, cache_dir=cache_dir, use_multi_gpu=use_multi_gpu)
    device = model.module.device if isinstance(model, torch.nn.DataParallel) else model.device
    print(f"Model loaded. Device: {device}")

    # Load dataset
    ds = load_data(dataset_path)

    # Extract fields
    questions = [ex["Question"] for ex in ds]
    # Convert Response: "Yes" -> 1, "No" -> 0, or handle integers directly
    true_labels = [1 if str(ex["Response"]).strip().lower() == "yes" else 0 for ex in ds]
    depths = [int(ex["Depth"]) for ex in ds] if "Depth" in ds.column_names else [-1] * len(ds)

    # Save ground truth
    save_outputs(os.path.join(output_base_path, "labels-and-depths.json"), [true_labels, depths])

    # Track depth completion - group examples by depth for sequential processing
    depth_to_indices = defaultdict(list)
    for idx, depth in enumerate(depths):
        depth_to_indices[depth].append(idx)
    
    # Storage for all outputs
    all_proof_chains = [''] * len(questions)
    all_preds = [-1] * len(questions)
    
    # Process examples depth by depth
    sorted_depths = sorted(depth_to_indices.keys(), reverse=True)  # Start from highest depth
    
    for depth in sorted_depths:
        indices = depth_to_indices[depth]
        depth_questions = [questions[i] for i in indices]
        depth_labels = [true_labels[i] for i in indices]
        
        print(f"\n{'='*80}")
        print(f"Processing Depth {depth}: {len(indices)} examples")
        print(f"{'='*80}")
        
        # Process this depth in batches
        depth_data = list(zip(depth_questions, depth_labels, indices))
        
        depth_preds = []
        for i in tqdm(range(0, len(depth_data), batch_size), desc=f"Depth {depth}", leave=False):
            batch = depth_data[i:i+batch_size]
            batch_questions, batch_labels, batch_indices = zip(*batch)
            
            # Generate proof chains from questions
            generation_prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": create_generation_prompt(q)}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for q in batch_questions
            ]
            
            # Generate proof chains
            proof_outputs = generate_text(model, tokenizer, generation_prompts, max_new_tokens=max_new_tokens)
            
            # Extract labels from proof chains (RULE-BASED)
            batch_preds = [extract_label_from_proof_chain(proof) for proof in proof_outputs]
            depth_preds.extend(batch_preds)
            
            # Store results
            for idx, proof in zip(batch_indices, proof_outputs):
                all_proof_chains[idx] = proof
                
            for idx, pred in zip(batch_indices, batch_preds):
                all_preds[idx] = pred
        
        # Compute accuracy for this depth
        valid_preds = [p for p in depth_preds if p in (0, 1)]
        if valid_preds:
            correct = sum(p == l for p, l in zip(depth_preds, depth_labels) if p in (0, 1))
            acc = correct / len(valid_preds)
            print(f"\n{'='*80}")
            print(f"Depth {depth} COMPLETED")
            print(f"  Accuracy: {acc:.4f} ({correct}/{len(valid_preds)} correct)")
            print(f"  Valid predictions: {len(valid_preds)}/{len(depth_preds)}")
            print(f"{'='*80}\n")
        else:
            print(f"\n{'='*80}")
            print(f"Depth {depth} COMPLETED - No valid predictions!")
            print(f"{'='*80}\n")
    
    # Final save
    save_outputs(os.path.join(output_base_path, "model-proof-chains.json"), all_proof_chains)
    save_outputs(os.path.join(output_base_path, "model-preds.json"), all_preds)

    # Compute and save metrics
    metrics = compute_metrics(all_preds, true_labels, depths)
    save_outputs(os.path.join(output_base_path, "evaluation_metrics.json"), metrics)
    
    print("\n" + "="*80)
    print("EVALUATION METRICS")
    print("="*80)
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Valid Predictions: {metrics['valid_predictions']}/{metrics['total_examples']}")
    print("\nAccuracy by Depth:")
    for depth in sorted(metrics['accuracy_by_depth'].keys()):
        acc = metrics['accuracy_by_depth'][depth]
        n_samples = metrics['samples_by_depth'][depth]
        print(f"  Depth {depth}: {acc:.4f} ({n_samples} samples)")
    print("="*80)


if __name__ == "__main__":
    hf_home_path = "/home/omy1cob/Causal_LLM/causality_grammar-DB41"
    output_base_path = "/home/omy1cob/Causal_LLM/causality_grammar-DB41/results/inference_hf_bashcache_depth0-50"
    # Use the extended depth 0-50 test dataset
    dataset_path = "/home/omy1cob/Causal_LLM/causality_grammar-DB41/data/alice_test_depth0-50_complete.parquet"


    # HuggingFace model from BashCache/Encoder-Decoder-Experiments
    finetuned_model_path = "BashCache/Logical_Reasoning_Qwen"

    # Unsloth cache dir
    cache_dir = "/home/omy1cob/Causal_LLM/causality_grammar-DB41/models"

    main(
        hf_home_path=hf_home_path,
        output_base_path=output_base_path,
        dataset_path=dataset_path,
        finetuned_model_path=finetuned_model_path,
        cache_dir=cache_dir,
        batch_size=8,
        max_new_tokens=1024,
        use_multi_gpu=True,  # Enable multi-GPU inference
    )
