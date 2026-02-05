"""
End-to-end inference script: Self-generate -> Repair (best, theory-consistent)

Pipeline:
  Stage 1: Given Question only -> draft proof chain (may be near-miss)
  Stage 2: Given Question + draft proof -> corrected proof chain + Label: {0,1}

Outputs:
  - model-proof-chains.json   (stage-2 outputs: corrected proof + label)
  - model-preds.json          (parsed labels)
  - labels-and-depths.json    (ground truth labels + depths)
  - evaluation_metrics.json   (accuracy overall and by depth)

Notes:
  - This script assumes the dataset has columns: Question, Response, Depth
  - Uses Unsloth FastLanguageModel for Qwen.
  - You can set SAVE_DRAFTS=True to also save stage-1 drafts for debugging/ablation.
  - Multi-GPU: Set CUDA_VISIBLE_DEVICES or use DataParallel
"""

import os, re, json
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
    # NOTE: you had a typo HF_DATSETS; leaving it out or fixing it.
    os.environ["HF_DATASETS_CACHE"] = os.path.join(base_path, "datasets")


# ----------------------------
# Prompting
# ----------------------------
def add_diffusion_masks_to_question(question: str, mask_ratio: float = 0.3) -> str:
    """
    DIFFUSION STRATEGY: Add [MASK] tokens to facts/rules (noisy input).
    The repair model must denoise and reconstruct the correct proof.
    
    This creates realistic noise by masking key tokens in the logical structure.
    """
    import random
    
    # Extract facts and rules sections
    parts = question.split('\n')
    masked_parts = []
    
    for line in parts:
        # Target lines with logical content (facts/rules, not headers)
        if any(keyword in line.lower() for keyword in ['alice', 'rule', '⇒', '∧', 'is ']):
            # Tokenize by spaces
            tokens = line.split()
            num_masks = max(1, int(len(tokens) * mask_ratio))
            
            # Randomly select positions to mask (avoid masking symbols)
            maskable_indices = [i for i, tok in enumerate(tokens) 
                              if tok not in ['⇒', '∧', 'Rule', 'Fact', ':', 'Alice', 'is']]
            
            if maskable_indices and num_masks > 0:
                mask_positions = random.sample(maskable_indices, 
                                              min(num_masks, len(maskable_indices)))
                for idx in mask_positions:
                    tokens[idx] = '[MASK]'
                masked_parts.append(' '.join(tokens))
            else:
                masked_parts.append(line)
        else:
            masked_parts.append(line)
    
    return '\n'.join(masked_parts)


def create_simple_prompt(question):
    """Simple direct prompt - just ask for proof from question."""
    return f"""You are a logical reasoning system. Solve this first-order logic problem.

Given: Facts, Rules, and a Query.
Task: Generate a proof chain to determine if the query is true.

Use symbolic notation:
- ∧ for AND, ⇒ for IMPLY
- Format: "Proof chain:" followed by the inference steps

{question}

OUTPUT:"""

def create_stage1_prompt(question: str) -> str:
    """
    Stage 1: Generate a corrupted/draft proof with NO guidance (blank prompt).
    The model is expected to produce an invalid or incomplete proof.
    """
    # Just return the question - no instructions, let model generate whatever
    return question

def create_stage2_prompt(question: str, draft_proof: str) -> str:
    """
    Stage 2: Given the corrupted draft, ask the model to repair it.
    """
    return f"""Question: {question}

Draft Proof (may contain errors):
{draft_proof}

Please repair the above proof. Provide a corrected step-by-step proof chain using the given rules and facts. Each step should clearly show which rule is applied and what new conclusion is derived.

If the query can be proven true, show the complete derivation. If it cannot be proven, explain which rule application fails and why.
"""

# Use the fixed label extraction function
extract_label_from_proof_chain = extract_label_from_symbolic_proof


def parse_final_answer(text: str) -> int:
    """Parse label from model output. Returns 0, 1, or -1 if unparseable."""
    # Try multiple patterns in order of specificity
    patterns = [
        r"Label:\s*([01])\b",
        r"Answer:\s*([01])\b",
        r"(?:^|\s)([01])(?:\s|$)",  # Any standalone 0 or 1
        r"\b([01])\b",  # Word boundary around 0 or 1
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            return int(match.group(1))
    
    # Fallback: look for keywords indicating true/false
    text_lower = text.lower()
    if any(word in text_lower for word in ['proven', 'true', 'valid', 'correct']):
        return 1
    if any(word in text_lower for word in ['not proven', 'false', 'invalid', 'incorrect', 'cannot']):
        return 0
    
    return -1


# ----------------------------
# Model / Data
# ----------------------------
def load_model(model_path: str, cache_dir: str, use_multi_gpu: bool = False):
    """
    model_path:
      - If you want base model: "unsloth/Qwen3-1.7B"
      - If you want your finetuned: set model_name=model_path
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
    dataset = load_dataset("parquet", data_files=data_path)["train"]
    return dataset


def save_outputs(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_text(model, tokenizer, prompts, max_new_tokens=1024, temperature=1.0, do_sample=True):
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


def compute_metrics_by_depth_incremental(all_preds, true_labels, depths, current_idx):
    """
    Compute metrics for predictions up to current_idx, showing progress per depth.
    """
    # Process only up to current_idx
    current_preds = all_preds[:current_idx]
    current_labels = true_labels[:current_idx]
    current_depths = depths[:current_idx]
    
    # Filter valid predictions
    valid_data = [(p, l, d) for p, l, d in zip(current_preds, current_labels, current_depths) if p in (0, 1)]
    
    if not valid_data:
        return None
    
    valid_preds, valid_labels, valid_depths = zip(*valid_data)
    
    # Overall accuracy
    correct = sum(p == l for p, l in zip(valid_preds, valid_labels))
    overall_acc = correct / len(valid_preds)
    
    # Accuracy by depth
    depth_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for pred, label, depth in zip(valid_preds, valid_labels, valid_depths):
        depth_stats[depth]["total"] += 1
        if pred == label:
            depth_stats[depth]["correct"] += 1
    
    return {
        "overall_accuracy": overall_acc,
        "valid_predictions": len(valid_preds),
        "total_processed": current_idx,
        "depth_stats": dict(sorted(depth_stats.items()))
    }


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
    max_new_tokens_stage1: int = 768,
    max_new_tokens_stage2: int = 1024,
    save_drafts: bool = True,
    use_multi_gpu: bool = False,
):
    set_env(hf_home_path)

    model, tokenizer = load_model(finetuned_model_path, cache_dir=cache_dir, use_multi_gpu=use_multi_gpu)
    device = model.module.device if isinstance(model, torch.nn.DataParallel) else model.device
    print(f"Model loaded. Device: {device}")

    ds = load_data(dataset_path)

    # Extract fields - TEST SET: no ground truth proofs
    questions = [ex["Question"] for ex in ds]
    # true_labels = [int(ex["Response"]) for ex in ds]
    true_labels = [1 if str(ex["Response"]).strip().lower() == "yes" else 0 for ex in ds]
    depths = [int(ex["Depth"]) for ex in ds] if "Depth" in ds.column_names else [-1] * len(ds)

    save_outputs(os.path.join(output_base_path, "labels-and-depths.json"), [true_labels, depths])

    # Use DataLoader over question strings
    loader = DataLoader(questions, batch_size=batch_size)

    # Check for checkpoint to resume from
    checkpoint_path = os.path.join(output_base_path, "checkpoint.json")
    start_idx = 0
    all_stage1_drafts = []
    all_stage2_outputs = []
    all_preds = []
    
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint, resuming from {checkpoint_path}")
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
            start_idx = checkpoint['batch_idx'] * batch_size
            all_stage1_drafts = checkpoint.get('stage1_drafts', [])
            all_stage2_outputs = checkpoint.get('stage2_outputs', [])
            all_preds = checkpoint.get('preds', [])
        print(f"Resuming from example {start_idx}/{len(questions)}")
        # Skip already processed examples
        loader = DataLoader(questions[start_idx:], batch_size=batch_size)

    # Track depth completion - group examples by depth for sequential processing
    depth_to_indices = defaultdict(list)
    for idx, depth in enumerate(depths):
        depth_to_indices[depth].append(idx)
    
    # Process examples depth by depth - START FROM HIGHEST DEPTH (50) for quick judgment
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
            
            # -------- STAGE 1: Generate corrupted proof (blank prompt) --------
            stage1_prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": create_stage1_prompt(q)}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for q in batch_questions
            ]
            stage1_outputs = generate_text(model, tokenizer, stage1_prompts, max_new_tokens=max_new_tokens_stage1)
            
            # -------- STAGE 2: Repair the corrupted proof --------
            stage2_prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": create_stage2_prompt(q, draft)}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for q, draft in zip(batch_questions, stage1_outputs)
            ]
            stage2_outputs = generate_text(model, tokenizer, stage2_prompts, max_new_tokens=max_new_tokens_stage2)
            
            # -------- Extract label from repaired proof (RULE-BASED) --------
            batch_preds = [extract_label_from_proof_chain(proof) for proof in stage2_outputs]
            depth_preds.extend(batch_preds)
            
            # Store stage1 drafts
            for idx, draft in zip(batch_indices, stage1_outputs):
                if idx >= len(all_stage1_drafts):
                    all_stage1_drafts.extend([''] * (idx + 1 - len(all_stage1_drafts)))
                all_stage1_drafts[idx] = draft
                    
            # Store stage2 outputs
            for idx, output in zip(batch_indices, stage2_outputs):
                if idx >= len(all_stage2_outputs):
                    all_stage2_outputs.extend([''] * (idx + 1 - len(all_stage2_outputs)))
                all_stage2_outputs[idx] = output
                
            # Store predictions
            for idx, pred in zip(batch_indices, batch_preds):
                if idx >= len(all_preds):
                    all_preds.extend([-1] * (idx + 1 - len(all_preds)))
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
    if save_drafts:
        save_outputs(os.path.join(output_base_path, "invalid-proof-chains-input.json"), all_stage1_drafts)
    save_outputs(os.path.join(output_base_path, "repaired-proof-chains.json"), all_stage2_outputs)
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
    hf_home_path = "/home/amartya/Causal_LLM/causality_grammar-DB41"
    output_base_path = "/home/amartya/Causal_LLM/causality_grammar-DB41/results/inference_repair_depth0-50"
    # dataset_path = "/home/amartya/Causal_LLM/causality_grammar-DB41/data/nl-depth12_test.parquet"
    dataset_path = "/home/amartya/Causal_LLM/causality_grammar-DB41/data/alice_test_depth0-50_complete.parquet"
   

    # Put your finetuned checkpoint here. Can be a local path or a Hugging Face Hub ID.
    # Example HF model ID: "Amartya77/LogicBench-Qwen-FT-Response"
    finetuned_model_path = "Amartya77/LogicBench-Qwen-FT-Response"
    # Unsloth cache dir (keep your existing)
    cache_dir = "/home/amartya/Causal_LLM/causality_grammar-DB41/models"

    main(
        hf_home_path=hf_home_path,
        output_base_path=output_base_path,
        dataset_path=dataset_path,
        finetuned_model_path=finetuned_model_path,
        cache_dir=cache_dir,
        batch_size=8,
        max_new_tokens_stage1=768,
        max_new_tokens_stage2=1024,
        save_drafts=True,
        use_multi_gpu=True,  # Enable multi-GPU inference
    )
