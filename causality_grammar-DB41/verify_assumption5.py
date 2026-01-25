#!/usr/bin/env python3
"""
Empirical Verification of Assumption 5 (Near-Miss Gradient Proximity)
======================================================================
Tests whether near-miss negatives (single-edit corruptions of valid proofs)
have LoRA gradients that are closer to the positive gradient than random/hard negatives.

Usage:
    python verify_assumption5.py \
        --dataset_path data/causal_proofs.csv \
        --model_name Qwen/Qwen2.5-1.5B-Instruct \
        --output_dir results/assumption5 \
        --n_samples 100 \
        --k_near_miss 3
"""

import argparse
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Transformers and PEFT
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, PeftModel, TaskType

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical tests
from scipy import stats
from scipy.spatial.distance import cosine as cosine_distance

# Edit distance
try:
    import Levenshtein
except ImportError:
    print("Installing python-Levenshtein...")
    import subprocess
    subprocess.check_call(["pip", "install", "python-Levenshtein"])
    import Levenshtein


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class GradientPair:
    """Container for gradient pair analysis results."""
    pair_type: str  # "near_miss", "random", "hard"
    question: str
    proof_positive: str
    proof_negative: str
    edit_distance: int
    cosine_sim: float
    rel_diff_r: float
    margin_delta: float
    logp_positive: float
    logp_negative: float
    len_positive: int
    len_negative: int


# =============================================================================
# Near-Miss Generation Functions
# =============================================================================

def tokenize_proof(proof: str) -> List[str]:
    """Tokenize proof into steps/tokens for editing."""
    # Split by newlines or semicolons (common step delimiters)
    steps = re.split(r'[\n;]+', proof.strip())
    return [s.strip() for s in steps if s.strip()]


def reconstruct_proof(steps: List[str]) -> str:
    """Reconstruct proof from steps."""
    return '\n'.join(steps)


def delete_one_step(proof: str) -> Optional[str]:
    """Near-miss edit: delete exactly one step from the proof."""
    steps = tokenize_proof(proof)
    if len(steps) <= 1:
        return None  # Cannot delete from single-step proof
    idx = random.randint(0, len(steps) - 1)
    new_steps = steps[:idx] + steps[idx+1:]
    return reconstruct_proof(new_steps)


def swap_entity_token(proof: str) -> Optional[str]:
    """Near-miss edit: swap one entity token with another from the proof."""
    # Find entity-like tokens (capitalized words, variables like X, Y, A, B)
    entity_pattern = r'\b([A-Z][a-zA-Z0-9_]*)\b'
    entities = re.findall(entity_pattern, proof)
    
    if len(entities) < 2:
        return None
    
    # Pick two different entities to swap
    unique_entities = list(set(entities))
    if len(unique_entities) < 2:
        return None
    
    e1, e2 = random.sample(unique_entities, 2)
    
    # Swap first occurrence of e1 with e2
    new_proof = proof.replace(e1, "__TEMP__", 1)
    new_proof = new_proof.replace("__TEMP__", e2, 1)
    
    return new_proof if new_proof != proof else None


def flip_relation_token(proof: str) -> Optional[str]:
    """Near-miss edit: flip a causal/logical relation token."""
    relation_flips = {
        'causes': 'prevents',
        'prevents': 'causes',
        'implies': 'contradicts',
        'contradicts': 'implies',
        'leads to': 'does not lead to',
        'does not lead to': 'leads to',
        '->': '<-',
        '<-': '->',
        '=>': '<=',
        '<=': '=>',
        'therefore': 'however',
        'because': 'although',
        'if': 'unless',
        'and': 'or',
        'increases': 'decreases',
        'decreases': 'increases',
        'activates': 'inhibits',
        'inhibits': 'activates',
        'True': 'False',
        'False': 'True',
        'positive': 'negative',
        'negative': 'positive',
    }
    
    proof_lower = proof.lower()
    for old, new in relation_flips.items():
        if old.lower() in proof_lower:
            # Case-insensitive replacement of first occurrence
            pattern = re.compile(re.escape(old), re.IGNORECASE)
            new_proof = pattern.sub(new, proof, count=1)
            if new_proof != proof:
                return new_proof
    
    return None


def make_near_miss(proof: str, edit_type: Optional[str] = None) -> Tuple[Optional[str], str, int]:
    """
    Create a near-miss negative by applying exactly one edit.
    
    Args:
        proof: Original valid proof string
        edit_type: Optional specific edit type ("delete", "swap_entity", "flip_relation")
                   If None, randomly choose one that succeeds.
    
    Returns:
        Tuple of (near_miss_proof, edit_type_used, token_edit_distance)
        Returns (None, "", 0) if no edit could be applied.
    """
    edit_functions = {
        'delete': delete_one_step,
        'swap_entity': swap_entity_token,
        'flip_relation': flip_relation_token,
    }
    
    if edit_type:
        func = edit_functions.get(edit_type)
        if func:
            result = func(proof)
            if result:
                edit_dist = Levenshtein.distance(proof, result)
                return result, edit_type, edit_dist
        return None, "", 0
    
    # Try each edit type in random order until one succeeds
    edit_types = list(edit_functions.keys())
    random.shuffle(edit_types)
    
    for et in edit_types:
        result = edit_functions[et](proof)
        if result:
            edit_dist = Levenshtein.distance(proof, result)
            return result, et, edit_dist
    
    return None, "", 0


def generate_near_misses(proof: str, k: int = 3) -> List[Tuple[str, str, int]]:
    """
    Generate up to K near-miss negatives for a proof.
    
    Returns:
        List of (near_miss_proof, edit_type, edit_distance) tuples
    """
    near_misses = []
    edit_types = ['delete', 'swap_entity', 'flip_relation']
    
    # Try to get one of each type first
    for et in edit_types:
        if len(near_misses) >= k:
            break
        result, actual_type, edit_dist = make_near_miss(proof, et)
        if result:
            near_misses.append((result, actual_type, edit_dist))
    
    # If we need more, try random edits
    attempts = 0
    while len(near_misses) < k and attempts < k * 3:
        result, actual_type, edit_dist = make_near_miss(proof)
        if result and result not in [nm[0] for nm in near_misses]:
            near_misses.append((result, actual_type, edit_dist))
        attempts += 1
    
    return near_misses


# =============================================================================
# Control Negative Generation
# =============================================================================

def compute_lexical_overlap(proof1: str, proof2: str) -> float:
    """Compute Jaccard similarity between token sets."""
    tokens1 = set(proof1.lower().split())
    tokens2 = set(proof2.lower().split())
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    return intersection / union if union > 0 else 0.0


def make_controls(
    question: str,
    positive_proof: str,
    df: pd.DataFrame,
    near_miss_edit_dist: int
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Create random and hard negative controls for a positive sample.
    
    Args:
        question: The question/context
        positive_proof: The valid proof (label=1)
        df: Full dataset DataFrame
        near_miss_edit_dist: Edit distance of the near-miss for comparison
    
    Returns:
        Tuple of (random_negative_info, hard_negative_info)
        Each is a dict with keys: proof, edit_distance, overlap
    """
    # Get negatives (label=0) from different questions
    negatives = df[(df['label'] == 0) & (df['question'] != question)]
    
    if len(negatives) == 0:
        return None, None
    
    # Random negative: just pick any negative from a different question
    random_neg_row = negatives.sample(n=1).iloc[0]
    random_neg = {
        'proof': random_neg_row['proof_chain'],
        'edit_distance': Levenshtein.distance(positive_proof, random_neg_row['proof_chain']),
        'overlap': compute_lexical_overlap(positive_proof, random_neg_row['proof_chain'])
    }
    
    # Hard negative: high lexical overlap but larger edit distance than near-miss
    negatives_with_overlap = []
    for _, row in negatives.iterrows():
        overlap = compute_lexical_overlap(positive_proof, row['proof_chain'])
        edit_dist = Levenshtein.distance(positive_proof, row['proof_chain'])
        if edit_dist > near_miss_edit_dist:  # Must have larger edit distance
            negatives_with_overlap.append({
                'proof': row['proof_chain'],
                'edit_distance': edit_dist,
                'overlap': overlap
            })
    
    if negatives_with_overlap:
        # Sort by overlap (descending) and pick highest
        negatives_with_overlap.sort(key=lambda x: x['overlap'], reverse=True)
        hard_neg = negatives_with_overlap[0]
    else:
        # Fallback: use random negative as hard negative too
        hard_neg = random_neg.copy()
    
    return random_neg, hard_neg


# =============================================================================
# Model and Gradient Computation
# =============================================================================

def load_model_and_tokenizer(
    model_name: str,
    lora_checkpoint: Optional[str] = None,
    device: str = "cuda"
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load Qwen model with QLoRA (4-bit quantization) or CPU mode.
    
    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-1.5B-Instruct")
        lora_checkpoint: Optional path to existing LoRA adapter
        device: Device to load model on ("cuda" or "cpu")
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check if GPU is available
    use_quantization = device == "cuda" and torch.cuda.is_available()
    
    if use_quantization:
        print("Using 4-bit quantization (QLoRA mode)")
        # Quantization config for 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load base model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    else:
        print("Using CPU mode (no quantization) - this may be slow for large models")
        # Load model without quantization for CPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None,  # Don't use device_map for CPU
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True,
        )
        model = model.to(device)
    
    if lora_checkpoint and os.path.exists(lora_checkpoint):
        # Load existing LoRA adapter
        print(f"Loading LoRA adapter from: {lora_checkpoint}")
        model = PeftModel.from_pretrained(model, lora_checkpoint)
    else:
        # Create new LoRA adapter
        print("Creating new LoRA adapter...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    
    return model, tokenizer


def compute_logp_and_grad(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    question: str,
    proof: str,
    device: str = "cuda"
) -> Tuple[float, torch.Tensor]:
    """
    Compute log probability and LoRA gradient for a (question, proof) pair.
    Uses teacher forcing: computes log p(proof | question).
    
    Args:
        model: The model with LoRA adapters
        tokenizer: Tokenizer
        question: The question/context
        proof: The proof/completion
        device: Compute device
    
    Returns:
        Tuple of (log_probability, flattened_gradient_vector)
    """
    model.train()  # Enable gradient computation
    model.zero_grad()
    
    # Format input: question followed by proof
    prompt = f"Question: {question}\nProof: "
    full_text = prompt + proof
    
    # Tokenize
    prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    full_tokens = tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
    
    input_ids = full_tokens["input_ids"].to(device)
    attention_mask = full_tokens["attention_mask"].to(device)
    
    # Get prompt length to mask loss
    prompt_len = prompt_tokens["input_ids"].shape[1]
    
    # Forward pass with autocast for mixed precision
    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Compute log probability only for proof tokens (after prompt)
        logits = outputs.logits[:, :-1, :].float()  # Cast to float32 for numerical stability
        labels = input_ids[:, 1:]  # Shifted labels
        
        # Mask prompt tokens
        mask = torch.zeros_like(labels, dtype=torch.float32, device=device)
        mask[:, prompt_len-1:] = 1.0  # Only compute for proof tokens
        
        # Log probabilities for each token
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        
        # Average log probability over proof tokens
        masked_log_probs = token_log_probs * mask
        total_log_prob = masked_log_probs.sum() / (mask.sum() + 1e-8)
    
    # Backward pass to compute gradients (outside autocast for stable gradients)
    total_log_prob.backward()
    
    # Extract LoRA gradients and flatten
    grad_list = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_list.append(param.grad.detach().view(-1).float())
    
    if grad_list:
        flat_grad = torch.cat(grad_list)
    else:
        flat_grad = torch.zeros(1, device=device)
    
    model.zero_grad()
    
    return total_log_prob.item(), flat_grad.cpu()


def compute_gradient_metrics(
    grad_pos: torch.Tensor,
    grad_neg: torch.Tensor
) -> Tuple[float, float]:
    """
    Compute metrics between positive and negative gradients.
    
    Args:
        grad_pos: Gradient vector for positive sample
        grad_neg: Gradient vector for negative sample
    
    Returns:
        Tuple of (cosine_similarity, relative_difference_r)
    """
    # Ensure same device and type
    grad_pos = grad_pos.float()
    grad_neg = grad_neg.float()
    
    # Cosine similarity
    norm_pos = torch.norm(grad_pos)
    norm_neg = torch.norm(grad_neg)
    
    if norm_pos < 1e-8 or norm_neg < 1e-8:
        cosine_sim = 0.0
    else:
        cosine_sim = torch.dot(grad_pos, grad_neg) / (norm_pos * norm_neg)
        cosine_sim = cosine_sim.item()
    
    # Relative difference: r = |g+ - g-| / (|g+| + |g-|)
    diff_norm = torch.norm(grad_pos - grad_neg)
    sum_norms = norm_pos + norm_neg
    
    if sum_norms < 1e-8:
        rel_diff = 0.0
    else:
        rel_diff = (diff_norm / sum_norms).item()
    
    return cosine_sim, rel_diff


# =============================================================================
# Statistical Analysis and Visualization
# =============================================================================

def run_stats_and_plots(
    results: List[GradientPair],
    output_dir: str
) -> None:
    """
    Run statistical tests and generate visualization plots.
    
    Args:
        results: List of GradientPair results
        output_dir: Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'pair_type': r.pair_type,
            'question': r.question[:50] + "..." if len(r.question) > 50 else r.question,
            'edit_distance': r.edit_distance,
            'cosine_sim': r.cosine_sim,
            'rel_diff_r': r.rel_diff_r,
            'margin_delta': r.margin_delta,
            'logp_positive': r.logp_positive,
            'logp_negative': r.logp_negative,
            'len_positive': r.len_positive,
            'len_negative': r.len_negative,
        }
        for r in results
    ])
    
    # Save CSV
    csv_path = os.path.join(output_dir, "assumption5_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # --------------------------------------------------------------------------
    # F1: Histogram/CDF of edit distances for near-miss negatives
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    near_miss_df = df[df['pair_type'] == 'near_miss']
    
    # Histogram
    axes[0].hist(near_miss_df['edit_distance'], bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Token Edit Distance')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Edit Distance Distribution (Near-Miss)')
    
    # CDF
    sorted_dist = np.sort(near_miss_df['edit_distance'])
    cdf = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist)
    axes[1].plot(sorted_dist, cdf, marker='.', linestyle='none')
    axes[1].set_xlabel('Token Edit Distance')
    axes[1].set_ylabel('CDF')
    axes[1].set_title('Cumulative Distribution of Edit Distance')
    
    plt.tight_layout()
    f1_path = os.path.join(output_dir, "F1_edit_distance_distribution.png")
    plt.savefig(f1_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {f1_path}")
    
    # --------------------------------------------------------------------------
    # F2: Boxplot of cosine similarity by pair type
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    order = ['near_miss', 'hard', 'random']
    sns.boxplot(data=df, x='pair_type', y='cosine_sim', order=order, ax=ax)
    sns.stripplot(data=df, x='pair_type', y='cosine_sim', order=order, 
                  color='black', alpha=0.3, size=4, ax=ax)
    
    ax.set_xlabel('Pair Type')
    ax.set_ylabel('Cosine Similarity (g⁺, g⁻)')
    ax.set_title('Gradient Cosine Similarity by Pair Type\n(Higher = More Similar)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    f2_path = os.path.join(output_dir, "F2_cosine_similarity_boxplot.png")
    plt.savefig(f2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {f2_path}")
    
    # --------------------------------------------------------------------------
    # F3: Boxplot of relative difference r by pair type
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(data=df, x='pair_type', y='rel_diff_r', order=order, ax=ax)
    sns.stripplot(data=df, x='pair_type', y='rel_diff_r', order=order,
                  color='black', alpha=0.3, size=4, ax=ax)
    
    ax.set_xlabel('Pair Type')
    ax.set_ylabel('Relative Difference r = |g⁺ - g⁻| / (|g⁺| + |g⁻|)')
    ax.set_title('Gradient Relative Difference by Pair Type\n(Lower = More Similar)')
    
    plt.tight_layout()
    f3_path = os.path.join(output_dir, "F3_relative_difference_boxplot.png")
    plt.savefig(f3_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {f3_path}")
    
    # --------------------------------------------------------------------------
    # F4: Distribution of margin delta by pair type
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for pt in order:
        subset = df[df['pair_type'] == pt]['margin_delta']
        sns.kdeplot(subset, label=pt, ax=ax, fill=True, alpha=0.3)
    
    ax.set_xlabel('Margin Δ = log p(y⁺|x) - log p(y⁻|x)')
    ax.set_ylabel('Density')
    ax.set_title('Log-Probability Margin Distribution by Pair Type')
    ax.legend(title='Pair Type')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    f4_path = os.path.join(output_dir, "F4_margin_delta_distribution.png")
    plt.savefig(f4_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {f4_path}")
    
    # --------------------------------------------------------------------------
    # Statistical Tests
    # --------------------------------------------------------------------------
    print("\n" + "="*70)
    print("STATISTICAL TESTS: Assumption 5 (Near-Miss Gradient Proximity)")
    print("="*70)
    
    near_miss = df[df['pair_type'] == 'near_miss']
    random_neg = df[df['pair_type'] == 'random']
    hard_neg = df[df['pair_type'] == 'hard']
    
    def wilcoxon_with_effect_size(x, y, name):
        """Compute Wilcoxon test with effect size."""
        # Match by index (paired test)
        n = min(len(x), len(y))
        x_vals = x.values[:n]
        y_vals = y.values[:n]
        
        try:
            stat, p_value = stats.wilcoxon(x_vals, y_vals, alternative='two-sided')
            # Effect size: r = Z / sqrt(N)
            z = stats.norm.ppf(1 - p_value/2) if p_value > 0 else np.inf
            effect_size = z / np.sqrt(n)
        except Exception as e:
            p_value = np.nan
            effect_size = np.nan
            stat = np.nan
        
        print(f"\n{name}:")
        print(f"  Wilcoxon statistic = {stat:.4f}")
        print(f"  p-value = {p_value:.6f}")
        print(f"  Effect size (r) = {effect_size:.4f}")
        print(f"  Interpretation: {'*SIGNIFICANT*' if p_value < 0.05 else 'Not significant'} at α=0.05")
        
        return p_value, effect_size
    
    print("\n--- Cosine Similarity Comparisons ---")
    print("(Higher cosine = more similar gradients)")
    print(f"Near-miss mean: {near_miss['cosine_sim'].mean():.4f} ± {near_miss['cosine_sim'].std():.4f}")
    print(f"Random mean: {random_neg['cosine_sim'].mean():.4f} ± {random_neg['cosine_sim'].std():.4f}")
    print(f"Hard mean: {hard_neg['cosine_sim'].mean():.4f} ± {hard_neg['cosine_sim'].std():.4f}")
    
    wilcoxon_with_effect_size(
        near_miss['cosine_sim'], random_neg['cosine_sim'],
        "Near-miss vs Random (Cosine Similarity)"
    )
    wilcoxon_with_effect_size(
        near_miss['cosine_sim'], hard_neg['cosine_sim'],
        "Near-miss vs Hard (Cosine Similarity)"
    )
    
    print("\n--- Relative Difference (r) Comparisons ---")
    print("(Lower r = more similar gradients)")
    print(f"Near-miss mean: {near_miss['rel_diff_r'].mean():.4f} ± {near_miss['rel_diff_r'].std():.4f}")
    print(f"Random mean: {random_neg['rel_diff_r'].mean():.4f} ± {random_neg['rel_diff_r'].std():.4f}")
    print(f"Hard mean: {hard_neg['rel_diff_r'].mean():.4f} ± {hard_neg['rel_diff_r'].std():.4f}")
    
    wilcoxon_with_effect_size(
        near_miss['rel_diff_r'], random_neg['rel_diff_r'],
        "Near-miss vs Random (Relative Difference)"
    )
    wilcoxon_with_effect_size(
        near_miss['rel_diff_r'], hard_neg['rel_diff_r'],
        "Near-miss vs Hard (Relative Difference)"
    )
    
    print("\n--- Margin Delta Comparisons ---")
    print(f"Near-miss mean: {near_miss['margin_delta'].mean():.4f} ± {near_miss['margin_delta'].std():.4f}")
    print(f"Random mean: {random_neg['margin_delta'].mean():.4f} ± {random_neg['margin_delta'].std():.4f}")
    print(f"Hard mean: {hard_neg['margin_delta'].mean():.4f} ± {hard_neg['margin_delta'].std():.4f}")
    
    print("\n" + "="*70)
    print("ASSUMPTION 5 VERIFICATION SUMMARY")
    print("="*70)
    
    # Check if assumption holds
    nm_cos = near_miss['cosine_sim'].mean()
    rand_cos = random_neg['cosine_sim'].mean()
    hard_cos = hard_neg['cosine_sim'].mean()
    
    nm_r = near_miss['rel_diff_r'].mean()
    rand_r = random_neg['rel_diff_r'].mean()
    hard_r = hard_neg['rel_diff_r'].mean()
    
    cos_check = nm_cos > rand_cos and nm_cos > hard_cos
    r_check = nm_r < rand_r and nm_r < hard_r
    
    print(f"\n✓ Cosine Similarity: Near-miss > Random? {nm_cos:.4f} > {rand_cos:.4f} = {nm_cos > rand_cos}")
    print(f"✓ Cosine Similarity: Near-miss > Hard? {nm_cos:.4f} > {hard_cos:.4f} = {nm_cos > hard_cos}")
    print(f"✓ Relative Diff: Near-miss < Random? {nm_r:.4f} < {rand_r:.4f} = {nm_r < rand_r}")
    print(f"✓ Relative Diff: Near-miss < Hard? {nm_r:.4f} < {hard_r:.4f} = {nm_r < hard_r}")
    
    if cos_check and r_check:
        print("\n★ ASSUMPTION 5 APPEARS TO HOLD: Near-miss pairs have significantly")
        print("  more similar gradients than both random and hard negatives.")
    elif cos_check or r_check:
        print("\n◐ ASSUMPTION 5 PARTIALLY HOLDS: Some metrics support the assumption,")
        print("  but results are mixed. See detailed statistics above.")
    else:
        print("\n✗ ASSUMPTION 5 DOES NOT HOLD: Near-miss pairs do not show")
        print("  significantly more similar gradients than controls.")
    
    print("="*70 + "\n")
    
    # Save summary to file
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Assumption 5 Verification Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total pairs analyzed: {len(df)}\n")
        f.write(f"Near-miss pairs: {len(near_miss)}\n")
        f.write(f"Random pairs: {len(random_neg)}\n")
        f.write(f"Hard pairs: {len(hard_neg)}\n\n")
        f.write("Mean Cosine Similarity:\n")
        f.write(f"  Near-miss: {nm_cos:.4f}\n")
        f.write(f"  Random: {rand_cos:.4f}\n")
        f.write(f"  Hard: {hard_cos:.4f}\n\n")
        f.write("Mean Relative Difference:\n")
        f.write(f"  Near-miss: {nm_r:.4f}\n")
        f.write(f"  Random: {rand_r:.4f}\n")
        f.write(f"  Hard: {hard_r:.4f}\n")
    
    print(f"Summary saved to: {summary_path}")


# =============================================================================
# Main Pipeline
# =============================================================================

def run_assumption5_verification(
    dataset_path: str,
    model_name: str,
    output_dir: str,
    lora_checkpoint: Optional[str] = None,
    n_samples: int = 100,
    k_near_miss: int = 3,
    seed: int = 42
) -> List[GradientPair]:
    """
    Main function to run the full Assumption 5 verification pipeline.
    
    Args:
        dataset_path: Path to CSV with columns: question, proof_chain, label
        model_name: HuggingFace model name
        output_dir: Directory for outputs
        lora_checkpoint: Optional path to LoRA adapter
        n_samples: Number of positive samples to analyze
        k_near_miss: Number of near-miss negatives per positive
        seed: Random seed
    
    Returns:
        List of GradientPair results
    """
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Validate columns
    required_cols = ['question', 'proof_chain', 'label']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    print(f"Dataset size: {len(df)}")
    print(f"Positives: {len(df[df['label'] == 1])}")
    print(f"Negatives: {len(df[df['label'] == 0])}")
    
    # Sample positives
    positives = df[df['label'] == 1].sample(n=min(n_samples, len(df[df['label'] == 1])), random_state=seed)
    print(f"\nSampled {len(positives)} positives for analysis")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model, tokenizer = load_model_and_tokenizer(model_name, lora_checkpoint, device)
    
    # Process samples
    results = []
    
    for idx, (_, row) in enumerate(tqdm(positives.iterrows(), total=len(positives), desc="Processing samples")):
        question = row['question']
        positive_proof = row['proof_chain']
        
        # Compute gradient for positive
        try:
            logp_pos, grad_pos = compute_logp_and_grad(model, tokenizer, question, positive_proof, device)
        except Exception as e:
            print(f"Error computing gradient for positive {idx}: {e}")
            continue
        
        # Generate near-miss negatives
        near_misses = generate_near_misses(positive_proof, k=k_near_miss)
        
        if not near_misses:
            print(f"Could not generate near-misses for sample {idx}")
            continue
        
        # Get controls (using first near-miss edit distance as reference)
        avg_near_miss_dist = sum(nm[2] for nm in near_misses) // len(near_misses)
        random_neg, hard_neg = make_controls(question, positive_proof, df, avg_near_miss_dist)
        
        # Process each near-miss
        for nm_proof, nm_type, nm_edit_dist in near_misses:
            try:
                logp_nm, grad_nm = compute_logp_and_grad(model, tokenizer, question, nm_proof, device)
                cos_sim, rel_diff = compute_gradient_metrics(grad_pos, grad_nm)
                
                results.append(GradientPair(
                    pair_type="near_miss",
                    question=question,
                    proof_positive=positive_proof,
                    proof_negative=nm_proof,
                    edit_distance=nm_edit_dist,
                    cosine_sim=cos_sim,
                    rel_diff_r=rel_diff,
                    margin_delta=logp_pos - logp_nm,
                    logp_positive=logp_pos,
                    logp_negative=logp_nm,
                    len_positive=len(positive_proof),
                    len_negative=len(nm_proof),
                ))
            except Exception as e:
                print(f"Error processing near-miss: {e}")
                continue
        
        # Process random negative
        if random_neg:
            try:
                logp_rand, grad_rand = compute_logp_and_grad(model, tokenizer, question, random_neg['proof'], device)
                cos_sim, rel_diff = compute_gradient_metrics(grad_pos, grad_rand)
                
                results.append(GradientPair(
                    pair_type="random",
                    question=question,
                    proof_positive=positive_proof,
                    proof_negative=random_neg['proof'],
                    edit_distance=random_neg['edit_distance'],
                    cosine_sim=cos_sim,
                    rel_diff_r=rel_diff,
                    margin_delta=logp_pos - logp_rand,
                    logp_positive=logp_pos,
                    logp_negative=logp_rand,
                    len_positive=len(positive_proof),
                    len_negative=len(random_neg['proof']),
                ))
            except Exception as e:
                print(f"Error processing random negative: {e}")
        
        # Process hard negative
        if hard_neg:
            try:
                logp_hard, grad_hard = compute_logp_and_grad(model, tokenizer, question, hard_neg['proof'], device)
                cos_sim, rel_diff = compute_gradient_metrics(grad_pos, grad_hard)
                
                results.append(GradientPair(
                    pair_type="hard",
                    question=question,
                    proof_positive=positive_proof,
                    proof_negative=hard_neg['proof'],
                    edit_distance=hard_neg['edit_distance'],
                    cosine_sim=cos_sim,
                    rel_diff_r=rel_diff,
                    margin_delta=logp_pos - logp_hard,
                    logp_positive=logp_pos,
                    logp_negative=logp_hard,
                    len_positive=len(positive_proof),
                    len_negative=len(hard_neg['proof']),
                ))
            except Exception as e:
                print(f"Error processing hard negative: {e}")
        
        # Clear GPU memory periodically
        if idx % 10 == 0:
            torch.cuda.empty_cache()
    
    print(f"\nTotal pairs collected: {len(results)}")
    
    # Run statistics and generate plots
    if results:
        run_stats_and_plots(results, output_dir)
    else:
        print("No results to analyze!")
    
    return results


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify Assumption 5 (Near-Miss Gradient Proximity) for causal reasoning"
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to CSV with columns: question, proof_chain, label"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default=None,
        help="Optional path to existing LoRA adapter"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/assumption5",
        help="Directory for output files"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of positive samples to analyze"
    )
    parser.add_argument(
        "--k_near_miss",
        type=int,
        default=3,
        help="Number of near-miss negatives per positive"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    results = run_assumption5_verification(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        lora_checkpoint=args.lora_checkpoint,
        n_samples=args.n_samples,
        k_near_miss=args.k_near_miss,
        seed=args.seed,
    )
    
    print(f"\nVerification complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
