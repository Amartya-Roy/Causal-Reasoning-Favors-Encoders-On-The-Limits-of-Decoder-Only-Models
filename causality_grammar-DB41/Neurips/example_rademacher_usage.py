#!/usr/bin/env python3
"""
Quick example: Using the Rademacher projection estimator
"""

import os
os.environ['ACCELERATE_USE_CPU'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pandas as pd
from verify_assumption5 import (
    load_model_and_tokenizer,
    generate_near_misses,
    estimate_jacobian_proximity
)
from peft import LoraConfig, get_peft_model, TaskType
from visualize_results import plot_jacobian_proximity_results

# Load one sample
df = pd.read_csv("data/assumption5_data.csv")
sample = df[df['label'] == 1].iloc[0]

question = sample['question']
valid_proof = sample['proof_chain']

print(f"Question: {question}")
print(f"Valid proof: {valid_proof}\n")

# Generate near-miss
near_misses = generate_near_misses(valid_proof, k=1)
near_miss = near_misses[0][0]  # Extract the near-miss proof string
print(f"Near-miss: {near_miss}\n")

# Load model with LoRA
print("Loading model...")
model, tokenizer = load_model_and_tokenizer(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    device="cpu"
)
# Note: LoRA is already created inside load_model_and_tokenizer()

# Estimate Jacobian proximity
print("Estimating Jacobian proximity with K=10 projections...")
metrics = estimate_jacobian_proximity(
    model=model,
    tokenizer=tokenizer,
    question=question,
    y_plus=valid_proof,
    y_minus=near_miss,
    K=10,
    device="cpu",
    seed=42
)

# Print results
print("\nResults:")
print(f"  Cosine similarity: {metrics['cosine_sim_mean']:.4f} ± {metrics['cosine_sim_std']:.4f}")
print(f"  95% CI: [{metrics['cosine_sim_ci_lower']:.4f}, {metrics['cosine_sim_ci_upper']:.4f}]")
print(f"\n  Relative difference: {metrics['rel_diff_mean']:.4f} ± {metrics['rel_diff_std']:.4f}")
print(f"  95% CI: [{metrics['rel_diff_ci_lower']:.4f}, {metrics['rel_diff_ci_upper']:.4f}]")

print("\nInterpretation:")
if metrics['cosine_sim_mean'] > 0.8:
    print("  ✓ High cosine similarity - gradients are very similar!")
else:
    print("  ✗ Low cosine similarity - gradients differ significantly")

if metrics['rel_diff_mean'] < 0.3:
    print("  ✓ Low relative difference - near-miss is close to valid proof")
else:
    print("  ✗ High relative difference - near-miss diverges from valid proof")

# Generate visualization
print("\n" + "="*80)
print("Generating visualization...")
print("="*80)

plot_jacobian_proximity_results(
    metrics_near_miss=metrics,
    save_path="results/jacobian_proximity_analysis.png",
    show=True
)

print("\nAnalysis complete!")

