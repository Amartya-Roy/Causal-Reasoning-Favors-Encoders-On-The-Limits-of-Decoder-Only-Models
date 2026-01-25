"""
Test script to debug near-miss generation and gradient computation
for a single valid proof sample.

Set breakpoints at:
- Line 35: After loading the valid proof
- Line 42: After generating near-miss variants
- Line 58: Inside the loop where each near-miss is processed
- Line 75: After computing gradients for original proof
- Line 88: After computing gradients for each near-miss
"""

import pandas as pd
import torch
from verify_assumption5 import (
    load_model_and_tokenizer,
    generate_near_misses,
    compute_logp_and_grad,
    compute_gradient_metrics
)

def main():
    print("="*80)
    print("DEBUG: Near-Miss Generation and Gradient Computation")
    print("="*80)
    
    # Load data
    df = pd.read_csv('data/assumption5_data.csv')
    valid_samples = df[df['label'] == 1].reset_index(drop=True)
    
    # Get first valid proof with good structure
    sample_idx = 1  # Change this to test different samples
    positive_sample = valid_samples.iloc[sample_idx]
    
    # SET BREAKPOINT HERE - Inspect the original valid proof
    question = positive_sample['question']
    proof_chain = positive_sample['proof_chain']
    
    print(f"\n[STEP 1] Original Valid Proof (Sample {sample_idx})")
    print("-"*80)
    print(f"Question: {question[:150]}...")
    print(f"\nProof Chain:\n{proof_chain}")
    
    # Generate near-miss variants
    # SET BREAKPOINT HERE - See how near-misses are generated
    near_misses = generate_near_misses(proof_chain, k=3, seed=42)
    
    print(f"\n[STEP 2] Generated {len(near_misses)} Near-Miss Variants")
    print("-"*80)
    
    for idx, (strategy, near_miss_proof) in enumerate(near_misses):
        # SET BREAKPOINT HERE - Inspect each near-miss variant
        print(f"\nNear-Miss #{idx+1} (Strategy: {strategy})")
        print(f"Proof: {near_miss_proof}")
    
    # Load model
    print("\n[STEP 3] Loading Model...")
    print("-"*80)
    model, tokenizer, lora_config = load_model_and_tokenizer(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        lora_r=8,
        lora_alpha=16,
        use_quantization=False
    )
    
    # Compute gradients for original proof
    print("\n[STEP 4] Computing Gradients for Original Proof")
    print("-"*80)
    # SET BREAKPOINT HERE - Step through gradient computation
    pos_logp, pos_grad = compute_logp_and_grad(
        model=model,
        tokenizer=tokenizer,
        question=question,
        proof=proof_chain,
        lora_config=lora_config
    )
    
    print(f"Original Proof Log-Probability: {pos_logp:.4f}")
    print(f"Gradient Vector Shape: {pos_grad.shape}")
    print(f"Gradient Norm: {torch.norm(pos_grad).item():.4f}")
    
    # Compute gradients for each near-miss
    print("\n[STEP 5] Computing Gradients for Near-Miss Variants")
    print("-"*80)
    
    near_miss_results = []
    for idx, (strategy, near_miss_proof) in enumerate(near_misses):
        # SET BREAKPOINT HERE - Compare gradients for each near-miss
        nm_logp, nm_grad = compute_logp_and_grad(
            model=model,
            tokenizer=tokenizer,
            question=question,
            proof=near_miss_proof,
            lora_config=lora_config
        )
        
        # Compute similarity metrics
        cos_sim, rel_diff = compute_gradient_metrics(pos_grad, nm_grad)
        
        near_miss_results.append({
            'strategy': strategy,
            'logp': nm_logp,
            'cos_sim': cos_sim,
            'rel_diff': rel_diff
        })
        
        print(f"\nNear-Miss #{idx+1} ({strategy}):")
        print(f"  Log-Probability: {nm_logp:.4f}")
        print(f"  Cosine Similarity: {cos_sim:.4f}")
        print(f"  Relative Difference: {rel_diff:.4f}")
    
    # Summary
    print("\n[STEP 6] Summary")
    print("="*80)
    print(f"Original Proof LogP: {pos_logp:.4f}")
    print(f"Near-Miss LogPs: {[r['logp'] for r in near_miss_results]}")
    print(f"Cosine Similarities: {[r['cos_sim'] for r in near_miss_results]}")
    print(f"Relative Differences: {[r['rel_diff'] for r in near_miss_results]}")
    print("\nInterpretation:")
    print("  - Higher cosine similarity = near-miss gradient closer to valid gradient")
    print("  - Lower relative difference = near-miss gradient more similar to valid gradient")
    print("="*80)

if __name__ == "__main__":
    main()
