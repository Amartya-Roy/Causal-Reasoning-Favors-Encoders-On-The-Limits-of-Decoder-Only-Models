"""
Compare proof quality between two models depth-wise.
Uses existing proof chain outputs and ground truth from dataset.
INCLUDES: Accuracy comparison + Text similarity metrics (BLEU, METEOR, ROUGE)
"""

import json
import os
from collections import defaultdict
import re

# Reuse robust metric and cleaning implementations (lazy-load heavy deps there)
from evaluate_proof_quality import clean_proof_chain, compute_bleu, compute_meteor, compute_rouge
from extract_label_from_proof import extract_label_from_symbolic_proof


def clean_proof_chain(proof_text):
    """Clean proof chain for evaluation"""
    if not proof_text:
        return ""
    
    cleaned = proof_text.strip()
    
    # Remove <think> tags
    if '<think>' in cleaned:
        think_end = cleaned.find('</think>')
        if think_end != -1:
            cleaned = cleaned[think_end + 8:].strip()
    
    # Extract after markers
    markers = ['proof chain:', 'corrected proof chain:', 'output:']
    for marker in markers:
        if marker in cleaned.lower():
            idx = cleaned.lower().find(marker)
            cleaned = cleaned[idx + len(marker):].strip()
            break
    
    # Remove common prefixes
    prefixes = ['the proof chain is:', 'here is the proof:', 'answer:']
    for prefix in prefixes:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    
    return cleaned


def tokenize(text):
    """Simple word tokenization"""
    return text.lower().split()


def count_reasoning_steps(proof_text):
    """Count reasoning steps in a proof chain"""
    if not proof_text:
        return 0
    cleaned = clean_proof_chain(proof_text)
    if not cleaned:
        return 0
    
    # Count lines with step-like patterns
    lines = cleaned.split('\n')
    step_count = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Step patterns: numbered items, arrows, logical connectives
        if re.match(r'^\d+[.:\)]\s*', line):
            step_count += 1
        elif re.match(r'^[-*•]\s+', line):
            step_count += 1
        elif '→' in line or '->' in line or 'therefore' in line.lower():
            step_count += 1
    
    return max(step_count, 1) if cleaned else 0


def compute_metrics(generated_proof, reference_proof):
    """Compute BLEU, METEOR, ROUGE scores using robust implementations.

    This wraps the shared implementations in `evaluate_proof_quality` which
    handle edge cases (empty proofs, missing deps) safely.
    """
    gen_clean = clean_proof_chain(generated_proof)
    ref_clean = clean_proof_chain(reference_proof)

    # Use the shared metric functions which already handle empty cases
    bleu_scores = compute_bleu(ref_clean, gen_clean)
    meteor_val = compute_meteor(ref_clean, gen_clean)
    rouge_scores = compute_rouge(ref_clean, gen_clean)

    metrics = {**bleu_scores, 'meteor': meteor_val, **rouge_scores}
    return metrics


def load_accuracy_metrics(results_dir):
    """Load accuracy metrics from evaluation_metrics.json"""
    metrics_path = os.path.join(results_dir, "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None


def main():
    # Paths
    repair_dir = "/home/omy1cob/Causal_LLM/causality_grammar-DB41/results/inference_repair_depth0-50"
    standard_dir = "/home/omy1cob/Causal_LLM/causality_grammar-DB41/results/inference_hf_bashcache_depth0-50"
    
    repair_proofs_path = os.path.join(repair_dir, "repaired-proof-chains.json")
    standard_proofs_path = os.path.join(standard_dir, "model-proof-chains.json")
    dataset_path = "/home/omy1cob/Causal_LLM/causality_grammar-DB41/data/alice_test_depth0-50_complete.parquet"
    output_path = "/home/omy1cob/Causal_LLM/causality_grammar-DB41/results/proof_quality_comparison_new.json"
    
    print("="*100)
    print("PROOF QUALITY COMPARISON: Repair vs Standard Model")
    print("="*100)
    
    # Load accuracy metrics
    print("\nLoading accuracy metrics...")
    repair_accuracy = load_accuracy_metrics(repair_dir)
    standard_accuracy = load_accuracy_metrics(standard_dir)
    
    if repair_accuracy and standard_accuracy:
        print(f"✓ Repair model overall accuracy: {repair_accuracy['overall_accuracy']:.4f}")
        print(f"✓ Standard model overall accuracy: {standard_accuracy['overall_accuracy']:.4f}")
    else:
        print("⚠ Could not load accuracy metrics from one or both models")
    
    # Load proofs
    print("\nLoading proof chains...")
    with open(repair_proofs_path, 'r') as f:
        repair_proofs = json.load(f)
    with open(standard_proofs_path, 'r') as f:
        standard_proofs = json.load(f)
    
    # Load ground truth (lazy import to avoid hard dependency at module import time)
    print("Loading ground truth from dataset...")
    try:
        from datasets import load_dataset
    except Exception as e:
        raise ImportError("Please install the 'datasets' package (and its dependencies like pyarrow/numpy) to load the parquet dataset.") from e

    dataset = load_dataset("parquet", data_files=dataset_path)["train"]
    ground_truth = [ex["Complex_CoT"] if "Complex_CoT" in ex and ex["Complex_CoT"] else "" for ex in dataset]
    depths = [int(ex["Depth"]) for ex in dataset]
    
    print(f"Total examples: {len(ground_truth)}")
    print(f"Repair proofs: {len(repair_proofs)}")
    print(f"Standard proofs: {len(standard_proofs)}")
    
    # Verify all three sources have same depth ordering
    repair_labels_depths_path = os.path.join(repair_dir, "labels-and-depths.json")
    with open(repair_labels_depths_path, 'r') as f:
        repair_depths = json.load(f)[1]
    assert repair_depths == depths, "Depth ordering mismatch between repair results and ground truth!"
    
    # Ensure alignment
    assert len(repair_proofs) == len(standard_proofs) == len(ground_truth), "Proof chain counts mismatch!"
    
    # Compute metrics depth-wise
    print("\nComputing metrics depth-wise...")
    repair_by_depth = defaultdict(list)
    standard_by_depth = defaultdict(list)
    
    for i, (repair_proof, standard_proof, gt_proof, depth) in enumerate(zip(repair_proofs, standard_proofs, ground_truth, depths)):
        # Repair model metrics
        repair_metrics = compute_metrics(repair_proof, gt_proof)
        repair_metrics['step_count'] = count_reasoning_steps(repair_proof)
        repair_by_depth[depth].append(repair_metrics)
        
        # Standard model metrics
        standard_metrics = compute_metrics(standard_proof, gt_proof)
        standard_metrics['step_count'] = count_reasoning_steps(standard_proof)
        standard_by_depth[depth].append(standard_metrics)
    
    # Aggregate by depth
    print("\nAggregating results...")
    comparison_results = {}
    sorted_depths = sorted(repair_by_depth.keys())
    
    for depth in sorted_depths:
        repair_depth_metrics = repair_by_depth[depth]
        standard_depth_metrics = standard_by_depth[depth]
        
        # Average each metric
        repair_avg = {}
        standard_avg = {}
        
        for metric in ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'meteor', 'rouge1', 'rouge2', 'rougeL', 'step_count']:
            repair_avg[metric] = sum(m[metric] for m in repair_depth_metrics) / len(repair_depth_metrics)
            standard_avg[metric] = sum(m[metric] for m in standard_depth_metrics) / len(standard_depth_metrics)
        
        # Add accuracy from evaluation_metrics.json
        depth_str = str(depth)
        if repair_accuracy and 'accuracy_by_depth' in repair_accuracy:
            repair_avg['accuracy'] = repair_accuracy['accuracy_by_depth'].get(depth_str, 0.0)
        if standard_accuracy and 'accuracy_by_depth' in standard_accuracy:
            standard_avg['accuracy'] = standard_accuracy['accuracy_by_depth'].get(depth_str, 0.0)
        
        comparison_results[depth] = {
            'repair': repair_avg,
            'standard': standard_avg,
            'num_examples': len(repair_depth_metrics)
        }
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Print ACCURACY comparison first (most important!)
    print("\n" + "="*100)
    print("ACCURACY COMPARISON BY DEPTH (Label Correctness)")
    print("="*100)
    print(f"{'Depth':<8} {'Repair Acc':<12} {'Standard Acc':<14} {'Δ (R-S)':<12} {'Winner':<12}")
    print("-" * 100)
    
    repair_wins = 0
    standard_wins = 0
    ties = 0
    
    for depth in sorted_depths:
        repair_acc = comparison_results[depth]['repair'].get('accuracy', 0.0)
        standard_acc = comparison_results[depth]['standard'].get('accuracy', 0.0)
        diff = repair_acc - standard_acc
        
        if diff > 0.01:
            winner = "✓ REPAIR"
            repair_wins += 1
        elif diff < -0.01:
            winner = "✗ Standard"
            standard_wins += 1
        else:
            winner = "~ Tie"
            ties += 1
        
        print(f"{depth:<8} {repair_acc:<12.4f} {standard_acc:<14.4f} {diff:+12.4f} {winner:<12}")
    
    print("-" * 100)
    print(f"Summary: Repair wins {repair_wins} depths, Standard wins {standard_wins} depths, Ties: {ties}")
    
    # Overall accuracy comparison
    repair_overall = repair_accuracy['overall_accuracy'] if repair_accuracy else 0.0
    standard_overall = standard_accuracy['overall_accuracy'] if standard_accuracy else 0.0
    print(f"\nOVERALL ACCURACY: Repair = {repair_overall:.4f}, Standard = {standard_overall:.4f}, Δ = {repair_overall - standard_overall:+.4f}")
    
    # Print text similarity metrics
    print("\n" + "="*100)
    print("TEXT SIMILARITY METRICS (BLEU-4, METEOR, ROUGE-L)")
    print("="*100)
    print(f"{'Depth':<8} {'Model':<12} {'BLEU-4':<10} {'METEOR':<10} {'ROUGE-L':<10} {'Steps':<8}")
    print("-" * 100)
    
    for depth in sorted_depths:
        repair_bleu4 = comparison_results[depth]['repair']['bleu4']
        repair_meteor = comparison_results[depth]['repair']['meteor']
        repair_rougeL = comparison_results[depth]['repair']['rougeL']
        repair_steps = comparison_results[depth]['repair']['step_count']
        
        standard_bleu4 = comparison_results[depth]['standard']['bleu4']
        standard_meteor = comparison_results[depth]['standard']['meteor']
        standard_rougeL = comparison_results[depth]['standard']['rougeL']
        standard_steps = comparison_results[depth]['standard']['step_count']
        
        print(f"{depth:<8} {'Repair':<12} {repair_bleu4:<10.4f} {repair_meteor:<10.4f} {repair_rougeL:<10.4f} {repair_steps:<8.1f}")
        print(f"{'':<8} {'Standard':<12} {standard_bleu4:<10.4f} {standard_meteor:<10.4f} {standard_rougeL:<10.4f} {standard_steps:<8.1f}")
        
        # Show difference
        diff_bleu4 = repair_bleu4 - standard_bleu4
        diff_meteor = repair_meteor - standard_meteor
        diff_rougeL = repair_rougeL - standard_rougeL
        diff_steps = repair_steps - standard_steps
        print(f"{'':<8} {'Δ (R-S)':<12} {diff_bleu4:+10.4f} {diff_meteor:+10.4f} {diff_rougeL:+10.4f} {diff_steps:+8.1f}")
        print("-" * 100)
    
    # Overall averages
    print("\n" + "="*100)
    print("OVERALL AVERAGES")
    print("="*100)
    
    all_repair_bleu4 = [comparison_results[d]['repair']['bleu4'] for d in sorted_depths]
    all_repair_meteor = [comparison_results[d]['repair']['meteor'] for d in sorted_depths]
    all_repair_rougeL = [comparison_results[d]['repair']['rougeL'] for d in sorted_depths]
    all_repair_acc = [comparison_results[d]['repair'].get('accuracy', 0) for d in sorted_depths]
    all_repair_steps = [comparison_results[d]['repair']['step_count'] for d in sorted_depths]
    
    all_standard_bleu4 = [comparison_results[d]['standard']['bleu4'] for d in sorted_depths]
    all_standard_meteor = [comparison_results[d]['standard']['meteor'] for d in sorted_depths]
    all_standard_rougeL = [comparison_results[d]['standard']['rougeL'] for d in sorted_depths]
    all_standard_acc = [comparison_results[d]['standard'].get('accuracy', 0) for d in sorted_depths]
    all_standard_steps = [comparison_results[d]['standard']['step_count'] for d in sorted_depths]
    
    print(f"{'Model':<12} {'Accuracy':<12} {'BLEU-4':<10} {'METEOR':<10} {'ROUGE-L':<10} {'Avg Steps':<10}")
    print("-" * 64)
    
    repair_avg_acc = sum(all_repair_acc)/len(all_repair_acc)
    repair_avg_bleu4 = sum(all_repair_bleu4)/len(all_repair_bleu4)
    repair_avg_meteor = sum(all_repair_meteor)/len(all_repair_meteor)
    repair_avg_rougeL = sum(all_repair_rougeL)/len(all_repair_rougeL)
    repair_avg_steps = sum(all_repair_steps)/len(all_repair_steps)
    
    standard_avg_acc = sum(all_standard_acc)/len(all_standard_acc)
    standard_avg_bleu4 = sum(all_standard_bleu4)/len(all_standard_bleu4)
    standard_avg_meteor = sum(all_standard_meteor)/len(all_standard_meteor)
    standard_avg_rougeL = sum(all_standard_rougeL)/len(all_standard_rougeL)
    standard_avg_steps = sum(all_standard_steps)/len(all_standard_steps)
    
    print(f"{'Repair':<12} {repair_avg_acc:<12.4f} {repair_avg_bleu4:<10.4f} {repair_avg_meteor:<10.4f} {repair_avg_rougeL:<10.4f} {repair_avg_steps:<10.1f}")
    print(f"{'Standard':<12} {standard_avg_acc:<12.4f} {standard_avg_bleu4:<10.4f} {standard_avg_meteor:<10.4f} {standard_avg_rougeL:<10.4f} {standard_avg_steps:<10.1f}")
    print(f"{'Δ (R-S)':<12} {repair_avg_acc - standard_avg_acc:+12.4f} {repair_avg_bleu4 - standard_avg_bleu4:+10.4f} {repair_avg_meteor - standard_avg_meteor:+10.4f} {repair_avg_rougeL - standard_avg_rougeL:+10.4f} {repair_avg_steps - standard_avg_steps:+10.1f}")
    
    # Analyze by depth ranges
    print("\n" + "="*100)
    print("ACCURACY BY DEPTH RANGES")
    print("="*100)
    
    depth_ranges = [(0, 5), (6, 10), (11, 20), (21, 30), (31, 40), (41, 50)]
    
    for start, end in depth_ranges:
        range_depths = [d for d in sorted_depths if start <= d <= end]
        if not range_depths:
            continue
        
        repair_range_acc = [comparison_results[d]['repair'].get('accuracy', 0) for d in range_depths]
        standard_range_acc = [comparison_results[d]['standard'].get('accuracy', 0) for d in range_depths]
        
        repair_avg = sum(repair_range_acc) / len(repair_range_acc)
        standard_avg = sum(standard_range_acc) / len(standard_range_acc)
        diff = repair_avg - standard_avg
        
        winner = "Repair ✓" if diff > 0.01 else ("Standard ✗" if diff < -0.01 else "Tie")
        print(f"Depth {start:2d}-{end:2d}: Repair = {repair_avg:.4f}, Standard = {standard_avg:.4f}, Δ = {diff:+.4f}  [{winner}]")
    
    print("\n" + "="*100)
    print(f"Results saved to: {output_path}")
    print("="*100)


if __name__ == "__main__":
    main()
