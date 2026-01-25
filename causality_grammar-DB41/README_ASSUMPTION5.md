# Assumption 5 Verification: Near-Miss Gradient Proximity

> **Last Updated**: January 26, 2026

This tool empirically verifies **Assumption 5** from the paper, which states that near-miss negatives (single-edit corruptions of valid proofs) have LoRA gradients more similar to the positive gradient than random or hard negatives.

## Quick Start

```bash
# 1. Install dependencies
pip install torch transformers peft bitsandbytes python-Levenshtein pandas numpy matplotlib seaborn scipy

# 2. Prepare data (if not already done)
python prepare_assumption5_data.py --input data/train_data.parquet --output data/assumption5_data.csv

# 3. Run verification (GPU required)
python verify_assumption5.py \
    --dataset_path data/assumption5_data.csv \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --output_dir results/assumption5 \
    --n_samples 100 \
    --k_near_miss 3
```

## Requirements

| Package | Version |
|---------|---------|
| Python | ≥ 3.8 |
| PyTorch | ≥ 2.0 |
| transformers | ≥ 4.40 |
| peft | ≥ 0.10 |
| bitsandbytes | ≥ 0.43 |

> ⚠️ **GPU Required**: 4-bit quantization needs CUDA. CPU mode is supported but extremely slow.

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_path` | Required | CSV with columns: `question`, `proof_chain`, `label` |
| `--model_name` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace model name |
| `--lora_checkpoint` | None | Optional path to pre-trained LoRA adapter |
| `--output_dir` | `results/assumption5` | Directory for outputs |
| `--n_samples` | 100 | Number of positive samples to analyze |
| `--k_near_miss` | 3 | Near-miss negatives per positive |
| `--seed` | 42 | Random seed |

## What It Tests

**Assumption 5 (empirical form):** For question x, let y⁺ be a valid proof and y⁻ be a near-miss (single edit to y⁺). Define g(χ) = ∇_θ_LoRA log p(y|x). The assumption holds if:

1. **High cosine similarity**: cos(g⁺, g⁻) is higher for near-misses than controls
2. **Low relative difference**: r = |g⁺ - g⁻| / (|g⁺| + |g⁻|) is lower for near-misses than controls

## Near-Miss Edit Types

| Edit Type | Example |
|-----------|---------|
| `delete` | Remove one reasoning step |
| `swap_entity` | Replace entity A with entity B |
| `flip_relation` | Change "causes" → "prevents" |

## Outputs

```
results/assumption5/
├── assumption5_results.csv     # All pair-wise metrics
├── F1_edit_distance_distribution.png
├── F2_cosine_similarity_boxplot.png
├── F3_relative_difference_boxplot.png
├── F4_margin_delta_distribution.png
└── summary.txt                 # Summary statistics
```

## Interpreting Results

The script prints a summary like:

```
ASSUMPTION 5 VERIFICATION SUMMARY
======================================================================
✓ Cosine Similarity: Near-miss > Random? 0.85 > 0.42 = True
✓ Cosine Similarity: Near-miss > Hard? 0.85 > 0.51 = True
✓ Relative Diff: Near-miss < Random? 0.12 < 0.58 = True
✓ Relative Diff: Near-miss < Hard? 0.12 < 0.48 = True

★ ASSUMPTION 5 APPEARS TO HOLD
```

## Example: Running on Google Colab

```python
# Install dependencies
!pip install torch transformers peft bitsandbytes python-Levenshtein

# Upload your data or use sample
# Then run:
!python verify_assumption5.py \
    --dataset_path data/assumption5_data.csv \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --n_samples 50
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| `No GPU found` | Use a GPU environment (Colab, cloud GPU) |
| `OutOfMemoryError` | Reduce `--n_samples` or use smaller model |
| `ModuleNotFoundError: peft` | Run `pip install peft bitsandbytes` |

## Citation

If you use this verification tool, please cite the paper.
