# Assumption 5 Verification: Near-Miss Gradient Proximity

## Overview

This script empirically tests **Assumption 5**  whether **near-miss negatives** (slightly corrupted valid proofs) produce LoRA gradients that are more similar to the positive gradient than random or hard negatives.

### What is Assumption 5?

**Hypothesis**: When a model learns from a valid proof versus a near-miss error (single token/step corruption), the resulting learning signals (gradients) should be closer to each other than to completely different proofs.

**Why it matters**: If true, this suggests models can exploit fine-grained semantic structure in proofs through gradient-based learning, which has implications for understanding encoder vs. decoder architectures in reasoning tasks.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Output Files](#output-files)
- [Interpreting Results](#interpreting-results)
- [Troubleshooting](#troubleshooting)
- [Technical Details](#technical-details)

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, ~16GB+ VRAM)
- ~10GB disk space for model weights

### Dependencies

```bash
pip install torch transformers peft accelerate
pip install pandas numpy scipy matplotlib seaborn tqdm
pip install python-Levenshtein bitsandbytes
```

Or use the provided environment:

```bash
conda create -n causal_verify python=3.9
conda activate causal_verify
pip install -r requirements.txt
```

---

## Quick Start

### Basic Usage

```bash
python verify_assumption5.py \
    --dataset_path data/assumption5_data.csv \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --output_dir results/assumption5 \
    --n_samples 100 \
    --k_near_miss 3
```

### Dataset Format

Your CSV must have these columns:

| Column       | Type    | Description                           |
|--------------|---------|---------------------------------------|
| `question`   | string  | The reasoning question/context        |
| `proof_chain`| string  | The proof/reasoning chain             |
| `label`      | int     | 1 = valid proof, 0 = invalid proof    |

Example:

```csv
question,proof_chain,label
"Does A cause C?","A causes B\nB causes C\nTherefore A causes C",1
"Does A cause C?","A causes B\nC causes D\nTherefore A causes C",0
```

---

## How It Works

### 1. Near-Miss Generation

The script creates **controlled corruptions** of valid proofs using three strategies:

#### a) Delete One Step
Removes exactly one line from multi-step proofs.

```
Original:  A causes B
           B causes C
           Therefore A causes C

Near-miss: A causes B
           Therefore A causes C  ← Deleted middle step
```

#### b) Swap Entity Tokens
Swaps two different capitalized entities within the proof.

```
Original:  Alice helps Bob
           Bob helps Carol
           Therefore Alice indirectly helps Carol

Near-miss: Carol helps Bob  ← Swapped Alice ↔ Carol
           Bob helps Carol
           Therefore Alice indirectly helps Carol
```

#### c) Flip Relation Token
Inverts a logical/causal relation operator.

```
Original:  X causes Y
           Y causes Z
           Therefore X causes Z

Near-miss: X prevents Y  ← Flipped causes → prevents
           Y causes Z
           Therefore X causes Z
```

### 2. Control Negative Generation

For comparison, the script also generates:

- **Random Negative**: Any invalid proof from a different question
- **Hard Negative**: Invalid proof with high lexical overlap but large edit distance (shares many words but wrong semantically)

### 3. Gradient Computation

For each proof (positive, near-miss, random, hard):

1. **Load Model + LoRA**: Uses LoRA (Low-Rank Adaptation) on top of base model
2. **Forward Pass**: Compute log probability `log P(proof | question)`
3. **Backward Pass**: Compute gradients `∇_θ log P(proof | question)` w.r.t. LoRA parameters
4. **Extract**: Flatten all LoRA gradients into a single vector

**Why LoRA?** 
- Only LoRA parameters are trainable (base model frozen)
- Gradients represent the "learning direction" for this specific example
- More stable than full model gradients

### 4. Gradient Comparison Metrics

#### Cosine Similarity
Measures angular similarity between gradient vectors:

```
cos(g⁺, g⁻) = (g⁺ · g⁻) / (||g⁺|| × ||g⁻||)
```

- Range: [-1, 1]
- **Higher** = more similar gradient direction
- **Expected**: cos(g⁺, g_near_miss) > cos(g⁺, g_random)

#### Relative Difference (r)
Measures normalized Euclidean distance:

```
r = ||g⁺ - g⁻|| / (||g⁺|| + ||g⁻||)
```

- Range: [0, 1]
- **Lower** = more similar gradient magnitude
- **Expected**: r(g⁺, g_near_miss) < r(g⁺, g_random)

#### Margin Delta
Log-probability difference:

```
Δ = log P(y⁺ | x) - log P(y⁻ | x)
```

- Shows model's confidence difference between positive and negative

### 5. Statistical Testing

Uses **Wilcoxon signed-rank test** (non-parametric) to test:

```
H₀: Near-miss gradients are NOT closer to positive than controls
H₁: Near-miss gradients ARE closer to positive than controls
```

Computes:
- **p-value**: Statistical significance (reject H₀ if p < 0.05)
- **Effect size**: Magnitude of the difference
- **Confidence intervals**: Via bootstrap (optional)

---

## Output Files

All outputs saved to `--output_dir` (default: `results/assumption5/`):

### 1. `assumption5_results.csv`
Raw data for all gradient pairs.

| Column          | Description                                    |
|-----------------|------------------------------------------------|
| `pair_type`     | near_miss / random / hard                      |
| `question`      | The question (truncated to 50 chars)           |
| `edit_distance` | Levenshtein distance from positive proof       |
| `cosine_sim`    | Cosine similarity between gradients            |
| `rel_diff_r`    | Relative difference metric                     |
| `margin_delta`  | Log-prob difference (positive - negative)      |
| `logp_positive` | Log P(positive proof \| question)              |
| `logp_negative` | Log P(negative proof \| question)              |
| `len_positive`  | Character length of positive proof             |
| `len_negative`  | Character length of negative proof             |

### 2. Visualizations

#### `F1_edit_distance_distribution.png`
Histogram and CDF of edit distances for near-miss negatives.

**Interpretation**: Shows how "near" the near-misses are (typically 1-20 character edits).

#### `F2_cosine_similarity_boxplot.png`
Boxplot comparing cosine similarity across pair types.

**Expected pattern**: Near-miss box higher than random/hard boxes.

#### `F3_relative_difference_boxplot.png`
Boxplot comparing relative difference (r) across pair types.

**Expected pattern**: Near-miss box lower than random/hard boxes.

#### `F4_margin_delta_distribution.png`
KDE plot of log-probability margins by pair type.

**Interpretation**: Shows how confidently the model distinguishes positive from each negative type.

### 3. `summary.txt`
Statistical summary including:
- Total pairs analyzed per type
- Mean ± std for each metric
- Wilcoxon test results
- Final verdict on Assumption 5

---

## Interpreting Results

### ✅ Assumption 5 Holds

```
Near-miss vs Random (Cosine Similarity):
  p-value = 0.000123  *SIGNIFICANT*
  Near-miss mean: 0.8234 > Random mean: 0.4521

Near-miss vs Hard (Cosine Similarity):
  p-value = 0.001456  *SIGNIFICANT*
  Near-miss mean: 0.8234 > Hard mean: 0.5632
```

**Conclusion**: Near-miss gradients are significantly closer to positive gradients, supporting the assumption.

**Implications**:
- Model can distinguish fine-grained semantic errors
- Gradient structure reflects compositional reasoning
- Supports theoretical claims about encoder advantages

### ❌ Assumption 5 Does Not Hold

```
Near-miss vs Random (Cosine Similarity):
  p-value = 0.342  Not significant
  Near-miss mean: 0.5123 ≈ Random mean: 0.5098
```

**Conclusion**: No significant difference between near-miss and control gradients.

**Possible reasons**:
- Model doesn't learn meaningful proof structure
- Near-misses too difficult to distinguish from random errors
- Dataset lacks sufficient structure
- LoRA rank too small (try increasing `r` in config)

### ◐ Mixed Results

Some metrics support the assumption, others don't. Check:
1. **Effect sizes**: Large effect size + borderline p-value → likely true
2. **Sample size**: Small n → low statistical power, try more samples
3. **Edit types**: Check which edit types work (delete vs swap vs flip)

---

## Troubleshooting

### "Could not generate near-misses for sample X"

**Cause**: Proof text doesn't support any edit operation.

**Examples**:
- Single-line proof → can't delete step
- No capitalized tokens → can't swap entities
- No relation keywords → can't flip relations

**Solution**: This is normal and expected. The script skips these samples. If too many samples fail:
- Check your data format
- Ensure proofs have multiple steps
- Add more relation keywords to `flip_relation_token()`

### CUDA Out of Memory (OOM)

**Solutions**:
1. Use smaller model: `--model_name Qwen/Qwen2.5-0.5B`
2. Reduce samples: `--n_samples 50`
3. Clear cache: `torch.cuda.empty_cache()` (already in code)
4. Use CPU: Remove CUDA environment or set `CUDA_VISIBLE_DEVICES=""`

### "cuBLAS API failed" or dtype errors

**Already fixed in current version** (removed quantization). If you still see this:
- Update PyTorch: `pip install --upgrade torch`
- Update transformers: `pip install --upgrade transformers`
- Check CUDA compatibility: `torch.cuda.is_available()`

### No results / Empty CSV

**Check**:
1. Dataset has `label=1` samples
2. Positive samples have sufficient structure
3. Model loaded successfully (check logs)
4. No errors in gradient computation (check terminal output)

### Statistical tests show NaN

**Cause**: Insufficient samples in one or more categories.

**Solution**: Need at least 6-10 samples per category for reliable statistics. Increase `--n_samples`.

---

## Technical Details

### Model Architecture

- **Base Model**: Qwen-2.5 (1.5B or 0.5B parameters)
- **Adaptation**: LoRA rank-16 on attention projections
- **Precision**: FP16 (no quantization for stable gradients)
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### LoRA Configuration

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # Rank of LoRA matrices
    lora_alpha=32,           # Scaling factor
    lora_dropout=0.05,       # Dropout rate
    target_modules=[...],    # Which layers to adapt
    bias="none",             # Don't adapt bias terms
    inference_mode=False,    # Enable training
)
```

**Tuning tips**:
- Increase `r` (16 → 32 → 64) for richer gradients
- Decrease `lora_dropout` (0.05 → 0.01) for more stable gradients
- Add more `target_modules` to capture more gradient information

### Gradient Extraction

Only LoRA parameters' gradients are extracted:

```python
for name, param in model.named_parameters():
    if 'lora' in name.lower() and param.requires_grad:
        grad_list.append(param.grad.view(-1).float())
```

**Why?**
- Base model is frozen (no gradients)
- LoRA gradients are cleaner and more interpretable
- Reduces memory footprint
- Avoids quantization issues

### Memory Requirements

| Model Size | GPU RAM (FP16) | GPU RAM (8-bit) |
|------------|----------------|-----------------|
| 0.5B       | ~4GB           | ~2GB            |
| 1.5B       | ~10GB          | ~5GB            |
| 7B         | ~28GB          | ~14GB           |

Add ~2GB overhead for LoRA + gradients + activations.

### Performance

On NVIDIA RTX 3090 (24GB):
- **Qwen-1.5B**: ~10-15 seconds per sample (3 near-miss + 2 controls)
- **100 samples**: ~25-40 minutes total
- **CPU mode**: 10-20x slower

---

## Advanced Usage

### Custom Near-Miss Functions

Add your own edit strategy:

```python
def flip_negation(proof: str) -> Optional[str]:
    """Flip negation words."""
    if 'not' in proof.lower():
        return proof.replace('not', '', 1)
    else:
        # Add 'not' somewhere
        words = proof.split()
        idx = random.randint(0, len(words)-1)
        words.insert(idx, 'not')
        return ' '.join(words)

# Add to edit_functions dict in make_near_miss()
edit_functions['flip_negation'] = flip_negation
```

### Using Pre-trained LoRA Adapters

If you've already fine-tuned LoRA weights:

```bash
python verify_assumption5.py \
    --dataset_path data/assumption5_data.csv \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --lora_checkpoint path/to/lora/adapter \
    --output_dir results/assumption5_finetuned
```

### Batch Processing Multiple Datasets

```python
datasets = [
    "data/math_proofs.csv",
    "data/logic_proofs.csv",
    "data/causal_proofs.csv"
]

for ds in datasets:
    name = ds.split('/')[-1].replace('.csv', '')
    run_assumption5_verification(
        dataset_path=ds,
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        output_dir=f"results/{name}",
        n_samples=100,
        k_near_miss=3,
        seed=42
    )
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{causality_encoders_2024,
  title={Causal Reasoning Favors Encoders: On The Limits of Decoder-Only Models},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2024}
}
```

---

## License

[Specify your license here - MIT, Apache 2.0, etc.]

---

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: [your-email@domain.com]
- Discussion forum: [link if applicable]

---

## Changelog

### v1.0.0 (Current)
- ✅ Removed quantization for stable gradient computation
- ✅ Added comprehensive error handling
- ✅ Fixed cuBLAS dtype mismatch issues
- ✅ Added detailed statistical reporting
- ✅ Implemented Wilcoxon tests with effect sizes
- ✅ Created visualization pipeline

### Future Improvements
- [ ] Add support for other model families (Llama, Mistral, GPT)
- [ ] Implement gradient attribution analysis
- [ ] Add interactive result explorer
- [ ] Support for multi-GPU processing
- [ ] Integration with Weights & Biases logging

---

## Acknowledgments

- Hugging Face Transformers team for the model implementations
- PEFT library for LoRA support
- Original paper authors for the theoretical framework
