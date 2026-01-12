# Distill Symbolic Reasoning into Qwen

Train Qwen LLM to perform logical reasoning by distilling knowledge from a symbolic forward-chaining solver.

## Overview

This script:
1. **Symbolic Solver** generates perfect step-by-step reasoning traces
2. **Qwen** learns to mimic these traces via finetuning
3. **Evaluates** before/after training to measure improvement

## Requirements

- Python 3.8+
- CUDA GPU (recommended) or CPU
- ~4GB GPU memory for Qwen2.5-0.5B

## Setup (Windows)

### 1. Clone/Copy Files

```
qcwt/
├── distill_reasoning.py
├── requirements.txt
├── README.md
└── data/
    └── train_data.parquet
```

### 2. Create Virtual Environment

```cmd
cd qcwt
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

**For GPU (CUDA 11.8):**
```cmd
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.35.0 pandas pyarrow tqdm
```

**For GPU (CUDA 12.1):**
```cmd
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.35.0 pandas pyarrow tqdm
```

**For CPU only:**
```cmd
pip install -r requirements.txt
```

### 4. Prepare Data

Copy your `train_data.parquet` file to the `data/` folder:
```cmd
mkdir data
copy path\to\train_data.parquet data\
```

## Usage

### Basic Run
```cmd
python distill_reasoning.py --parquet_path data/train_data.parquet
```

### Recommended Settings (GPU)
```cmd
python distill_reasoning.py ^
    --parquet_path data/train_data.parquet ^
    --train_samples 2000 ^
    --eval_samples 200 ^
    --epochs 5 ^
    --batch_size 8 ^
    --lr 2e-5
```

### Quick Test (CPU)
```cmd
python distill_reasoning.py ^
    --parquet_path data/train_data.parquet ^
    --train_samples 100 ^
    --eval_samples 50 ^
    --epochs 1 ^
    --batch_size 1
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--parquet_path` | required | Path to train_data.parquet |
| `--model` | Qwen/Qwen2.5-0.5B-Instruct | HuggingFace model name |
| `--train_samples` | 500 | Number of training examples |
| `--eval_samples` | 100 | Number of evaluation examples |
| `--epochs` | 2 | Training epochs |
| `--batch_size` | 2 | Batch size (increase for GPU) |
| `--lr` | 2e-5 | Learning rate |
| `--seed` | 42 | Random seed |

## Expected Output

```
================================================================================
DISTILLING SYMBOLIC REASONING INTO QWEN
================================================================================

[1] Loading and generating CoT data...
Generated 500 train, 100 eval examples

[2] Loading Qwen/Qwen2.5-0.5B-Instruct...

[3] Pre-training evaluation...
Pre-training accuracy: 30.0%

[4] Training for 2 epochs...
Epoch 1: Loss = 1.2345
Epoch 1: Accuracy = 45.0%
Epoch 2: Loss = 0.8765
Epoch 2: Accuracy = 60.0%

[5] Final evaluation...
Final accuracy: 65.0%

================================================================================
IMPROVEMENT: 30.0% → 65.0% (+35.0%)
================================================================================
Model saved to ./checkpoints/qwen_cot_finetuned
```

## How It Works

### 1. Symbolic Reasoning (Teacher)
```
Input:  Facts: brave, smart
        Rules: brave ∧ smart → confident
        Query: Is 'confident' true?

Output: Known facts: brave, smart.
        Since brave, smart are all true, and (brave ∧ smart) → confident,
        therefore confident is true.
        Answer: Yes
```

### 2. LLM Training (Student)
- Model sees the input prompt
- Model learns to generate the reasoning trace
- Loss computed only on the reasoning part (not the prompt)

### 3. Inference
After training, the model can generate reasoning for new problems.

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch_size` to 1 or 2
- Use `--model Qwen/Qwen2.5-0.5B-Instruct` (smallest)

### Slow Training
- Increase `--batch_size` if GPU memory allows
- Use GPU instead of CPU

### Model Download Issues
```cmd
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set TRANSFORMERS_OFFLINE=0
```

## License

MIT
