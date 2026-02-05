"""
Push the fine-tuned model to Hugging Face Hub
"""

from huggingface_hub import login, upload_folder
import os

# Login with your Hugging Face credentials
print("Logging in to Hugging Face...")
login()

# Model directory
model_path = "/home/omy1cob/Causal_LLM/causality_grammar-DB41/models/Qwen3-1.7B-repair-fullfinetuned-temp/checkpoint-30000"

# Check if model exists
if not os.path.exists(model_path):
    print(f"ERROR: Model path does not exist: {model_path}")
    exit(1)

print(f"\nUploading model from: {model_path}")
print(f"To repository: Amartya77/LogicBench-Qwen-FT-Response")

# Push your model files
upload_folder(
    folder_path=model_path,
    repo_id="Amartya77/LogicBench-Qwen-FT-Response",
    repo_type="model"
)

print("\n✓ Model successfully uploaded to Hugging Face Hub!")
print("View at: https://huggingface.co/Amartya77/LogicBench-Qwen-FT-Response")
