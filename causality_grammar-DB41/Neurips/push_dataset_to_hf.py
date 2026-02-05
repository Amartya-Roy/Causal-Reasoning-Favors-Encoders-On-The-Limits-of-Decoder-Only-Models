"""
Push the training dataset to Hugging Face Hub
"""

from huggingface_hub import login, upload_folder
import os
from pathlib import Path

# Use HF token if provided to avoid interactive login
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("No HF_TOKEN found in environment — calling interactive login() (may block)")
    login()

# Repositories and folders to upload
repo_id = "Amartya77/LogicBench_Test"
folders_to_upload = [
    "/home/amartya/Causal_LLM/causality_grammar-DB41/results/inference_hf_bashcache_depth0-50",
    "/home/amartya/Causal_LLM/causality_grammar-DB41/results/inference_repair_depth0-50",
]

allowed_exts = ('.parquet', '.csv', '.jsonl', '.json', '.txt')

print(f"Preparing to upload folders to dataset: {repo_id}")

for folder in folders_to_upload:
    path = Path(folder)
    if not path.exists():
        print(f"WARNING: Folder does not exist, skipping: {folder}")
        continue

    # Print summary of files to upload (filtering by allowed extensions)
    print(f"\nFolder: {folder}")
    files = [p for p in path.rglob('*') if p.is_file() and p.suffix in allowed_exts]
    if not files:
        print("  No dataset files found to upload (looking for .parquet/.csv/.jsonl/.json/.txt)")
        continue

    print("Files to upload:")
    total_size = 0
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += f.stat().st_size
        print(f"  - {f.relative_to(path)} ({size_mb:.2f} MB)")

    print(f"Uploading {len(files)} files (≈{total_size/(1024*1024):.2f} MB) to repo {repo_id} under path {path.name}")

    # Upload folder into a subpath in the HF dataset repo to keep folders separate
    try:
        # Note: some versions of huggingface_hub.UploadFolder/ HfApi.upload_folder
        # do not accept a `max_workers` argument. Keep call minimal for compatibility.
        upload_folder(
            folder_path=str(path),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=path.name,
            token=hf_token,
        )
        print(f"\n✓ Uploaded folder {path.name} to dataset {repo_id}:{path.name}")
    except Exception as e:
        # Provide a helpful hint if the error is due to an unexpected arg
        msg = str(e)
        if 'unexpected keyword argument' in msg or 'max_workers' in msg:
            msg += "\nHint: Your installed 'huggingface_hub' may be older/newer and not accept 'max_workers'. Try upgrading with 'pip install -U huggingface_hub' or remove unsupported kwargs."
        print(f"ERROR uploading {folder}: {msg}")

print("\nAll done.")
