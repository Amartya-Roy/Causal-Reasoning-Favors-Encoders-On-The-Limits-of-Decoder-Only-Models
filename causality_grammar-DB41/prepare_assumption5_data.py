#!/usr/bin/env python3
"""
Prepare data for Assumption 5 verification.
Converts the parquet training data to the CSV format expected by verify_assumption5.py
"""

import pandas as pd
import argparse
import os


def prepare_data(input_path: str, output_path: str, sample_size: int = None):
    """
    Convert parquet data to CSV format for Assumption 5 verification.
    
    Expected input columns: Question, Complex_CoT, Response, Depth
    Output columns: question, proof_chain, label
    """
    print(f"Loading data from: {input_path}")
    df = pd.read_parquet(input_path)
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Rename columns to match expected format
    df_out = pd.DataFrame({
        'question': df['Question'].astype(str),
        'proof_chain': df['Complex_CoT'].astype(str),
        'label': df['Response'].astype(int),
        'depth': df['Depth'].astype(int) if 'Depth' in df.columns else 0
    })
    
    # Clean up any newline issues in column names
    df_out.columns = [c.strip() for c in df_out.columns]
    
    # Remove any rows with empty proofs
    df_out = df_out[df_out['proof_chain'].str.len() > 0]
    
    print(f"\nLabel distribution:")
    print(df_out['label'].value_counts())
    
    if sample_size and sample_size < len(df_out):
        # Stratified sample
        pos = df_out[df_out['label'] == 1].sample(n=min(sample_size//2, len(df_out[df_out['label'] == 1])), random_state=42)
        neg = df_out[df_out['label'] == 0].sample(n=min(sample_size//2, len(df_out[df_out['label'] == 0])), random_state=42)
        df_out = pd.concat([pos, neg]).sample(frac=1, random_state=42)  # Shuffle
        print(f"\nSampled {len(df_out)} rows")
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    print(f"Final shape: {df_out.shape}")
    
    # Show sample
    print("\nSample row:")
    print(df_out.iloc[0])
    
    return df_out


def main():
    parser = argparse.ArgumentParser(description="Prepare data for Assumption 5 verification")
    parser.add_argument("--input", type=str, default="data/train_data.parquet", help="Input parquet file")
    parser.add_argument("--output", type=str, default="data/assumption5_data.csv", help="Output CSV file")
    parser.add_argument("--sample_size", type=int, default=2000, help="Number of samples (None for all)")
    
    args = parser.parse_args()
    prepare_data(args.input, args.output, args.sample_size)


if __name__ == "__main__":
    main()
