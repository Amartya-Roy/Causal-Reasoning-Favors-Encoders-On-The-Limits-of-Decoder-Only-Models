"""
Distill Symbolic Reasoning into LLM.

Elegant idea: 
1. Symbolic solver generates PERFECT reasoning traces
2. Train Qwen to mimic these traces (learn to reason, not just answer)
3. LLM internalizes forward-chaining as a skill

This is "teaching a student by showing worked examples."
"""

import re
import random
import argparse
from dataclasses import dataclass
from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm


# ============================================================================
# SYMBOLIC REASONING ENGINE (the teacher)
# ============================================================================

@dataclass
class Rule:
    antecedents: List[str]
    consequent: str
    
    def __repr__(self):
        if len(self.antecedents) == 1:
            return f"{self.antecedents[0]} → {self.consequent}"
        return f"({' ∧ '.join(self.antecedents)}) → {self.consequent}"


class SymbolicReasoner:
    """Forward chaining with full derivation trace."""
    
    def __init__(self, facts: Set[str], rules: List[Rule]):
        self.initial_facts = set(facts)
        self.facts = set(facts)
        self.rules = rules
        self.trace: List[Tuple[str, Rule, List[str]]] = []  # (derived, rule, antecedents_used)
    
    def step(self) -> bool:
        for rule in self.rules:
            if all(ant in self.facts for ant in rule.antecedents):
                if rule.consequent not in self.facts:
                    self.facts.add(rule.consequent)
                    self.trace.append((rule.consequent, rule, list(rule.antecedents)))
                    return True
        return False
    
    def solve(self, query: str) -> Tuple[bool, List[str]]:
        """Returns (is_derivable, reasoning_steps)"""
        # Run to fixpoint
        while self.step():
            pass
        
        is_derivable = query.lower() in self.facts
        
        # Generate human-readable reasoning trace
        reasoning = self._generate_reasoning(query.lower(), is_derivable)
        return is_derivable, reasoning
    
    def _generate_reasoning(self, query: str, is_derivable: bool) -> List[str]:
        """Generate natural language reasoning trace."""
        steps = []
        
        # Start with initial facts
        if self.initial_facts:
            steps.append(f"Known facts: {', '.join(sorted(self.initial_facts))}")
        else:
            steps.append("No initial facts given.")
        
        # Show derivation steps
        for derived, rule, ants in self.trace:
            if len(ants) == 1:
                steps.append(f"Since {ants[0]} is true, and {ants[0]} → {derived}, therefore {derived} is true.")
            else:
                steps.append(f"Since {', '.join(ants)} are all true, and ({' ∧ '.join(ants)}) → {derived}, therefore {derived} is true.")
        
        # Conclusion
        if is_derivable:
            if query in self.initial_facts:
                steps.append(f"'{query}' is directly stated as a fact. Answer: Yes")
            else:
                steps.append(f"We derived '{query}'. Answer: Yes")
        else:
            steps.append(f"Cannot derive '{query}' from the given facts and rules. Answer: No")
        
        return steps


def parse_question(question: str) -> Tuple[Set[str], List[Rule], str]:
    """Parse question into (facts, rules, query)."""
    facts = set()
    rules = []
    query = ""
    
    question = question.replace('[PERIOD]', '. ').replace('[', '').replace(']', '')
    
    # Query
    qm = re.search(r'Query:\s*\w+\s+is\s+(\w+(?:-\w+)?)', question, re.IGNORECASE)
    if qm:
        query = qm.group(1).lower()
    
    # Facts
    fm = re.search(r'Facts:\s*(.+?)(?:Rules:|Query:|$)', question, re.IGNORECASE | re.DOTALL)
    if fm:
        for m in re.finditer(r'Alice\s+is\s+(\w+(?:-\w+)?)', fm.group(1), re.IGNORECASE):
            facts.add(m.group(1).lower())
    
    # Rules
    rm = re.search(r'Rules:\s*(.+?)(?:Query:|$)', question, re.IGNORECASE | re.DOTALL)
    if rm:
        for segment in re.split(r'[.\n]+', rm.group(1)):
            if 'IMPLY' in segment.upper():
                parts = re.split(r'\s+IMPLY\s+', segment, flags=re.IGNORECASE)
                if len(parts) == 2:
                    ants = [a.strip().lower() for a in re.split(r'\s+AND\s+', parts[0], flags=re.IGNORECASE)]
                    cons = parts[1].strip().lower()
                    if ants and cons:
                        rules.append(Rule(ants, cons))
    
    return facts, rules, query


# ============================================================================
# COT DATA GENERATION
# ============================================================================

def generate_cot_example(question: str, label: int) -> Optional[Dict]:
    """Generate a Chain-of-Thought training example."""
    facts, rules, query = parse_question(question)
    if not query:
        return None
    
    reasoner = SymbolicReasoner(facts, rules)
    is_derivable, reasoning_steps = reasoner.solve(query)
    
    # Verify symbolic answer matches label
    pred = 1 if is_derivable else 0
    if pred != label:
        return None  # Skip mismatches (shouldn't happen)
    
    # Build input prompt
    input_text = f"""Given facts and rules, determine if the query can be derived. Think step by step.

Facts: {', '.join(sorted(facts)) if facts else 'None'}
Rules: {'; '.join(str(r) for r in rules[:10])}{'...' if len(rules) > 10 else ''}
Query: Is '{query}' true?

Reasoning:"""
    
    # Build target (the reasoning + answer)
    target_text = " " + " ".join(reasoning_steps)
    
    return {
        "input": input_text,
        "target": target_text,
        "full": input_text + target_text,
        "label": label,
        "query": query
    }


# ============================================================================
# TRAINING
# ============================================================================

class CoTDataset(Dataset):
    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Tokenize full sequence
        encoding = self.tokenizer(
            ex["full"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels (mask input, only predict target)
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # Find where target starts
        input_len = len(self.tokenizer(ex["input"], truncation=True, max_length=self.max_length)["input_ids"])
        
        labels = input_ids.clone()
        labels[:input_len] = -100  # Mask input tokens
        labels[attention_mask == 0] = -100  # Mask padding
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_accum: int = 4):
    model.train()
    total_loss = 0
    
    optimizer.zero_grad()
    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / grad_accum
        loss.backward()
        
        total_loss += loss.item() * grad_accum
        
        if (i + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    
    return total_loss / len(dataloader)


def evaluate(model, tokenizer, examples: List[Dict], device, max_new_tokens: int = 150) -> Tuple[float, List[Dict]]:
    """Evaluate model and return accuracy + sample outputs."""
    model.eval()
    correct = 0
    results = []
    
    for ex in tqdm(examples[:100], desc="Evaluating"):  # Evaluate on subset
        inputs = tokenizer(ex["input"], return_tensors="pt", truncation=True, max_length=400)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Extract answer
        pred = 1 if "answer: yes" in response.lower() else 0
        is_correct = (pred == ex["label"])
        correct += int(is_correct)
        
        results.append({
            "query": ex["query"],
            "label": ex["label"],
            "pred": pred,
            "correct": is_correct,
            "response": response[:200]
        })
    
    return correct / len(examples[:100]), results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_samples", type=int, default=500)
    parser.add_argument("--eval_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("=" * 80)
    print("DISTILLING SYMBOLIC REASONING INTO QWEN")
    print("=" * 80)
    
    # Load data
    print("\n[1] Loading and generating CoT data...")
    df = pd.read_parquet(args.parquet_path)
    
    all_examples = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating CoT"):
        ex = generate_cot_example(row["Question"], int(row["Response"]))
        if ex:
            all_examples.append(ex)
        if len(all_examples) >= args.train_samples + args.eval_samples:
            break
    
    random.shuffle(all_examples)
    train_examples = all_examples[:args.train_samples]
    eval_examples = all_examples[args.train_samples:args.train_samples + args.eval_samples]
    
    print(f"Generated {len(train_examples)} train, {len(eval_examples)} eval examples")
    
    # Show example
    print("\n[Example CoT]")
    print(train_examples[0]["full"][:600] + "...")
    
    # Load model
    print(f"\n[2] Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.to(device)
    
    # Pre-training evaluation
    print("\n[3] Pre-training evaluation...")
    pre_acc, pre_results = evaluate(model, tokenizer, eval_examples, device)
    print(f"Pre-training accuracy: {pre_acc*100:.1f}%")
    print(f"Sample: {pre_results[0]}")
    
    # Train
    print(f"\n[4] Training for {args.epochs} epochs...")
    train_dataset = CoTDataset(train_examples, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=total_steps)
    
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
        
        # Eval after epoch
        acc, _ = evaluate(model, tokenizer, eval_examples, device)
        print(f"Epoch {epoch+1}: Accuracy = {acc*100:.1f}%")
    
    # Final evaluation
    print("\n[5] Final evaluation...")
    final_acc, final_results = evaluate(model, tokenizer, eval_examples, device)
    print(f"Final accuracy: {final_acc*100:.1f}%")
    
    # Show some results
    print("\n[Sample Outputs]")
    for r in final_results[:3]:
        print(f"  Query: {r['query']} | Label: {r['label']} | Pred: {r['pred']} | Correct: {r['correct']}")
        print(f"  Response: {r['response'][:150]}...")
        print()
    
    # Improvement
    improvement = (final_acc - pre_acc) * 100
    print("=" * 80)
    print(f"IMPROVEMENT: {pre_acc*100:.1f}% → {final_acc*100:.1f}% (+{improvement:.1f}%)")
    print("=" * 80)
    
    # Save
    save_path = "./checkpoints/qwen_cot_finetuned"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
