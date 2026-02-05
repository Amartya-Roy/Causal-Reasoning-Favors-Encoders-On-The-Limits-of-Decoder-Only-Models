def extract_label_from_symbolic_proof(proof_text: str) -> int:
    """
    Extract label from proof text that matches derive_with_explanation output format.
    
    Label 1 (True): Proof contains inference steps like "A ∧ B ⇒ C"
    Label 0 (False): Proof contains failure explanations like:
        - "No rule concludes 'X'"  
        - "Cannot apply rule ... because missing: ..."
    
    Args:
        proof_text: Generated proof chain from model
        
    Returns:
        0 or 1 based on proof content
    """
    # Clean the input
    cleaned = proof_text.strip()
    
    # Remove common prefixes that models might add (anywhere in text, not just start)
    prefixes_to_remove = [
        'proof chain:',
        'explanation:',
        'steps:',
        'reasoning:'
    ]
    
    cleaned_lower = cleaned.lower()
    for prefix in prefixes_to_remove:
        if prefix in cleaned_lower:
            idx = cleaned_lower.find(prefix)
            cleaned = cleaned[idx + len(prefix):].strip()
            cleaned_lower = cleaned.lower()
            break
    
    # Handle empty or very short responses
    if not cleaned or len(cleaned.strip()) < 3:
        return 0
    
    # PRIMARY CHECK: Explicit failure patterns (Label 0)
    failure_patterns = [
        'no rule concludes',           # "No rule concludes 'query'"
        'cannot apply rule',           # "Cannot apply rule ... because missing: ..."
        'because missing:',            # Part of failure explanation
        'missing:',                    # Shortened version
        'no applicable rule',          # Alternative phrasing
        'no valid rule',               # Alternative phrasing
        'cannot prove',                # General failure
        'impossible to prove',         # General failure
        'no rules lead to',            # Alternative phrasing
        'cannot derive',               # Alternative phrasing
        'derivation failed',           # Alternative phrasing
        'proof failed'                 # Alternative phrasing
    ]
    
    # Check for explicit failure indicators
    for pattern in failure_patterns:
        if pattern in cleaned_lower:
            return 0
    
    # SECONDARY CHECK: Look for inference steps (Label 1)
    # These patterns indicate successful proof derivation
    success_patterns = [
        '⇒',                          # Implication symbol: "A ∧ B ⇒ C"
        '∧',                          # Conjunction symbol: "shy ∧ vivacious"
        '→',                          # Alternative implication
        ' implies ',                  # Text version of implication
        ' imply ',                    # Text version of implication
        'therefore',                  # Logical conclusion
        'thus',                       # Logical conclusion
        'hence'                       # Logical conclusion
    ]
    
    # Check for inference indicators
    for pattern in success_patterns:
        if pattern in cleaned_lower:
            return 1
    
    # TERTIARY CHECK: Pattern matching for rule applications
    # Look for patterns like "A and B implies C" or "A, B → C"
    import re
    
    # Pattern: "word and word implies word" or "word, word implies word"
    rule_pattern = r'\b\w+\s+(and|∧|,)\s+\w+\s+(implies?|⇒|→)\s+\w+'
    if re.search(rule_pattern, cleaned_lower):
        return 1
    
    # Pattern: Multi-line proof steps (each line is an inference)
    lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
    if len(lines) >= 2:
        # Check if lines look like proof steps vs failure explanations
        inference_lines = 0
        failure_lines = 0
        
        for line in lines:
            line_lower = line.lower()
            if any(pattern in line_lower for pattern in success_patterns):
                inference_lines += 1
            elif any(pattern in line_lower for pattern in failure_patterns):
                failure_lines += 1
        
        # If more inference lines than failure lines, likely a proof
        if inference_lines > failure_lines and inference_lines >= 1:
            return 1
        elif failure_lines > 0:
            return 0
    
    # QUATERNARY CHECK: Length and content heuristics
    # Very short responses are likely failures
    if len(cleaned) < 10:
        return 0
    
    # If we have substantial content but no clear patterns,
    # check for reasoning-like structure (exclude structural markers)
    structural_markers = ['<think>', '</think>', 'proof chain:', 'explanation:', 'steps:', 'reasoning:']
    content_lines = [line for line in lines if line.lower().strip() not in structural_markers]
    
    if len(content_lines) >= 2 and all(len(line) >= 10 for line in content_lines):
        # Multiple substantial content lines suggest a proof attempt
        return 1
    
    # FINAL FALLBACK: Conservative approach
    # If unclear, default to failure (0) to avoid false positives
    return 0


def test_extraction_function():
    """Test the extraction function with example inputs"""
    
    test_cases = [
        # Success cases (Label 1)
        ("Proof chain:\nshy ∧ vivacious ⇒ aggressive", 1),
        ("uptight ⇒ tame\nfriendly ∧ tame ⇒ fragile", 1),
        ("Alice is hot\nhot implies aggressive\nTherefore Alice is aggressive", 1),
        
        # Failure cases (Label 0)  
        ('No rule concludes "lonely."', 0),
        ("Cannot apply rule \"uptight ∧ tame ⇒ fragile\" because missing: tame.", 0),
        ("Cannot apply rule because missing: hot, vivacious.", 0),
        ("No applicable rule found", 0),
        ("", 0),
        ("Proof failed", 0),
        
        # Edge cases
        ("Very short", 0),  # Too short, default to failure
        ("This is a longer text but contains no logical indicators whatsoever", 0),  # No clear pattern
    ]
    
    print("Testing extraction function:")
    print("="*80)
    
    for i, (proof, expected) in enumerate(test_cases):
        result = extract_label_from_symbolic_proof(proof)
        status = "✓" if result == expected else "✗"
        print(f"{status} Test {i+1}: Expected {expected}, Got {result}")
        print(f"   Input: {repr(proof[:60])}{'...' if len(proof) > 60 else ''}")
        if result != expected:
            print(f"   ⚠️  MISMATCH!")
        print()
    
    return True

if __name__ == "__main__":
    test_extraction_function()