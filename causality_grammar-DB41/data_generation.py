import argparse
import time
import re
import json
import random
from tqdm import tqdm
import heapq


def derive_with_explanation(facts, rules, query):
    """
    facts:   set[str]              e.g. {"aggressive","hot"}
    rules:   list[ (tuple[str,...], str) ]
             e.g. [(("difficult","uptight"), "tense"), …]
    query:   str                   e.g. "tense"

    Returns: (label:int, explanation:list[str])
    """
    # --- 1) Forward‐chaining distances & parents ---
    dist   = {a: 0 for a in facts}
    parent = {}   # parent[conclusion] = (rule_index, premises)
    pq     = [(0, a) for a in facts]
    heapq.heapify(pq)

    # Build waiting list for hyperedges
    waiting   = {}   # atom → [rule_idx,…]
    sat_count = [0]*len(rules)
    for i, (prem, conc) in enumerate(rules):
        for p in prem:
            waiting.setdefault(p, []).append(i)

    # Dijkstra‐style forward
    while pq:
        cost_x, x = heapq.heappop(pq)
        if cost_x > dist[x]:
            continue
        if x == query:
            break

        for ri in waiting.get(x, []):
            sat_count[ri] += 1
            prem, conc = rules[ri]
            if sat_count[ri] == len(prem):
                c = max(dist[p] for p in prem) + 1
                if conc not in dist or c < dist[conc]:
                    dist[conc]         = c
                    parent[conc]       = (ri, prem)
                    heapq.heappush(pq, (c, conc))

    def build_proof(atom):
        if atom in facts:
            # print(f"atom: \"{atom}\" is in facts.")
            return []
        ri, prem = parent[atom]
        proof = []
        for p in prem:
            proof += build_proof(p)

        # proof.append(f"{' ^ '.join(prem)} -> {atom}")
        proof.append(f"{' ∧ '.join(prem)} ⇒ {atom}")
        return proof

    # Decide label and explanation in one place
    if query in dist:
        label       = 1
        explanation = build_proof(query)
    else:
        label       = 0
        # build failure explanation as before
        expl = []
        candidate_rules = [
            (i, prem) for i,(prem,conc) in enumerate(rules) if conc==query
        ]
        if not candidate_rules:
            expl.append(f"No rule concludes “{query}.”")
        else:
            for _, prem in candidate_rules:
                available = [p for p in prem if p in dist]
                missing   = [p for p in prem if p not in dist]
                for p in available:
                    expl += build_proof(p)
                expl.append(
                    f"Cannot apply rule “{' ∧ '.join(prem)} ⇒ {query}” "
                    f"because missing: {', '.join(missing)}."
                )
        explanation = expl

    return label, explanation

def read_vocab(vocab_file):
    vocab = []
    with open(vocab_file, 'r') as fin:
        vocab = [line.strip() for line in fin.readlines()]
    print('vocabulary size: ', len(vocab))
    return vocab


def sample_one_rule(preds):
    head_num = random.randint(1, 3)
    lits = random.sample(preds, min(head_num + 1, len(preds)))
    random.shuffle(lits)
    return (lits[:-1], lits[-1])

def sample_rule_priority(preds):
    pred_num = len(preds)
    rule_num = random.randint(0, 4 * pred_num)
    fact_num = random.randint(0, pred_num)

    cache = set()
    rules = []
    for _ in range(0, rule_num):
        rule = None
        while True:
            rule = sample_one_rule(preds)
            rule_hash = ' '.join(sorted(rule[0])) + ' ' + rule[1]
            if rule_hash not in cache:
                cache.add(rule_hash)
                break
        rules.append(rule)

    facts = random.sample(preds, fact_num)

    query = random.sample(preds, 1)[0]

    return rules, facts, query

def forward_chain(rules, facts):
    res = {}
    for fact in facts:
        res[fact] = 0

    depth = 1
    prev_len = 0
    while len(res) > prev_len:
        new_facts = []
        for rule in rules:
            head, tail = rule
            if all([lit in res for lit in head]):
                new_facts.append(tail)
        prev_len = len(res)
        for fact in new_facts:
            if fact not in res:
                res[fact] = depth
        depth += 1

    return res

def backward_chain_(u, depth, rules, facts, max_depth, ances):
    INF = 100000000
    if u in facts:
        return INF
    if u in ances or depth == max_depth:
        return depth

    res = depth
    for rule in [x for x in rules if x[1] == u]:
        head, _ = rule
        tmp = INF
        for lit in head:
            ances.add(u)
            tmp = min(tmp, backward_chain_(lit,
                depth + 1, rules, facts, max_depth, ances))
            ances.remove(u)
        res = max(res, tmp)
    return res


def backward_chain(query, rules, facts, max_depth):
    return backward_chain_(query, 0, rules, facts, max_depth, set())


def process_example(example):
    
    res = forward_chain(example['rules'], example['facts'])
    res_label = 1 if example['query'] in res else 0 

    label, explanation = derive_with_explanation(example['facts'], example['rules'], example['query']) 

    depth = len(explanation)

    if res_label == label:
        example['label'] = label
        example['explanation'] = explanation

    else:
        print("Not Equal")
        example = None
    
    return example

def sample_one_example(vocab, min_pred_num, max_pred_num, algo):
    pred_num = random.randint(min_pred_num, max_pred_num)
    preds = random.sample(vocab, pred_num)
    if algo == 'RP':
        rules, facts, query = sample_rule_priority(preds)
    # if algo == 'LP':
    #     rules, facts, query = sample_label_priority(preds)
    # if algo == 'LP_STAR':
    #     rules, facts, query = sample_lp_star(preds)

    if query is None:
        return None

    example = {
        'preds': preds,
        'rules': rules,
        'facts': facts,
        'query': query
    }

    example = process_example(example)

    return example

def sample_examples(num_per_depth, depths, vocab, min_pred_num, max_pred_num, algo):
    
    examples = []
    reasoning_depth = {str(i): [0, 0] for i in depths}

    num_depth = 0  
    total_time = 0
    start = time.time()
    print(reasoning_depth.keys())
    while True:
        example = None
        while example is None:
            example = sample_one_example(vocab, min_pred_num, max_pred_num, algo)

        depth = len(example['explanation'])

        if str(depth) in reasoning_depth.keys() and str(depth) != '0':

            if example['label'] == 1:            
                if reasoning_depth[str(depth)][0] < num_per_depth // 2:
            
                    reasoning_depth[str(depth)][0] += 1     
                    example['depth'] = depth
                    examples.append(example)

            elif example['label'] == 0:            
                if reasoning_depth[str(depth)][1] < num_per_depth // 2:
                    
                    reasoning_depth[str(depth)][1] += 1     
                    example['depth'] = depth
                    examples.append(example)
            
            else:
                continue
        
        elif str(depth) == '0':
            # print('depth = 0')
            if reasoning_depth[str(depth)][0] < num_per_depth:
        
                reasoning_depth[str(depth)][0] += 1     
                example['depth'] = depth
                examples.append(example)
        
        else:
            continue


        counts = [(key, value[0] + value[1]) for key, value in reasoning_depth.items() if value[0] + value[1] == num_per_depth]

        if len(counts) > num_depth:
            end = time.time()
            num_depth += 1
            print(counts, end = ' ')
            print(f"time taken: {end - start: .2f} s")
            total_time += end - start
            start = time.time()


        if sum([x[1] for x in counts]) == num_per_depth*len(depths):
            break

    print(f"Total time taken to generate samples: {total_time / 60 : .2f} mins")
    return examples

def write_examples(data, output_file):
    random.shuffle(data[1])
    with open(output_file, 'w') as fout:
        json.dump(data, fout)

def main():
    min_pred_num, max_pred_num = 5, 30
    algo = "RP"
    
    vocab = read_vocab('./data/vocab.txt')
    
    # depths = list(range(8))
    # num_per_depth = 5000
    depths = list(range(12))
    num_per_depth = 2000

    # for depth in reasoning_depth:

    examples = sample_examples(num_per_depth, depths, vocab, min_pred_num, max_pred_num, algo)
    write_examples([algo, examples], './data/test_samples1.json')    
    return None

if __name__ == "__main__":

    main()
# 1. data format 
# 2. hf dataset: prompt, depth, message
