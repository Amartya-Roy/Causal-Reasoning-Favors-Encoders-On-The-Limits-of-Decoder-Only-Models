import json
import pandas as pd

from datasets import Dataset

def prepare_rules(rules):
    
    rules_text = "Rules:\n"
    
    for rule in rules:
        if len(rule[0]) == 1:
            rules_text = rules_text + rule[0][0] + " [IMPLY] " + rule[1] + " [PERIOD]\n"
        else:
            for i in range(len(rule[0])-1):
                rules_text = rules_text + rule[0][i] + " [AND] "
            rules_text = rules_text + rule[0][-1] + " [IMPLY] " + rule[1] + " [PERIOD]\n"

    return rules_text

def prepare_facts(facts):
    facts_text = "Facts: "
    for fact in facts:
        facts_text = facts_text + " Batman is " + fact + " [PERIOD]" 

    return facts_text + "\n"

def prepare_explanations(explanations, label):
    explanations_text = "Proof chain:\n"
    for explanation in explanations:
        explanations_text = explanations_text + "   " + explanation + "\n"

    return explanations_text

def prepare_prompt(rules, facts, query):

    content = ""
    facts_text = prepare_facts(facts)
    rules_text = prepare_rules(rules)

    content = facts_text + rules_text + f"Query: Batman is {query} [PERIOD]"

    return content

def prepare_sample(example):

    prompt = prepare_prompt(example['rules'], example['facts'], example['query'])
    answer = prepare_explanations(example['explanation'], example['label'])

    return prompt, answer


def load_examples(input_file):
    with open(input_file, "r") as f:
        data = json.load(f)
    return data


def main(model_name):

    data = load_examples("./data/test_samples1.json")

    examples = data[1]
    algo = data[0]

    if model_name == "mistral":
        data = {"prompt": list(), "depth": list(), "messages": list()}
        for example in examples:
            
            prompt, answer = prepare_sample(example)
            answer = answer + f"Label: {example['label']}"
            message = [{"content": prompt, "role": "user"}, {"content": answer, "role": "assistant"}]

            data["prompt"].append(prompt)
            data["depth"].append(example['depth'])
            data["messages"].append(message)

        dataset = Dataset.from_dict(data)

    elif model_name == "deepseek":
        data = {"Question": list(), "Complex_CoT": list(), "Response": list(), "Depth": list()}
        for example in examples:
            
            prompt, answer = prepare_sample(example)

            data["Question"].append(prompt)
            data["Complex_CoT"].append(answer) 
            data["Response"].append(example['label'])
            data["Depth"].append(example['depth'])

        dataset = Dataset.from_dict(data)

    else:
        pass
    
    df = dataset.to_pandas()
    df.to_parquet(f"./data/{model_name}_2k_{algo}_test.parquet", index=False)
    
    print(dataset)
    print(dataset.features)

if __name__ == "__main__":

    main('mistral')
    main('deepseek')
    # for example in data[4:6]:
    #     prompt, answer = prepare_sample(example)
    #     print(f"PROMPT:\n{prompt}\n")
    #     print(f"ANSWER:\n{answer}\n")

    # print(data[0].keys())
    # print(data[0]['rules'])
