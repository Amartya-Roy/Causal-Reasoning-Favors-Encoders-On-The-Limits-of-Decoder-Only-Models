import requests
import pandas as pd
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# import evaluate
from collections import defaultdict

from huggingface_hub import InferenceClient


def load_data():

    # train = load_dataset("causality-grammar/logic_explanations", data_files={
    #     "train": "deepseek_5k_RP.parquet"
    # })

    test = load_dataset("causality-grammar/logic_explanations", data_files={
        "test": "deepseek_2k_RP_test.parquet"
    })    

    return test


def prepare_model(provider, model_name, nvidia_api_key=None, hf_token=None, openai_key=None, deepseek_key=None):
    """
    Returns a unified `predict(prompt)` function.
    
    provider: one of ['hf-local', 'hf-api', 'openai', 'deepseek']
    """

    if provider == "hf-local":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        def predict(prompt):
            out = pipe(prompt, max_new_tokens=10, do_sample=False)[0]["generated_text"]
            return out.strip()

    elif provider == "hf-api":
        assert hf_token is not None, "Hugging Face token required for HF API"
        client = InferenceClient(model=model_name, token=hf_token)

        def predict(prompt):
            out = client.text_generation(prompt, max_new_tokens=10, do_sample=False)
            return out.strip()

    elif provider == "openai":
        assert openai_key is not None, "OpenAI API key required"
        openai.api_key = openai_key

        def predict(prompt):
            response = openai.ChatCompletion.create(
                model=model_name,  # "gpt-3.5-turbo" or "gpt-4"
                messages=[
                    {"role": "system", "content": "Answer with only 1 (true) or 0 (false)."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10,
            )
            return response.choices[0].message["content"].strip()

    elif provider == "deepseek":
        assert deepseek_key is not None, "DeepSeek API key required"

        def predict(prompt):
            headers = {
                "Authorization": f"Bearer {deepseek_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model_name,  # usually "deepseek-reasoner"
                "messages": [
                    {"role": "system", "content": "Answer with only 1 (true) or 0 (false)."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0,
                "max_tokens": 10,
            }
            resp = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            else:
                print("Error:", resp.text)
                return "ERROR"

    elif provider == "nvidia-nim":
        assert nvidia_api_key is not None, "NVIDIA API key is required for NIM"

        def predict(prompt):
            headers = {
                "Authorization": f"Bearer {nvidia_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": model_name,  # e.g., "meta/llama3-70b-instruct"
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that only answers with 1 (true) or 0 (false)."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0,
                "max_tokens": 10
            }

            response = requests.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers=headers,
                json=payload
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                print("NVIDIA API Error:", response.status_code, response.text)
                return "ERROR"

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return predict

def get_metrics(results):

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save to CSV (optional)
    df.to_csv("model_predictions.csv", index=False)

    # === Evaluation: Accuracy by depth ===
    accuracy = evaluate.load("accuracy")
    grouped = df[df["predicted"].isin(["0", "1"])].groupby("depth")

    print("\n=== Accuracy by Depth ===")
    for depth, group in grouped:
        acc = accuracy.compute(predictions=group["predicted"], references=group["expected"])
        # print(f"Depth {depth}: {acc['accuracy']:.3f} ({len(group)} samples)")

    # === Overall accuracy ===
    valid_df = df[df["predicted"].isin(["0", "1"])]
    overall = accuracy.compute(predictions=valid_df["predicted"], references=valid_df["expected"])
    # print(f"\nOverall Accuracy: {overall['accuracy']:.3f}")

    return grouped, overall



def main():

    test = load_data()
    test_data = test["test"].select(range(10))

    depth_samples = []

    for d in range(12):  # depths from 0 to 11
        # Filter dataset for current depth
        subset = test["test"].filter(lambda x: x["Depth"] == d)
        
        sampled = subset.shuffle(seed=42).select(range(2)) # sampling 2 points from each depth
        
        depth_samples.append(sampled)

    # Combine all sampled subsets
    sampled_dataset = concatenate_datasets(depth_samples)

    print("dataset ready")

    predict = prepare_model(
        provider="nvidia-nim",  # or "openai", "hf-api", "hf-local"
        model_name="deepseek-r1",
        # hf_token="hf_...",         # if using HF API
        # openai_key="sk-...",       # if using OpenAI
        # deepseek_key="sk-..."      # if using DeepSeek
        nvidia_api_key = "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"
    )
    print("model ready")
    # accuracy = evaluate.load("accuracy")
    results = []

    for item in sampled_dataset:
        prompt = item["Question"]
        expected = str(item["Response"])  # Make sure it's a string for consistency
        depth = item["Depth"]

        output = predict(prompt)

        # Parse predicted label from output
        if "1" in output:
            pred = "1"
        elif "0" in output:
            pred = "0"
        else:
            pred = "UNK"

        results.append({
            "prompt": prompt,
            "expected": expected,
            "predicted": pred,
            "raw_output": output,
            "depth": depth
        })

    for i in results:
        print(i["expected"], i["predicted"])
        print(i["raw_output"])


if __name__ == "__main__":

    main()
