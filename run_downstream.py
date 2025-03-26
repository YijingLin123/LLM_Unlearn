import argparse
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re
import json
import numpy as np

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()
    return tokenizer, model

# === ARC ===
def format_arc_prompt(question, choices):
    prompt = question.strip() + "\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice.strip()}\n"
    prompt += "Answer:"
    return prompt

def evaluate_arc(model_path, data_path, max_samples=None):
    tokenizer, model = load_model(model_path)
    model.eval()
    dataset = Dataset.from_json(data_path)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    total = len(dataset)

    for item in tqdm(dataset, desc="Evaluating ARC (LL ranking)"):
        question = item["question"]
        choices = item["choices"]["text"]
        answer_key = item["answerKey"]

        prompt = format_arc_prompt(question, choices)

        choice_scores = []
        for i, choice in enumerate(choices):
            label = chr(65 + i)
            full_input = prompt + " " + label
            inputs = tokenizer(full_input, return_tensors="pt").to(model.device)
            input_ids = inputs["input_ids"]

            with torch.no_grad():
                logits = model(**inputs).logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                selected = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
                score = selected.sum().item()
                choice_scores.append(score)

        pred_idx = int(np.argmax(choice_scores))
        pred = chr(65 + pred_idx)

        if pred == answer_key:
            correct += 1

    acc = correct / total
    print(f"\n✅ ARC Accuracy (LL ranking): {acc * 100:.2f}%")

# === GSM8K ===
def evaluate_gsm8k(model_path, data_path, max_samples=None):
    tokenizer, model = load_model(model_path)
    dataset = Dataset.from_json(data_path)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    for item in tqdm(dataset, desc="Evaluating GSM8K"):
        prompt = item["question"].strip() + "\nAnswer:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        outputs = model.generate(input_ids, max_new_tokens=64)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred = re.findall(r"-?\d+\.?\d*", decoded.replace(prompt, ""))
        pred_answer = pred[0] if pred else None
        gt_answer = re.findall(r"-?\d+\.?\d*", item["answer"])
        if pred_answer and gt_answer and pred_answer == gt_answer[0]:
            correct += 1

    acc = correct / len(dataset)
    print(f"\n✅ GSM8K Accuracy: {acc * 100:.2f}%")

# === HumanEval ===
def evaluate_humaneval(model_path, data_path, max_samples=None):
    tokenizer, model = load_model(model_path)
    dataset = Dataset.from_json(data_path)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    pass_count = 0
    total = 0

    for item in tqdm(dataset, desc="Evaluating HumanEval"):
        prompt = item["prompt"]
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        outputs = model.generate(input_ids, max_new_tokens=128)
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "")
        code = prompt + completion
        try:
            exec_globals = {}
            exec(code, exec_globals)
            fn_name = re.findall(r"def (\w+)\(", prompt)[0]
            test_case = item["test"]
            exec(test_case, exec_globals)
            pass_count += 1
        except Exception:
            pass
        total += 1

    acc = pass_count / total
    print(f"\n✅ HumanEval pass@1: {acc * 100:.2f}%")

# === MMLU ===
def convert_numeric_answer_to_letter(example):
    if isinstance(example["answer"], str) and example["answer"].isdigit():
        idx = int(example["answer"])
        if 0 <= idx < 4:
            example["answer"] = chr(65 + idx)  # "0" → "A", "1" → "B", etc.
    return example

def format_mmlu_prompt(question, choices):
    prompt = question.strip() + "\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice.strip()}\n"
    prompt += "\nAnswer (A/B/C/D):"
    return prompt

def evaluate_mmlu(model_path, data_path, max_samples=None):
    tokenizer, model = load_model(model_path)
    model.eval()

    dataset = Dataset.from_json(data_path)

    def fix(example):
        answer = str(example["answer"])
        if answer.isdigit():
            example["answer"] = chr(65 + int(answer))  # "0" -> "A"
        return example
    dataset = dataset.map(fix)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    total = len(dataset)

    for item in tqdm(dataset, desc="Evaluating MMLU (LL-based)"):
        question = item["question"]
        choices = item["choices"]
        gt_answer = item["answer"]

        # 构造统一 prompt
        prompt = format_mmlu_prompt(question, choices)

        choice_scores = []
        for i, choice in enumerate(choices):
            label = chr(65 + i)  # A, B, C, D
            full_input = prompt + " " + label

            inputs = tokenizer(full_input, return_tensors="pt").to(model.device)
            input_ids = inputs["input_ids"]

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            # shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            selected_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

            # 计算该选项 log-prob 总和
            option_score = selected_log_probs.sum().item()
            choice_scores.append(option_score)

        pred_idx = int(np.argmax(choice_scores))
        pred_answer = chr(65 + pred_idx)

        if pred_answer == gt_answer:
            correct += 1

    acc = correct / total
    print(f"Correct: {correct} / Total: {total}")
    print(f"\n✅ MMLU Accuracy (Log-Likelihood Ranking): {acc * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["arc", "gsm8k", "humaneval", "mmlu"], help="Evaluation task")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--max_samples", type=int, default=None, help="Max number of samples")
    args = parser.parse_args()

    if args.task == "arc":
        evaluate_arc(args.model_path, "./dataset_cache/downstream_tasks/downstream_arc_test.json", args.max_samples)
    elif args.task == "gsm8k":
        evaluate_gsm8k(args.model_path, "./dataset_cache/downstream_tasks/downstream_gsm8k_test.json", args.max_samples)
    elif args.task == "humaneval":
        evaluate_humaneval(args.model_path, "./dataset_cache/downstream_tasks/downstream_humaneval.json",
                           args.max_samples)
    elif args.task == "mmlu":
        evaluate_mmlu(args.model_path, "./dataset_cache/downstream_tasks/mmlu_econometrics.json", args.max_samples)
