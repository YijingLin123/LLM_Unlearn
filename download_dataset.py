import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, load_from_disk, DatasetDict

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

parser = argparse.ArgumentParser(description="Download unlearning datasets")
parser.add_argument("--dataset", type=str, choices=["arxiv", "github"], help="Choose dataset: arxiv, github")
parser.add_argument("--all", action="store_true", help="Download all datasets")
parser.add_argument("--downstream", action="store_true", help="Also download downstream eval datasets")

args = parser.parse_args()

# **预处理：转换 `text` 为 `input_ids` 和 `labels`**
def get_preprocess_function(tokenizer):
    def preprocess(examples):
        tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    return preprocess

model_name = "Yi-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
preprocess_function = get_preprocess_function(tokenizer)

def download_downstream_task_data():
    os.makedirs("./dataset_cache/downstream_tasks", exist_ok=True)

    print("\n📦 下载下游评估任务数据集...")

    # 1. ARC
    arc = load_dataset("ai2_arc", "ARC-Challenge", split="test")
    arc.to_json("./dataset_cache/downstream_tasks/downstream_arc_test.json")
    print("✅ ARC 已保存")

    # 2. GSM8K
    gsm8k = load_dataset("gsm8k", "main", split="test")
    gsm8k.to_json("./dataset_cache/downstream_tasks/downstream_gsm8k_test.json")
    print("✅ GSM8K 已保存")

    # 3. HumanEval
    try:
        humaneval = load_dataset("openai_humaneval")
        humaneval["test"].to_json("./dataset_cache/downstream_tasks/downstream_humaneval.json")
        print("✅ HumanEval 已保存")
    except Exception as e:
        print(f"❌ HumanEval 下载失败：{e}")

    print("📚 正在下载 MMLU 子任务：abstract_algebra")
    dataset = load_dataset("cais/mmlu", name="abstract_algebra", split="test")
    dataset.to_json("./dataset_cache/downstream_tasks/mmlu_abstract_algebra.json")
    print(f"✅ MMLU 子任务abstract_algebra已保存")
    print("🎉 所有下游任务数据集已下载并保存！")

if args.downstream:
    download_downstream_task_data()

datasets_to_download = ["arxiv", "github"]
if args.dataset:
    datasets_to_download = [args.dataset]  # 只下载指定的数据集
elif args.all:
    datasets_to_download = ["arxiv", "github"]  # 下载所有数据集

for dataset_name in datasets_to_download:
    print(f"🚀 Downloading dataset: {dataset_name} ...")

    for split in ["forget", "approximate", "retain"]:
        print(f"   🔹 Loading split: {split} ...")
        dataset = load_dataset("llmunlearn/unlearn_dataset", name=dataset_name, split=split, cache_dir="./dataset_cache")
        dataset = dataset.map(preprocess_function, remove_columns=["text"])
        save_path = f"./dataset_cache/unlearn_dataset_{dataset_name}_{split}"
        dataset.save_to_disk(save_path)
        print(f"   ✅ Saved: {save_path}")

print("🎉 All selected datasets have been downloaded and processed!")

# **加载数据集**
# forget_dataset = load_dataset("llmunlearn/unlearn_dataset", name="arxiv", split="forget", cache_dir="./dataset_cache")
# forget_dataset = forget_dataset.map(preprocess_function, remove_columns=["text"])
# forget_dataset.save_to_disk("./dataset_cache/unlearn_dataset_forget")
# print("Save dataset ./dataset_cache/unlearn_dataset_forget")
#
# approximate_dataset = load_dataset("llmunlearn/unlearn_dataset", name="arxiv", split="approximate", cache_dir="./dataset_cache")
# approximate_dataset = approximate_dataset.map(preprocess_function, remove_columns=["text"])
# approximate_dataset.save_to_disk("./dataset_cache/unlearn_dataset_approximate")
# print("Save dataset ./dataset_cache/unlearn_dataset_approximate")
#
# retain_dataset = load_dataset("llmunlearn/unlearn_dataset", name="arxiv", split="retain", cache_dir="./dataset_cache")
# retain_dataset = retain_dataset.map(preprocess_function, remove_columns=["text"])
# retain_dataset.save_to_disk("./dataset_cache/unlearn_dataset_retain")
# print("Save dataset ./dataset_cache/unlearn_dataset_retain")

