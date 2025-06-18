import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, load_from_disk
import math
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import numpy as np
import argparse

# **🚀 解析命令行参数**
parser = argparse.ArgumentParser(description="Evaluate Unlearning Methods")
parser.add_argument("--method", type=str, choices=["gradient_ascent", "approximate_retrain", "random_label", "ascent_plus_descent_retain", "ascent_plus_descent_general", "ascent_plus_kl_retain", "ascent_plus_kl_general"], required=True,
                    help="Choose the model to evaluate: 'gradient_ascent' or 'approximate_retrain' or 'random_label' or 'ascent_plus_descent_retain' or 'ascent_plus_descent_general'")
args = parser.parse_args()

# **🚀 设置模型路径**
if args.method == "gradient_ascent":
    unlearned_model_path = "./lora_gradient_ascent_model"
elif args.method == "approximate_retrain":
    unlearned_model_path = "./lora_approximate_retrain_model"
elif args.method == "random_label":
    unlearned_model_path = "./lora_random_label_model"
elif args.method == "adversarial_sample":
    unlearned_model_path = "./lora_adversarial_sample_model"
elif args.method == "ascent_plus_descent_retain":
    unlearned_model_path = "./lora_ascent_plus_descent_retain_model"
elif args.method == "ascent_plus_descent_general":
    unlearned_model_path = "./lora_ascent_plus_descent_general_model"
elif args.method == "ascent_plus_kl_retain":
    unlearned_model_path = "./lora_ascent_plus_kl_retain_model"
elif args.method == "ascent_plus_kl_general":
    unlearned_model_path = "./lora_ascent_plus_kl_general_model"

tokenizer = AutoTokenizer.from_pretrained(unlearned_model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    unlearned_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

# **预处理：转换 `text` 为 `input_ids` 和 `labels`**
def preprocess_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()  # 🚀 关键步骤
    return tokenized

forget_dataset = load_from_disk("./dataset_cache/unlearn_dataset_forget")
approximate_dataset = load_from_disk("./dataset_cache/unlearn_dataset_approximate")
# print(approximate_dataset[0])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

# MIA Atttack
def compute_min_k_ppl_acc(selected_log_probs, mask, k, predicts_mask):
    average_log_probs = []
    for sample_log_probs, sample_mask in zip(selected_log_probs, mask):
        sample_log_probs_nonpad = sample_log_probs[sample_mask]
        k_value = int(k * sample_log_probs_nonpad.size(0))
        if k_value > 0:
            min_k_log_probs = torch.topk(sample_log_probs_nonpad, k_value, largest=False).values
            average_log_probs.append(min_k_log_probs.mean())
    return torch.stack(average_log_probs).cpu().numpy()


def compute_mia_scores(model, dataset, batch_size=1):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    model.eval()

    selected_log_probs_list, mask_list = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔄 计算 MIA 进度", unit="batch"):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits

            labels = batch["labels"][:, 1:]
            pad_token_mask = labels != -100
            log_probs = torch.log_softmax(logits, dim=-1)[:, :-1]

            input_ids_expanded = labels.unsqueeze(-1)
            input_ids_expanded[input_ids_expanded == -100] = 0
            selected_log_probs = log_probs.gather(2, input_ids_expanded).squeeze(-1) * pad_token_mask

            selected_log_probs_list.append(selected_log_probs)
            mask_list.append(pad_token_mask)

    selected_log_probs = torch.cat(selected_log_probs_list, dim=0)
    mask = torch.cat(mask_list, dim=0)

    mia_scores = {}
    for ratio in [0.3, 0.4, 0.5, 0.6, 1]:
        mia_scores[f"min_{int(ratio * 100)}_value"] = compute_min_k_ppl_acc(selected_log_probs, mask, ratio, None)

    return mia_scores

def compute_auc(forget_scores, approximate_scores):
    labels = np.concatenate([np.ones_like(forget_scores), np.zeros_like(approximate_scores)])
    scores = np.concatenate([forget_scores, approximate_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score

mia_forget_scores = compute_mia_scores(model, forget_dataset)
mia_approximate_scores = compute_mia_scores(model, approximate_dataset)

for key in mia_forget_scores.keys():
    auc_result = compute_auc(mia_forget_scores[key], mia_approximate_scores[key])
    print(f"🚀 MIA Attack AUC ({key}): {auc_result[2]:.4f}")

# 🚀 MIA Attack AUC (min_30_value): 0.6373
# 🚀 MIA Attack AUC (min_40_value): 0.6359
# 🚀 MIA Attack AUC (min_50_value): 0.6351
# 🚀 MIA Attack AUC (min_60_value): 0.6352
# 🚀 MIA Attack AUC (min_100_value): 0.6365

