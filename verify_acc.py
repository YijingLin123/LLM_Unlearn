import os
import random
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
    get_scheduler,
    BitsAndBytesConfig
)
from peft import PeftModel,get_peft_model, LoraConfig, prepare_model_for_kbit_training
from datasets import load_from_disk, concatenate_datasets
from safetensors.torch import load_file as load_safetensors
from torch.optim import AdamW
from tqdm import tqdm


model_name = "Yi-6B"
dataset_select = "arxiv"
# dataset_select = "github"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir="./hf_cache"
).eval()

# 3. 注入 LoRA（关闭 dropout）
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model = prepare_model_for_kbit_training(model)

# 4. 只更新 LoRA 层
for name, param in model.named_parameters():
    param.requires_grad = ("lora" in name)

# 5. 加载待遗忘数据集
forget_dataset = load_from_disk("./dataset_cache/unlearn_dataset_"+dataset_select+ "_forget")
forget_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

retain_dataset = load_from_disk("./dataset_cache/unlearn_dataset_"+dataset_select+"_retain")


data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

def compute_min_k_ppl_acc(selected_log_probs, mask, k, predicts_mask):
    average_log_probs = []
    average_accs = []
    for sample_log_probs, sample_mask, sample_predicts_mask in zip(
        selected_log_probs, mask, predicts_mask
    ):
        sample_log_probs_nonpad = sample_log_probs[sample_mask]  # 过滤 padding
        k_value = int(k * sample_log_probs_nonpad.size(0))  # ✅ 修正 `.size()` 为 `.size(0)`
        if k_value > 0:
            topk_results = torch.topk(sample_log_probs_nonpad, k_value, largest=False)
            min_k_log_probs = topk_results.values
            topk_indices = topk_results.indices
            sample_average_log_prob = min_k_log_probs.mean()
            average_log_probs.append(sample_average_log_prob)

            # 🚀 计算 `Top-k` 预测的正确率
            sample_acc = sample_predicts_mask[topk_indices].float().mean()  # ✅ 转换 `bool -> float`
            average_accs.append(sample_acc)

    ppl = torch.exp(-torch.stack(average_log_probs).mean())
    acc = (sum(average_accs) / len(average_accs)) * 100
    return ppl.item(), acc.item()
def compute_accuracy(model, dataset, batch_size=1):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    model.eval()

    selected_log_probs_list, mask_list, predicts_mask_list = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔄 计算 Accuracy 进度", unit="batch"):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits

            labels = batch["labels"][:, 1:]  # 🚀 去掉起始 token
            pad_token_mask = labels != -100  # 🚀 计算非 padding 部分
            log_probs = torch.log_softmax(logits, dim=-1)[:, :-1]  # 🚀 计算 `log_probs`

            # 🚀 获取 `input_ids` 对应的 log_probs
            input_ids_expanded = labels.unsqueeze(-1)
            input_ids_expanded[input_ids_expanded == -100] = 0
            selected_log_probs = log_probs.gather(2, input_ids_expanded).squeeze(-1) * pad_token_mask

            pred = logits.argmax(dim=-1)[:, :-1]  # 🚀 计算 `argmax` 预测值
            predicts_mask = pred == labels  # 🚀 预测正确性 mask

            selected_log_probs_list.append(selected_log_probs)
            mask_list.append(pad_token_mask)
            predicts_mask_list.append(predicts_mask)

    # 🚀 **合并所有 batch 计算 `min-k` PPL & Accuracy**
    selected_log_probs = torch.cat(selected_log_probs_list, dim=0)
    mask = torch.cat(mask_list, dim=0)
    predicts_mask = torch.cat(predicts_mask_list, dim=0)
    ratio = 1
    ppl, accuracy = compute_min_k_ppl_acc(selected_log_probs, mask, ratio, predicts_mask)

    return ppl, accuracy

# 6. 保存初始 LoRA adapter 权重到 CPU
initial_sd = {
    n: p.detach().cpu().clone()
    for n, p in model.named_parameters() if "lora" in n
}

ppl_forget, acc_forget = compute_accuracy(model, forget_dataset) # vanilla

print(f"🚀 Forget Dataset PPL: {ppl_forget}")
print(f"🚀 Forget Dataset Acc: {acc_forget}")