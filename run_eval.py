import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, load_from_disk
import math
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm import tqdm
import argparse

# **🚀 解析命令行参数**
parser = argparse.ArgumentParser(description="Evaluate Unlearning Methods")
parser.add_argument("--method", type=str, choices=["gradient_ascent", "approximate_retrain", "random_label", "adversarial_sample", "ascent_plus_descent_retain", "ascent_plus_descent_general", "ascent_plus_kl_retain", "ascent_plus_kl_general"], required=True,
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

print(unlearned_model_path)

tokenizer = AutoTokenizer.from_pretrained(unlearned_model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    unlearned_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

forget_dataset = load_from_disk("./dataset_cache/unlearn_dataset_forget")
retain_dataset = load_from_disk("./dataset_cache/unlearn_dataset_retain")

training_args = TrainingArguments(
    output_dir="./lora_unlearning_eval",
    per_device_eval_batch_size=1,
    do_eval=True,
    logging_steps=50,
    fp16=True,
    bf16=False,
    report_to="none"
)

# **6. 运行评估**
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=forget_dataset,
    tokenizer=tokenizer,
)

# 显存有限，逐步计算
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


# **🚀 1. 创建 DataLoader 时，确保返回的是 Tensor**
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
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

loss_forget = trainer.evaluate()
ppl_forget, acc_forget = compute_accuracy(model, forget_dataset)
print(f"🚀 Forget Dataset Loss: {loss_forget}")
print(f"🚀 Forget Dataset PPL: {ppl_forget}")
print(f"🚀 Forget Dataset Acc: {acc_forget}")
# 🚀 Forget Dataset Loss: {'eval_loss': 1.9884018898010254, 'eval_model_preparation_time': 0.0001, 'eval_runtime': 80.6816, 'eval_samples_per_second': 6.197, 'eval_steps_per_second': 6.197}
# 🚀 Forget Dataset PPL: 7.3038530349731445
# 🚀 Forget Dataset Acc: 59.91386032104492

trainer.eval_dataset = retain_dataset
loss_retain = trainer.evaluate()
ppl_retain, acc_retain = compute_accuracy(model, retain_dataset)
print(f"🚀 Retain Dataset Loss: {loss_retain}")
print(f"🚀 Retain Dataset PPL: {ppl_retain}")
print(f"🚀 Retain Dataset Acc: {acc_retain}")
# 🚀 Retain Dataset Loss: {'eval_loss': 2.2239267826080322, 'eval_model_preparation_time': 0.0001, 'eval_runtime': 312.6988, 'eval_samples_per_second': 6.396, 'eval_steps_per_second': 6.396}
# 🚀 Retain Dataset PPL: 9.24355697631836
# 🚀 Retain Dataset Acc: 59.43987274169922