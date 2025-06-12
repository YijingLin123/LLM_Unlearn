import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, load_from_disk, concatenate_datasets
from torch.nn.functional import softmax


parser = argparse.ArgumentParser(description="Run Unlearning methods")
parser.add_argument("--method", type=str, choices=["gradient_ascent", "approximate_retrain", "random_label", "adversarial_sample", "ascent_plus_descent_retain", "ascent_plus_descent_general", "ascent_plus_kl_retain", "ascent_plus_kl_general"], required=True,
                    help="Choose the unlearning method: 'gradient_ascent' or 'approximate_retrain' or 'random_label' or 'adversarial_sample' or 'ascent_plus_descent_retain' or 'ascent_plus_descent_general'")
args = parser.parse_args()

# **加载Yi-6B**
model_name = "Yi-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir="./hf_cache"
).eval()

# inputs = tokenizer("What is the capital of france?", return_tensors="pt")
# outputs = model.generate(inputs.input_ids.cuda(), max_new_tokens=30)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# **配置 LoRA 进行微调**
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# **让 `Yi-6B` 仅训练 LoRA 层**
model = get_peft_model(model, lora_config)
model = prepare_model_for_kbit_training(model)

# 4. 只更新 LoRA 层
for name, param in model.named_parameters():
    param.requires_grad = ("lora" in name)

# for name, param in model.named_parameters():
#     if "lora" in name:
#         param.requires_grad = True  # 只更新 LoRA 层
#     else:
#         param.requires_grad = False

# **Gradient Ascent 训练（遗忘 `forget`数据）**
forget_dataset = load_from_disk("./dataset_cache/unlearn_dataset_arxiv_forget")
# print(forget_dataset[0])
class UnlearningTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = torch.nn.CrossEntropyLoss()(outputs.logits.view(-1, outputs.logits.size(-1)),
                                           inputs["input_ids"].view(-1))

        # 关键步骤：梯度上升（反向最优化 loss，让模型遗忘 `forget` 数据）
        loss = -loss  # 反转 loss，让模型远离 `forget` 数据
        return (loss, outputs) if return_outputs else loss

# **近似重训练（Approximate Retrain）**
approximate_dataset = load_from_disk("./dataset_cache/unlearn_dataset_arxiv_approximate")
class ApproximateRetrainTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = torch.nn.CrossEntropyLoss()(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            inputs["input_ids"].view(-1)
        )

        return (loss, outputs) if return_outputs else loss  # **正常 loss 计算，不反转 loss**

class RandomLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # **🚀 生成与 input_ids 长度相同的随机 token**
        random_labels = torch.randint(
            low=0,
            high=model.config.vocab_size,
            size=inputs["input_ids"].shape,
            dtype=torch.long
        ).to(inputs["input_ids"].device)

        # soft random labels
        # labels = inputs["input_ids"].clone()
        # mask = torch.rand(labels.shape, device=labels.device) < 0.5  # 50% 概率替换
        # labels[mask] = random_labels[mask]
        # inputs["labels"] = labels

        # **替换 labels**
        inputs["labels"] = random_labels

        # 🚀 计算 loss（正常训练方式）
        outputs = model(**inputs)
        loss = torch.nn.CrossEntropyLoss()(outputs.logits.view(-1, outputs.logits.size(-1)),
                                           inputs["labels"].view(-1))
        return (loss, outputs) if return_outputs else loss

def generate_adversarial_labels(model, tokenizer, inputs, rm_groundtruth=True):
    """
    给定模型与输入，生成 adversarial token 替换 labels。
    """
    model.eval()
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (batch, seq_len, vocab_size)

    if rm_groundtruth:
        # 将原始 token 的概率置为 -inf，避免选中自己
        input_ids = inputs["input_ids"]
        for i in range(input_ids.size(0)):
            for j in range(input_ids.size(1)):
                logits[i, j, input_ids[i, j]] = float("-inf")

    # softmax 概率
    probs = softmax(logits, dim=-1)

    # 取每个位置预测概率最高的（非 ground truth）token
    adversarial_labels = probs.argmax(dim=-1)

    return adversarial_labels  # shape: (batch, seq_len)

class AdversarialTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 获取 adversarial labels
        adv_labels = generate_adversarial_labels(model, self.tokenizer, inputs)

        inputs["labels"] = adv_labels  # 替换原始 labels

        outputs = model(**inputs)
        loss = torch.nn.CrossEntropyLoss()(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            inputs["labels"].view(-1)
        )
        return (loss, outputs) if return_outputs else loss

class AscentPlusDescentDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        if "factor" in features[0]:
            batch["factor"] = torch.tensor([f["factor"] for f in features])
        return batch

class AscentPlusDescentTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "factor" not in inputs:
            return super().compute_loss(model, inputs, return_outputs)

        factors = inputs.pop("factor")
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        loss = loss.view(shift_logits.size(0), -1)

        valid_counts = (shift_labels != -100).sum(dim=-1).float()
        loss = loss.sum(dim=-1) / valid_counts

        loss = (loss * factors).mean()
        return (loss, outputs) if return_outputs else loss

class AscentPlusKLTrainer(Trainer):
    def __init__(self, *args, pretrain_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrain_model = pretrain_model.eval()  # frozen pretrained model
        self.label_names = ["input_ids", "attention_mask", "labels", "factor"]  # ✅ 关键修复点

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        factors = inputs.pop("factor")
        device = model.device
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        labels = inputs["labels"]
        shift_labels = labels[..., 1:].contiguous()

        batch_size, seq_len = shift_labels.shape
        vocab_size = shift_logits.size(-1)

        loss = torch.zeros(batch_size, device=device)

        for i in range(batch_size):
            if factors[i] == -1:  # forget -> gradient ascent
                ce_loss = torch.nn.functional.cross_entropy(
                    shift_logits[i], shift_labels[i], ignore_index=-100, reduction="mean"
                )
                loss[i] = -ce_loss
            else:  # retain/general -> KL divergence
                with torch.no_grad():
                    ref_logits = self.pretrain_model(
                        input_ids=inputs["input_ids"][i].unsqueeze(0),
                        attention_mask=inputs["attention_mask"][i].unsqueeze(0)
                    ).logits[..., :-1, :].squeeze(0).contiguous()

                ref_prob = torch.softmax(ref_logits, dim=-1)
                cur_log_prob = torch.log_softmax(shift_logits[i], dim=-1)
                kl = -(ref_prob * cur_log_prob).sum(-1)  # forward KL
                mask = shift_labels[i] != -100
                loss[i] = kl[mask].mean()

        final_loss = loss.mean()
        return (final_loss, outputs) if return_outputs else final_loss

# **训练参数**
training_args = TrainingArguments(
    output_dir=f"./lora_{args.method}",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=85,
    num_train_epochs=1,  # 适当减少 epoch，防止过度遗忘
    save_steps=100,
    logging_steps=50,
    learning_rate=2e-5,
    fp16=True,
    do_train=True,
    report_to="none"
)

# **选择训练方法**
if args.method == "gradient_ascent":
    print("🚀 运行梯度上升（Gradient Ascent）...")
    trainer = UnlearningTrainer(
        model=model,
        args=training_args,
        train_dataset=forget_dataset,
        tokenizer=tokenizer
    )
elif args.method == "approximate_retrain":
    print("🚀 运行近似重训练（Approximate Retrain）...")
    trainer = ApproximateRetrainTrainer(
        model=model,
        args=training_args,
        train_dataset=approximate_dataset,  # 🚀 用 approximate set 进行训练
        tokenizer=tokenizer
    )
elif args.method == "random_label":
    print("🚀 运行随机标签微调（Finetune with Random Labels）...")
    trainer = RandomLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=forget_dataset,
        tokenizer=tokenizer
    )
elif args.method == "adversarial_sample":
    print("🚀 运行错误标签微调（Unlearning with Adversarial sample）...")
    trainer = AdversarialTrainer(
        model=model,
        args=training_args,
        train_dataset=forget_dataset,
        tokenizer=tokenizer
    )
elif args.method == "ascent_plus_descent_retain" :
    print("🚀 运行梯度上升 + 下行微调（Gradient Ascent + Descent）...")
    if args.method == "ascent_plus_descent_retain":
        descent_dataset = load_from_disk("./dataset_cache/unlearn_dataset_retain")
    else:
        descent_dataset = load_from_disk("./dataset_cache/unlearn_dataset_general")
        pass
    forget_dataset = forget_dataset.map(lambda x: {**x, "factor": -1.0})
    descent_dataset = descent_dataset.map(lambda x: {**x, "factor": 1.0})
    combined_dataset = concatenate_datasets([forget_dataset, descent_dataset])
    data_collator = AscentPlusDescentDataCollator(tokenizer=tokenizer)
    trainer = AscentPlusDescentTrainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
elif args.method == "ascent_plus_kl_retain":
    print("🚀 运行梯度上升 + KL约束（Gradient Ascent + KL Divergence）...")
    if args.method == "ascent_plus_kl_retain":
        kl_dataset = load_from_disk("./dataset_cache/unlearn_dataset_retain")
    else:
        kl_dataset = load_from_disk("./dataset_cache/unlearn_dataset_general")

    forget_dataset = forget_dataset.map(lambda x: {**x, "factor": -1.0})
    kl_dataset = kl_dataset.map(lambda x: {**x, "factor": 1.0})
    combined_dataset = concatenate_datasets([forget_dataset, kl_dataset])
    data_collator = AscentPlusDescentDataCollator(tokenizer=tokenizer)  # 复用已有 collator

    pretrain_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="./hf_cache"
    ).eval()  # 🚨 不加 .eval() 会引入dropout等干扰

    trainer = AscentPlusKLTrainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        pretrain_model=pretrain_model  # 传入 frozen 模型用于 KL
    )

trainer.train()

# **保存 Unlearning 过的模型**
unlearned_model_path = f"./lora_{args.method}_model"
model.save_pretrained(unlearned_model_path)
tokenizer.save_pretrained(unlearned_model_path)
print(f"🚀 Unlearning 训练完成，模型已保存到: {unlearned_model_path}")




