import os
import random
import numpy as np
import torch
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

# unlearn_method = "GA"
unlearn_method = "AR"
# unlearn_method = "FT"
# unlearn_method = "UA"
# unlearn_method = "GAD"

# dataset_select = "arxiv"
dataset_select = "github"

model_name = "Yi-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
# base_model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     device_map="auto",
#     cache_dir="./hf_cache"
# )
# model = base_model.to("cuda:0").eval()

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

approximate_dataset = load_from_disk("./dataset_cache/unlearn_dataset_"+dataset_select+ "_approximate")
approximate_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

# 6. 保存初始 LoRA adapter 权重到 CPU
initial_sd = {
    n: p.detach().cpu().clone()
    for n, p in model.named_parameters() if "lora" in n
}

# 7. 定义 Trainer（禁用 shuffle）
class UnlearningTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = self._prepare_inputs(inputs)
        outputs = model(**inputs)
        loss = -outputs.loss
        return (loss, outputs) if return_outputs else loss

class ApproximateRetrainTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = torch.nn.CrossEntropyLoss()(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            inputs["input_ids"].view(-1)
        )
        return (loss, outputs) if return_outputs else loss

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

def generate_adversarial_labels(model, inputs, rm_groundtruth=True):
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
        adv_labels = generate_adversarial_labels(model, inputs)

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
            batch["factor"] = torch.tensor([f["factor"] for f in features], dtype=torch.float32)
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

# 8. 训练参数（FP16=False）

training_args = TrainingArguments(
    output_dir=unlearn_method+"_"+dataset_select+"_"+"pou_proof",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=10,
    num_train_epochs=1,
    save_steps=2,
    save_total_limit=10,
    logging_steps=10,
    learning_rate=2e-5,
    fp16=True,
    report_to="none"
)

if unlearn_method == "GA":
    trainer = UnlearningTrainer(
        model=model,
        args=training_args,
        train_dataset=forget_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
elif unlearn_method == "AR":
    trainer = ApproximateRetrainTrainer(
        model=model,
        args=training_args,
        train_dataset=approximate_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

elif unlearn_method == "FT":
    trainer = RandomLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=forget_dataset,
        tokenizer=tokenizer
    )
elif unlearn_method == "UA":
    trainer = AdversarialTrainer(
        model=model,
        args=training_args,
        train_dataset=forget_dataset,
        tokenizer=tokenizer
    )
elif unlearn_method == "GAD":
    descent_dataset = load_from_disk("./dataset_cache/unlearn_dataset_"+dataset_select+ "_retain")
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

# 9. 运行梯度上升 Unlearning
trainer.train()

# 10. 提取训练后 LoRA adapter 权重
final_sd = {
    n: p.detach().cpu().clone()
    for n, p in model.named_parameters() if "lora" in n
}

# 11. 计算欧氏距离
distance = sum(
    torch.norm(final_sd[n] - initial_sd[n]).item() for n in initial_sd
)
print(f"🚨 1 epoch 后 LoRA adapter 欧氏距离: {distance:.6f}")

# 12. 保存最终模型和 tokenizer
os.makedirs(unlearn_method+"_"+dataset_select+"_"+"pou_proof/model_final", exist_ok=True)
model.save_pretrained(unlearn_method+"_"+dataset_select+"_"+"pou_proof/model_final")
tokenizer.save_pretrained(unlearn_method+"_"+dataset_select+"_"+"pou_proof/model_final")
print("模型已保存到"+unlearn_method+"_"+dataset_select+"_"+ "pou_proof/model_final")