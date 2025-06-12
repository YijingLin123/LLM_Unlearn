import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

# ✅ 超参
model_name = "roberta-base"
tofu_subset_name = "forget10"  # 可改为 forget1, forget5, forget50 等
threshold = 0.99
learning_rate = 2e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ 加载 TOFU 数据集
print("📥 加载 TOFU 数据...")
forget = load_dataset("locuslab/TOFU", tofu_subset_name, split="train")
retain = load_dataset("locuslab/TOFU", "retain90", split="train")

# ✅ 保留 question 字段并打标签
# 将这个改为answer
forget = forget.rename_column("question", "text").remove_columns("answer")
retain = retain.rename_column("question", "text").remove_columns("answer")
forget = forget.map(lambda x: {"label": 1})
retain = retain.map(lambda x: {"label": 0})

dataset = forget.train_test_split(test_size=0.1, seed=42)
retain_split = retain.train_test_split(test_size=0.1, seed=42)
dataset["train"] = concatenate_datasets([dataset["train"], retain_split["train"]])
dataset["test"] = concatenate_datasets([dataset["test"], retain_split["test"]])

# ✅ 加权交叉熵（关键）
num_class_0 = dataset["train"].filter(lambda x: x["label"] == 0).num_rows
num_class_1 = dataset["train"].filter(lambda x: x["label"] == 1).num_rows
class_weights = torch.tensor([
    (num_class_0 + num_class_1) / num_class_0,
    (num_class_0 + num_class_1) / num_class_1
]).float().to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# ✅ Tokenization
def tokenize_fn(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)

# ✅ compute_metrics（对齐 repo 中标准）
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()
    preds = np.where(probs[:, 1] > threshold, 1, 0)
    acc = (preds == labels).mean()
    errors = np.abs(preds - labels).sum()
    return {"acc": acc, "errors": errors}

# ✅ 自定义 Trainer 注入 loss_fn
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ✅ 模型和训练参数
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, device_map="auto")

training_args = TrainingArguments(
    output_dir="./tofu_classifier_output",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=30,
    learning_rate=learning_rate,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="errors",
    greater_is_better=False,
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer),
)

print("🚀 开始训练 TOFU classifier")
trainer.train()
trainer.save_model("./tofu_classifier_output")
tokenizer.save_pretrained("./tofu_classifier_output")
