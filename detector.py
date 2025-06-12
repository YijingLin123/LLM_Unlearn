import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# ✅ 配置参数
DATASET_NAME = "arxiv"
MODEL_NAME = "roberta-base"
OUTPUT_DIR = f"./fatpatch_output_classifier_{DATASET_NAME}"
EPOCHS = 30
BATCH_SIZE = 16
MAX_LENGTH = 512
CACHE_DIR = "./dataset_cache"

def download_and_prepare(dataset_name="arxiv"):
    print(f"📥 下载并处理数据集：{dataset_name}")

    def preprocess(example):
        return {
            "text": example["text"]
        }

    print("🔹 加载 forget 数据...")
    # forget_dataset = load_dataset("llmunlearn/unlearn_dataset", name="arxiv", split="forget",
    #                               cache_dir="./dataset_cache")
    # print(forget_dataset[0])
    forget = load_dataset(
        "llmunlearn/unlearn_dataset", name=dataset_name, split="forget", cache_dir=CACHE_DIR
    ).map(preprocess)
    forget = forget.map(lambda x: {"label": 1})

    print("🔹 加载 retain 数据...")
    retain = load_dataset(
        "llmunlearn/unlearn_dataset", name=dataset_name, split="retain", cache_dir=CACHE_DIR
    ).map(preprocess)
    retain = retain.map(lambda x: {"label": 0})

    print("📦 拼接并打乱...")
    full_dataset = concatenate_datasets([forget, retain]).shuffle(seed=42)
    return full_dataset

def tokenize_fn(example, tokenizer):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ 初始化模型 & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

    # ✅ 加载数据并编码
    dataset = download_and_prepare(DATASET_NAME)
    tokenized_dataset = dataset.map(lambda x: tokenize_fn(x, tokenizer), batched=True)

    # ✅ 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=2e-5,
        save_strategy="epoch",
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        report_to="none",
    )

    # Lobo A, Hansen JK, Hansen LN, Kjær ED. Differences among six woody perennials native to Northern Europe in their level of genetic differentiation and adaptive potential at fine local scale

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    print("🚀 开始训练 FatPatch 输出分类器...")
    trainer.train()

    print("💾 保存模型至：", OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
