# 🚩 模块0：环境与依赖准备
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, load_from_disk, concatenate_datasets
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import os


# =======================
# 模块1：数据加载与预处理
# =======================
def preprocess_function(example):
    text = example["text"]
    if "### Response:" in text:
        parts = text.split("### Response:")
        prompt = parts[0].strip()
        response = parts[1].strip() if len(parts) > 1 else ""
    else:
        prompt = text
        response = ""
    return {"prompt": prompt, "response": response}

retain_dataset = load_dataset("llmunlearn/unlearn_dataset", name="arxiv", split="retain", cache_dir="./data").map(preprocess_function)
approx_dataset = load_dataset("llmunlearn/unlearn_dataset", name="arxiv", split="approximate", cache_dir="./data").map(preprocess_function)
approx_dataset = approx_dataset.filter(lambda ex: len(ex["text"].strip()) > 5).map(preprocess_function)
forget_dataset = load_dataset("llmunlearn/unlearn_dataset", name="arxiv", split="forget", cache_dir="./data").map(preprocess_function)

# =======================
# 模块2（黑盒版本）：教师模型文本生成接口（Yi-6B）
# =======================
model_name = "Yi-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.padding_side = "left"
teacher_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir="./hf_cache"
).eval()

def get_teacher_responses(dataset, batch_size=4, max_new_tokens=100):
    responses = []
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    for batch in tqdm(dataloader, desc="Generating Teacher Responses (device_map=auto safe)"):
        texts = [t for t in batch["text"] if len(t.strip()) > 5]
        if not texts:
            continue

        # ✅ 不要 .to(device)，保持原始CPU张量
        inputs = tokenizer.batch_encode_plus(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            output_ids = teacher_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens
            )

        for i, generated in enumerate(output_ids):
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
            responses.append({"text": texts[i], "response": decoded})

    return responses



if os.path.exists("retain_supervision.json"):
    with open("retain_supervision.json", "r", encoding="utf-8") as f:
        retain_supervision = json.load(f)
else:
    retain_supervision = get_teacher_responses(retain_dataset, batch_size=16)
    with open("retain_supervision.json", "w", encoding="utf-8") as f:
        json.dump(retain_supervision, f, ensure_ascii=False, indent=2)

if os.path.exists("approx_supervision.json"):
    with open("approx_supervision.json", "r", encoding="utf-8") as f:
        approx_supervision = json.load(f)
else:
    approx_supervision = get_teacher_responses(approx_dataset, batch_size=4)
    with open("approx_supervision.json", "w", encoding="utf-8") as f:
        json.dump(approx_supervision, f, ensure_ascii=False, indent=2)

# =======================
# 模块3：学生模型初始化（tinyllama + LoRA）
# =======================
student_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir="./hf_cache"
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

student_model = prepare_model_for_kbit_training(student_model)
student_model = get_peft_model(student_model, lora_config)

# =======================
# 模块4：黑盒文本监督蒸馏训练
# =======================
from transformers import DataCollatorForLanguageModeling

def build_training_dataset(examples):
    full_texts = []

    for e in examples:
        text = e["text"]
        response = e["response"]

        if "### Response:" in text:
            prompt = text.split("### Response:")[0].strip()
        else:
            prompt = text.strip()

        full_text = prompt + response
        if len(full_text.strip()) < 5:
            continue  # 🔍 跳过无效文本

        full_texts.append(full_text)

    encodings = tokenizer(
        full_texts,
        truncation=True,
        padding="max_length",
        max_length=512
    )

    dataset_dicts = []
    for input_ids, attention_mask in zip(encodings["input_ids"], encodings["attention_mask"]):
        labels = [
            token_id if mask == 1 else -100  # 🔥 忽略 padding 部分
            for token_id, mask in zip(input_ids, attention_mask)
        ]
        dataset_dicts.append({
            "input_ids": input_ids,
            "labels": labels
        })

    return dataset_dicts



from datasets import Dataset
retain_tokenized = Dataset.from_list(build_training_dataset(retain_supervision))
approx_tokenized = Dataset.from_list(build_training_dataset(approx_supervision))

combined_dataset = concatenate_datasets([retain_tokenized, approx_tokenized])

training_args = TrainingArguments(
    output_dir="./blackbox_student_model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=50,
    fp16=True,
    save_steps=500,
    save_total_limit=2,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=student_model,
    args=training_args,
    train_dataset=combined_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# =======================
# 模块5 和 模块6 可按需扩展（例如用 CosSim 惩罚 forget 数据 或 用 MIA 验证）
# 此处不再使用 logits，保持真实黑盒场景
