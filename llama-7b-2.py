import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# **1. 加载 Llama2 7B**
model_name = "NousResearch/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# **2. 配置 LoRA 进行微调**
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# **3. 选择 Unlearning 数据**
dataset = load_dataset("squad_v2", split="train[:1%]", cache_dir="./dataset_cache")

def preprocess_function(examples):
    prompt = f"Context: {examples['context']} Question: {examples['question']} Answer:"
    tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenizer(examples["answers"]["text"][0] if examples["answers"]["text"] else "UNKNOWN",
                                    truncation=True, padding="max_length", max_length=512)["input_ids"]
    return tokenized

dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)

# **4. 确保 LoRA 层可训练**
model = get_peft_model(model, lora_config)
model = prepare_model_for_kbit_training(model)
for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# **5. 配置训练参数**
training_args = TrainingArguments(
    output_dir="./lora_unlearning",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="adamw_torch",
    num_train_epochs=20,
    save_steps=100,
    logging_steps=50,
    learning_rate=5e-5,
    fp16=True,
    do_train=True,
    report_to="none"
)

# **6. 训练 Unlearning**
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = torch.nn.CrossEntropyLoss()(outputs.logits.view(-1, outputs.logits.size(-1)), inputs["input_ids"].view(-1))
        loss.backward(retain_graph=True)
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

trainer.train()

# **7. 测试 Unlearning 结果**
def generate_response(model, question, context):
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    output = model.generate(**inputs, max_length=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# **8. 保存 Unlearning 过的模型**
unlearned_model_path = "./lora_unlearned_model"
model.save_pretrained(unlearned_model_path)
tokenizer.save_pretrained(unlearned_model_path)

print(f"Unlearned model saved at: {unlearned_model_path}")

for i in range(5):
    question = dataset[i]["question"]
    context = dataset[i]["context"]
    response = generate_response(model, question, context)
    print(f"Question: {question}")
    print(f"Generated Answer: {response}")

