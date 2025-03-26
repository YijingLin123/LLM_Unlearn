import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# **1. 加载 Llama2 7B**
model_name = "NousResearch/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 量化模型，减少显存占用（8-bit 量化）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    # load_in_8bit=True  # 使用 8-bit 量化减少显存
)

# **2. 配置 LoRA 进行微调**
lora_config = LoraConfig(
    r=8,  # 低秩子空间大小
    lora_alpha=32,  # LoRA 缩放因子
    target_modules=["q_proj", "v_proj"],  # 仅调整 Q-V 注意力层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"  # 适用于 LLM
)

# 添加 LoRA 适配层
model = get_peft_model(model, lora_config)
model = prepare_model_for_kbit_training(model)  # 适配 8-bit 训练
model.print_trainable_parameters()  # 查看可训练参数

def generate_response(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    output = model.generate(**inputs, max_length=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

before_unlearning = generate_response(model,"What is the capital of France?")
print(f"Before Unlearning: {before_unlearning}")

learned_model_path = "./lora_learned_model"
model.save_pretrained(learned_model_path)
tokenizer.save_pretrained(learned_model_path)

print(f"Learned model saved at: {learned_model_path}")

# **3. 选择 Unlearning 数据**
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]", cache_dir="./dataset_cache", download_mode="reuse_cache_if_exists")  # 仅取 1% 数据
# dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir="./dataset_cache", download_mode="reuse_cache_if_exists").select(range(10))
dataset = dataset.map(lambda x: {"input_ids": tokenizer(x["text"], truncation=True, padding="max_length", max_length=512)["input_ids"]}, batched=True)

# **4. 设定梯度上升（Gradient Ascent）损失函数**
def loss_function(logits, labels):
    """梯度上升损失函数，反向优化目标知识"""
    loss = torch.nn.CrossEntropyLoss()
    return -loss(logits.view(-1, logits.size(-1)), labels.view(-1))  # 负损失 = 梯度上升

# 训练前，确保只训练 LoRA 层
for name, param in model.named_parameters():
    if "lora" not in name:  # 冻结非LoRA参数
        param.requires_grad = False
    if "lora" in name:  # 只解冻 LoRA 参数
        param.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params} || All params: {all_params} || Trainable%: {100 * trainable_params / all_params:.2f}%")

# **5. 配置训练参数**
training_args = TrainingArguments(
    output_dir="./lora_unlearning",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="adamw_torch",
    num_train_epochs=3,
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
        loss = -loss
        loss.requires_grad_(True)
        return (loss, outputs) if return_outputs else loss

model.train()  # 确保模型处于训练模式
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

trainer.train()

# **7. 测试 Unlearning 结果**

after_unlearning = generate_response(model,"What is the capital of France?")

print(f"After Unlearning: {after_unlearning}")  # 可能返回模糊答案或错误答案

unlearned_model_path = "./lora_unlearned_model"
model.save_pretrained(unlearned_model_path)
tokenizer.save_pretrained(unlearned_model_path)

print(f"Unlearned model saved at: {unlearned_model_path}")
