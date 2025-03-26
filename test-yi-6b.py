import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, load_from_disk
import math

print(math.exp(1.9884018898010254))
print(math.exp(2.2239267826080322))

def generate_response(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="./hf_cache"
    ).eval()

    inputs = tokenizer("What is the capital of france?", return_tensors="pt")
    outputs = model.generate(inputs.input_ids.cuda(), max_new_tokens=30)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# **加载Yi-6B**
model_name = "Yi-6B"
generate_response(model_name)

unlearned_model_path = "./lora_unlearned_model"
generate_response(unlearned_model_path)
