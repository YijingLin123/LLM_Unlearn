import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import pandas as pd

def generate_response(model, question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding="max_length", max_length=256).to("cuda")

    # 让模型生成最多 50 个新的 token，避免长度限制问题
    output = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(output[0], skip_special_tokens=True)

dataset = load_dataset("squad_v2", split="train[:1%]", cache_dir="./dataset_cache")

# print(dataset[0])
#
# # 选择前 10 条数据进行查看
# sample_data = dataset.select(range(10))
#
# # 转换为 DataFrame 方便查看
# df = pd.DataFrame(sample_data)
#
# # 打印前 10 条数据
# print(df[['question', 'answers']])


# 直接加载已保存的模型
unlearned_model_path = "./lora_unlearned_model"

# 加载原始 Llama2 7B 模型
model_name = "NousResearch/Llama-2-7b-chat-hf"
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
question = "In which decade did Beyonce rise to fame?"

# 生成模型的回答
response = generate_response(base_model, question)

# 打印问题、生成的回答和真实答案
print(f"Question: {question}")
print(f"Generated Answer: {response}")
print("----------------------------")

# 加载 Unlearning 过的 LoRA 适配器
model = PeftModel.from_pretrained(base_model, unlearned_model_path)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(unlearned_model_path)

print("Unlearned model loaded successfully!")



# 直接用 Unlearning 过的模型生成回答
model.eval()
# Question: Who managed the Destiny's Child group?
# Answer: ['Mathew Knowles']
for i in range(1):
    question = dataset[i]["question"]
    question = "In which decade did Beyonce rise to fame?"

    # 生成模型的回答
    response = generate_response(model, question)

    # 打印问题、生成的回答和真实答案
    print(f"Question: {question}")
    print(f"Generated Answer: {response}")
    print(f"Ground Truth: {dataset[i]['answers']['text']}")
    print("=======")