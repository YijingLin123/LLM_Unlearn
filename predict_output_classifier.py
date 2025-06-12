import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys

MODEL_PATH = "./fatpatch_output_classifier_arxiv"  # 你保存的模型目录
MAX_LENGTH = 512

# ✅ 加载模型和 tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

# ✅ 推理函数
def classify_text(text, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()
        label = int(probs[1] > threshold)
    return label, probs[1].item()

if __name__ == "__main__":
    text = input("请输入你想判断的文本：\n> ")
    label, score = classify_text(text)
    print(f"\n🔍 是否命中 Forget 区域: {'✅ 是' if label else '❌ 否'}（得分: {score:.4f}）")
