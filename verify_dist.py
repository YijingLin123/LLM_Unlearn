import os
import random
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
    get_scheduler,
    BitsAndBytesConfig
)
from peft import PeftModel,get_peft_model, LoraConfig, prepare_model_for_kbit_training
from datasets import load_from_disk, concatenate_datasets
from safetensors.torch import load_file as load_safetensors
from torch.optim import AdamW
from tqdm import tqdm
import shutil
from torch.nn.functional import softmax
import torch.nn.functional as F
import scipy.stats as stats



# ========== 计算 checkpoint 间欧氏距离 ==========
def compute_checkpoint_distance(step_a, step_b, ckpt_dir):
    sd_a = load_safetensors(f"{ckpt_dir}/checkpoint-{step_a}/adapter_model.safetensors", device="cpu")
    sd_b = load_safetensors(f"{ckpt_dir}/checkpoint-{step_b}/adapter_model.safetensors", device="cpu")
    return sum(torch.norm(sd_a[k] - sd_b[k]).item() for k in sd_a)

# def generate_adversarial_labels(model, batch, rm_groundtruth=True):
#     model.eval()
#     inputs = {k: v.to(model.device) for k, v in batch.items()}
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits  # (batch, seq_len, vocab_size)
#
#     if rm_groundtruth:
#         # 将原始 token 的概率置为 -inf，避免选中自己
#         input_ids = inputs["input_ids"]
#         for i in range(input_ids.size(0)):
#             for j in range(input_ids.size(1)):
#                 logits[i, j, input_ids[i, j]] = float("-inf")
#
#     # softmax 概率
#     probs = softmax(logits, dim=-1)
#
#     # 取每个位置预测概率最高的（非 ground truth）token
#     adversarial_labels = probs.argmax(dim=-1)
#
#     return adversarial_labels  # shape: (batch, seq_len)

def generate_adversarial_labels(model, inputs, rm_groundtruth=True, perturb_ratio=0.3):
    model.eval()
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    if rm_groundtruth:
        input_ids = inputs["input_ids"]
        mask = torch.nn.functional.one_hot(input_ids, num_classes=logits.size(-1)).bool()
        logits = logits.masked_fill(mask, -1e9)

    probs = softmax(logits, dim=-1)
    top2 = probs.topk(2, dim=-1).indices  # (batch, seq_len, 2)

    # 默认选第2高作为扰动目标
    alt_labels = top2[:, :, 1]
    adv_labels = inputs["input_ids"].clone()

    # 随机替换其中一部分 token
    rand_mask = torch.rand_like(adv_labels.float()) < perturb_ratio
    adv_labels[rand_mask] = alt_labels[rand_mask]

    # 打印 diff ratio
    diff_ratio = (adv_labels != inputs["input_ids"]).float().mean().item()
    # print(f"Adversarial diff ratio: {diff_ratio:.4f}")

    return adv_labels

def verify_reverse_step(dataset_select, unlearn_method, step_i_path, step_i1_path, dataset_slice, grad_accum=10):
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    model_name = "Yi-6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    # base = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     quantization_config=BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_use_double_quant=True
    #     ),
    #     device_map={"": "cuda:0"},
    #     cache_dir="./hf_cache"
    # ).eval()
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="./hf_cache"
    ).eval()

    lora_config = LoraConfig(
        r=8, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    # model = PeftModel.from_pretrained(
    #     base, step_i_path, peft_config=peft_cfg, device_map={"": "cuda:0"}
    # )
    model = PeftModel.from_pretrained(
        base,
        step_i_path,
        peft_config=lora_config,
        is_trainable=True,  # 关键
    )
    # model = get_peft_model(model, lora_config)
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_disable()
    except AttributeError:
        pass
    model.train()

    for n, p in model.named_parameters():
        p.requires_grad = ("lora" in n)

    dataloader = DataLoader(
        dataset_slice,
        batch_size=1,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer, return_tensors="pt"),
        drop_last=True
    )

    optimizer = AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=2e-5)
    total_updates = len(dataloader) // grad_accum
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_updates)

    scaler = GradScaler()
    model.zero_grad()

    if dataset_select == "arxiv":
        if unlearn_method == "GA" or unlearn_method ==  "FT" or unlearn_method ==  "UA":
            desc = f"Verifying {step_i_path[-2:]}→{step_i1_path[-2:]}"
        elif unlearn_method == "AR" or unlearn_method == "GAD":
            desc = f"Verifying {step_i_path[-3:]}→{step_i1_path[-3:]}"
    elif dataset_select == "github":
        desc = f"Verifying {step_i_path[-3:]}→{step_i1_path[-3:]}"
    for step, batch in enumerate(tqdm(dataloader, desc=desc)):
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        if batch["input_ids"].dim() != 2 or batch["input_ids"].size(1) > 2048:
            print(f"⚠️ Skip step {step} due to illegal shape: {batch['input_ids'].shape}")
            continue
        with autocast():
            if unlearn_method == "GA":
                loss = -model(**batch).loss / grad_accum
            elif unlearn_method == "AR":
                outputs = model(**batch)
                logits = outputs.logits.view(-1, outputs.logits.size(-1))
                labels = batch["input_ids"].view(-1)
                loss = torch.nn.CrossEntropyLoss()(logits, labels) / grad_accum
            elif unlearn_method == "FT":
                random_labels = torch.randint(
                    low=0,
                    high=model.config.vocab_size,
                    size=batch["input_ids"].shape,
                    dtype=torch.long,
                    device=batch["input_ids"].device
                )
                batch["labels"] = random_labels
                outputs = model(**batch)
                logits = outputs.logits.view(-1, outputs.logits.size(-1))
                labels = batch["labels"].view(-1)
                loss = torch.nn.CrossEntropyLoss()(logits, labels) / grad_accum
            elif unlearn_method == "GAD":
                factors = batch.pop("factor")  # shape: (batch_size,)
                outputs = model(**batch)
                logits = outputs.logits
                labels = batch["labels"]

                # 语言模型 shift：自动对齐输入输出
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                loss = loss.view(shift_logits.size(0), -1)  # shape: (batch, seq_len)

                # mask 掉 label=-100 的 pad/token 部分
                valid_counts = (shift_labels != -100).sum(dim=-1).float()  # shape: (batch,)
                loss = loss.sum(dim=-1) / valid_counts  # shape: (batch,)

                # 使用 ascent/descent 的 factor 加权，每个样本乘一个 ±1.0
                loss = (loss * factors).mean() / grad_accum
        if unlearn_method == "UA":
            adv_labels = generate_adversarial_labels(model, batch)
            batch["labels"] = adv_labels
            outputs = model(**batch)
            logits = outputs.logits.view(-1, outputs.logits.size(-1))
            labels = batch["labels"].view(-1)
            loss = torch.nn.CrossEntropyLoss()(logits, labels)/ grad_accum


        scaler.scale(loss).backward()
        if (step + 1) % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

    tmp_dir = "tmp_verify"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    model.save_pretrained("tmp_verify", safe_serialization=False)
    sd_new = torch.load("tmp_verify/adapter_model.bin", map_location="cpu")
    sd_target = load_safetensors(f"{step_i1_path}/adapter_model.safetensors", device="cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    d_1 = sum(torch.norm(sd_new[k] - sd_target[k], p=1).item() for k in sd_new)
    d_2 = sum(torch.norm(sd_new[k] - sd_target[k]).item() for k in sd_new)
    # 将所有参数拼接成一个向量（注意要 flatten 再 concat）
    vec_new = torch.cat([v.view(-1) for v in sd_new.values()])
    vec_target = torch.cat([v.view(-1) for v in sd_target.values()])
    d_cos = 1 - F.cosine_similarity(vec_new.unsqueeze(0), vec_target.unsqueeze(0)).item()
    d_inf =max(torch.norm(sd_new[k] - sd_target[k], p=float('inf')).item() for k in sd_new)
    return d_1, d_2, d_cos, d_inf


# ========== 主程序 ==========
def main():
    # 固定随机性
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)

    # 加载索引与数据
    # unlearn_method = "GA"
    # unlearn_method = "AR"
    # unlearn_method = "FT"
    # unlearn_method = "UA"
    unlearn_method = "GAD"

    # dataset_select = "arxiv"
    dataset_select = "github"

    ckpt_dir = unlearn_method+"_"+dataset_select+"_"+"pou_proof"

    if dataset_select == "arxiv":
        if unlearn_method == "GA" or unlearn_method == "FT" or unlearn_method == "UA":
            base_step = 32
            target_step = 50
            # 趋势验证（32 → 多个 k）
            target_steps = [34, 36, 38, 40, 42, 44]
            forget_dataset = load_from_disk("./dataset_cache/unlearn_dataset_arxiv_forget")
            indices = list(range(len(forget_dataset)))
            full_ds = load_from_disk("./dataset_cache/unlearn_dataset_arxiv_forget")
        elif unlearn_method == "AR":
            base_step = 598
            target_step = 614
            target_steps = [600, 602, 604, 606, 608, 610]
            approximate_dataset = load_from_disk("./dataset_cache/unlearn_dataset_arxiv_approximate")
            indices = list(range(len(approximate_dataset)))
            full_ds = load_from_disk("./dataset_cache/unlearn_dataset_arxiv_approximate")
        elif unlearn_method == "GAD":
            base_step = 232
            target_step = 250
            target_steps = [234, 236, 238, 240, 242, 244]
            forget_dataset = load_from_disk("./dataset_cache/unlearn_dataset_arxiv_forget")  # 被遗忘数据
            descent_dataset = load_from_disk("./dataset_cache/unlearn_dataset_arxiv_retain")  # 保留数据（梯度下降）

            # 添加 factor 字段：上升为 -1，下降为 +1
            forget_dataset = forget_dataset.map(lambda x: {**x, "factor": -1.0})
            descent_dataset = descent_dataset.map(lambda x: {**x, "factor": 1.0})

            # 合并两个 dataset
            combined_dataset = concatenate_datasets([forget_dataset, descent_dataset])
            indices = list(range(len(combined_dataset)))
            full_ds = combined_dataset
    elif dataset_select == "github":
        if unlearn_method == "GA" or unlearn_method == "FT" or  unlearn_method == "UA":
            base_step = 182
            target_step = 200
            # 趋势验证（32 → 多个 k）
            target_steps = [184, 186, 188, 190, 192, 194]
            forget_dataset = load_from_disk("./dataset_cache/unlearn_dataset_github_forget")
            indices = list(range(len(forget_dataset)))
            full_ds = load_from_disk("./dataset_cache/unlearn_dataset_github_forget")
        elif unlearn_method == "AR":
            base_step = 510
            target_step = 527
            target_steps = [512, 514, 516, 518, 520, 522]
            approximate_dataset = load_from_disk("./dataset_cache/unlearn_dataset_github_approximate")
            indices = list(range(len(approximate_dataset)))
            full_ds = load_from_disk("./dataset_cache/unlearn_dataset_github_approximate")
        elif unlearn_method == "GAD":
            base_step = 582
            target_step = 600
            target_steps = [584, 586, 588, 590, 592, 594]
            forget_dataset = load_from_disk("./dataset_cache/unlearn_dataset_github_forget")  # 被遗忘数据
            descent_dataset = load_from_disk("./dataset_cache/unlearn_dataset_github_retain")  # 保留数据（梯度下降）

            # 添加 factor 字段：上升为 -1，下降为 +1
            forget_dataset = forget_dataset.map(lambda x: {**x, "factor": -1.0})
            descent_dataset = descent_dataset.map(lambda x: {**x, "factor": 1.0})

            # 合并两个 dataset
            combined_dataset = concatenate_datasets([forget_dataset, descent_dataset])
            indices = list(range(len(combined_dataset)))
            full_ds = combined_dataset

    grad_accum = 10
    if unlearn_method == "GAD":
        full_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels", "factor"])
    else:
        full_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    ds_slice = full_ds.select(indices[base_step * grad_accum: target_step * grad_accum])
    d_1, d_2, d_cos, d_inf = verify_reverse_step(
        dataset_select,
        unlearn_method,
        f"{ckpt_dir}/checkpoint-{base_step}",
        f"{ckpt_dir}/checkpoint-{target_step}",
        ds_slice,
        grad_accum=grad_accum
    )
    print(f"\n🔁 复现 {base_step}→{target_step} 后的 LoRA 距离: ", d_1, d_2, d_cos, d_inf)

    orig_distances = [compute_checkpoint_distance(base_step, k, ckpt_dir) for k in target_steps]
    repro_distances_d1 = []
    repro_distances_d2 = []
    repro_distances_dcos = []
    repro_distances_dinf = []
    for k in target_steps:
        ds_k = full_ds.select(indices[base_step * grad_accum: k * grad_accum])
        if unlearn_method == "GAD":
            ds_k.set_format("torch", columns=["input_ids", "attention_mask", "labels", "factor"])
        else:
            ds_k.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        d_1, d_2, d_cos, d_inf = verify_reverse_step(
            dataset_select,
            unlearn_method,
            f"{ckpt_dir}/checkpoint-{base_step}",
            f"{ckpt_dir}/checkpoint-{k}",
            ds_k,
            grad_accum=grad_accum,
        )
        repro_distances_d1.append(d_1)
        repro_distances_d2.append(d_2)
        repro_distances_dcos.append(d_cos)
        repro_distances_dinf.append(d_inf)
        print(f"\n🔁 复现 {base_step}→{k} 梯度上升后的 LoRA 距离: ",d_1, d_2, d_cos, d_inf)

        # malicious repro dist
        # if len(repro_distances) != 0:
        #     repro_distances.append(repro_distances[0])
        # else:
        #  repro_distances.append(d)

    r1 = np.corrcoef(orig_distances, repro_distances_d1)[0, 1]
    r2 = np.corrcoef(orig_distances, repro_distances_d2)[0, 1]
    print("\n📈 [Trend Verification]")
    print("Target steps            :", target_steps)
    print("Original distances      :", [f"{d:.4f}" for d in orig_distances])
    print("Reproduced distances  d1  :", [f"{d:.4f}" for d in repro_distances_d1])
    print("Reproduced distances  d2  :", [f"{d:.4f}" for d in repro_distances_d2])
    # print("Reproduced distances  dcos  :", [f"{d:.4f}" for d in repro_distances_dcos])
    # print("Reproduced distances  dinf  :", [f"{d:.4f}" for d in repro_distances_dinf])
    print(f"📊 Pearson correlation r: {r1:.4f}")
    print(f"📊 Pearson correlation r: {r2:.4f}")

    r1, p1 = stats.pearsonr(orig_distances, repro_distances_d1)
    r2, p2 = stats.pearsonr(orig_distances, repro_distances_d2)
    print(r1,p1)
    print(r2, p2)

    # if r > 0.9:
    #     print("✅ 趋势一致性良好（高相关性）")
    # elif r > 0.5:
    #     print("🟡 有一定趋势一致性，但存在偏差")
    # else:
    #     print("❌ 趋势未重现，请检查复现流程")


if __name__ == "__main__":
    main()

