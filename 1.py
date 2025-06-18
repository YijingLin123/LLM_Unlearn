import numpy as np
import os

Original = list(map(float, ['0.2540', '0.3604', '0.4440', '0.5147', '0.5749', '0.6260']))

Reproduce = list(map(float, ['0.5363', '0.6145', '0.6712', '0.7287', '0.7892', '0.8541']))

Malicious = list(map(float, ['0.5408', '0.6951', '1.0127', '0.9469', '1.2487', '1.3455']))

r = np.corrcoef(Original, Reproduce)[0, 1]
r1 = np.corrcoef(Original, Malicious)[0, 1]

# 加载索引与数据
# unlearn_method = "GA"
# unlearn_method = "AR"
# unlearn_method = "FT"
# unlearn_method = "UA"
unlearn_method = "GAD"

# dataset_select = "arxiv"
dataset_select = "github"

ckpt_dir = unlearn_method+"_"+dataset_select+"_"+"pou_proof/checkpoint-582"

size_target = os.path.getsize(f"{ckpt_dir}/adapter_model.safetensors") / (1024 * 1024)

print(f"📁 W_t' file size: {size_target:.2f} MB")


print(r)
print(r1)


