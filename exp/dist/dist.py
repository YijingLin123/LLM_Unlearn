import matplotlib.pyplot as plt
import numpy as np

# 数据
x = [2, 4, 6, 8, 10, 12]

arxiv_ga = [float(v) for v in ['0.5363', '0.6145', '0.6712', '0.7287', '0.7892', '0.8541']]
arxiv_ar = [float(v) for v in ['0.7768', '0.8664', '0.8887', '0.9859', '1.0679', '1.1066']]
arxiv_ft = [float(v) for v in ['0.5112', '0.6024', '0.6811', '0.8476', '0.9596', '1.0844']]
arxiv_ua = [float(v) for v in ['0.5033', '0.6201', '0.7257', '0.8354', '0.9471', '1.0779']]
arxiv_gad = [float(v) for v in ['0.5408', '0.6951', '1.0127', '0.9469', '1.2487', '1.3455']]

github_ga = [float(v) for v in ['0.1205', '0.1488', '0.2574', '0.4052', '0.5655', '0.7308']]
github_ar = [float(v) for v in ['0.4017', '0.5235', '0.6437', '0.7506', '0.8540', '0.9465']]
github_ft = [float(v) for v in ['0.6974', '0.9175', '1.0762', '1.2155', '1.3871', '1.5781']]
github_ua = [float(v) for v in ['0.6864', '0.8946', '1.1038', '1.3162', '1.5324', '1.7528']]
github_gad = [float(v) for v in ['0.6575', '0.7996', '0.9241', '1.0286', '1.1212', '1.2090']]

# 绘制 arxiv 折线图
plt.figure(figsize=(8, 5))
plt.plot(x, arxiv_ga, marker='o', label='GA')
plt.plot(x, arxiv_ar, marker='s', label='AR')
plt.plot(x, arxiv_ft, marker='^', label='FT')
plt.plot(x, arxiv_ua, marker='v', label='UA')
plt.plot(x, arxiv_gad, marker='*', label='GAD')
plt.xlabel("Checkpoint Interval (k)")
plt.ylabel(r'$\left\|| W_t - W_{t^\prime} \right\||_2$')
plt.title("LoRA Distance on Arxiv")
plt.ylim(0.4, 1.6)
plt.yticks(np.arange(0.4, 1.6, 0.2))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 绘制 github 折线图
plt.figure(figsize=(8, 5))
plt.plot(x, github_ga, marker='o', label='GA')
plt.plot(x, github_ar, marker='s', label='AR')
plt.plot(x, github_ft, marker='^', label='FT')
plt.plot(x, github_ua, marker='v', label='UA')
plt.plot(x, github_gad, marker='*', label='GAD')
plt.xlabel("Checkpoint Interval (k)")
plt.ylabel(r'$\left\|| W_t - W_{t^\prime} \right\||_2$')
plt.title("LoRA Distance on GitHub")
plt.ylim(0, 1.8)
plt.yticks(np.arange(0, 1.8, 0.3))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()