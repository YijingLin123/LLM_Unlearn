import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams.update({
    'font.size': 14,            # 所有字体大小（坐标轴标签、刻度、legend 等）
    'axes.labelsize': 14,       # 坐标轴标签
    'xtick.labelsize': 14,      # x轴刻度
    'ytick.labelsize': 14,      # y轴刻度
    'legend.fontsize': 14,      # 图例文字
})
# 数据
x = [2, 4, 6, 8, 10, 12]

arxiv_ga_d1 = [float(v) for v in ['66.9507', '80.7067', '89.9316', '97.1509', '104.8357', '112.9829']]
arxiv_ar_d1 = [float(v) for v in ['79.4521', '99.0135', '124.8265', '123.8527', '141.1513', '140.7518']]
arxiv_ft_d1 = [float(v) for v in ['58.8068', '74.6219', '91.2868', '99.4563', '124.4605', '142.4127']]
arxiv_ua_d1 = [float(v) for v in ['57.9633', '73.5206', '89.6998', '105.6648', '122.4644', '141.0747']]
arxiv_gad_d1 = [float(v) for v in ['64.5405', '108.2159', '112.5031', '132.3311', '149.6644', '164.8512']]

plt.figure(figsize=(8, 6))
plt.plot(x, arxiv_ga_d1, marker='o', label='GA+PoU')
plt.plot(x, arxiv_ar_d1, marker='s', label='AR+PoU')
plt.plot(x, arxiv_ft_d1, marker='^', label='FT+PoU')
plt.plot(x, arxiv_ua_d1, marker='v', label='UA+PoU')
plt.plot(x, arxiv_gad_d1, marker='*', label='GAD+PoU')
plt.xlabel("Checkpoint Interval (k)")
plt.ylabel(r'$\left\|| W_t - W_{t^\prime} \right\||_1$')
plt.ylim(40, 180)
plt.xlim(2, 12)
# plt.yticks(np.arange(0.4, 1.6, 0.2))
# plt.legend()
# plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5), ncol=5)  # 放在图外右侧中间
plt.grid(True)
plt.tight_layout()
plt.savefig("arxiv_d1_dist.pdf", dpi=300, format='pdf')
plt.show()

arxiv_ga_d2 = [float(v) for v in ['0.4197', '0.5361', '0.6140', '0.6708', '0.7288', '0.7899']]
arxiv_ar_d2 = [float(v) for v in ['0.4822', '0.6423', '0.7921', '0.8337', '0.9267', '0.9606']]
arxiv_ft_d2 = [float(v) for v in ['0.3795', '0.5096', '0.6292', '0.6994', '0.8480', '0.9590']]
arxiv_ua_d2 = [float(v) for v in ['0.3749', '0.5024', '0.6189', '0.7267', '0.8335', '0.9489']]
arxiv_gad_d2 = [float(v) for v in ['0.3837', '0.7028', '0.6952', '0.8293', '0.9473', '1.0523']]

plt.figure(figsize=(8, 6))
plt.plot(x, arxiv_ga_d2, marker='o', label='GA')
plt.plot(x, arxiv_ar_d2, marker='s', label='AR')
plt.plot(x, arxiv_ft_d2, marker='^', label='FT')
plt.plot(x, arxiv_ua_d2, marker='v', label='UA')
plt.plot(x, arxiv_gad_d2, marker='*', label='GAD')
plt.xlabel("Checkpoint Interval (k)")
plt.ylabel(r'$\left\|| W_t - W_{t^\prime} \right\||_2$')
plt.ylim(0.3, 1.2)
plt.xlim(2, 12)
# plt.yticks(np.arange(0.4, 1.6, 0.2))
# plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("arxiv_d2_dist.pdf", dpi=300, format='pdf')
plt.show()

github_ga_d1 = [float(v) for v in ['9.3130', '17.4477', '20.3669', '37.2403', '61.6382', '86.4126']]
github_ar_d1 = [float(v) for v in ['63.4353', '66.3460', '86.0098', '105.4336', '122.7063', '136.9114']]
github_ft_d1 = [float(v) for v in ['79.1963', '111.2383', '143.4929', '167.9928', '188.6660', '215.2871']]
github_ua_d1 = [float(v) for v in ['78.9023', '108.9166', '139.3809', '171.0482', '203.0373', '236.3264']]
github_gad_d1 = [float(v) for v in ['80.8231', '102.2939', '120.6789', '137.0978', '151.1570', '163.3074']]

plt.figure(figsize=(8, 6))
plt.plot(x, github_ga_d1, marker='o', label='GA')
plt.plot(x, github_ar_d1, marker='s', label='AR')
plt.plot(x, github_ft_d1, marker='^', label='FT')
plt.plot(x, github_ua_d1, marker='v', label='UA')
plt.plot(x, github_gad_d1, marker='*', label='GAD')
plt.xlabel("Checkpoint Interval (k)")
plt.ylabel(r'$\left\|| W_t - W_{t^\prime} \right\||_1$')
plt.ylim(0, 250)
plt.xlim(2, 12)
# plt.yticks(np.arange(0.4, 1.6, 0.2))
# plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("github_d1_dist.pdf", dpi=300, format='pdf')
plt.show()

github_ga_d2 = [float(v) for v in ['0.0643', '0.1205', '0.1483', '0.2563', '0.4048', '0.5659']]
github_ar_d2 = [float(v) for v in ['0.4017', '0.5235', '0.6436', '0.7506', '0.8540', '0.9466']]
github_ft_d2 = [float(v) for v in ['0.4769', '0.6977', '0.9171', '1.0780', '1.2138', '1.3876']]
github_ua_d2 = [float(v) for v in ['0.4755', '0.6856', '0.8944', '1.1035', '1.3148', '1.5330']]
github_gad_d2 = [float(v) for v in ['0.4900', '0.6569', '0.7985', '0.9229', '1.0288', '1.1207']]

plt.figure(figsize=(8, 6))
plt.plot(x, github_ga_d2, marker='o', label='GA')
plt.plot(x, github_ar_d2, marker='s', label='AR')
plt.plot(x, github_ft_d2, marker='^', label='FT')
plt.plot(x, github_ua_d2, marker='v', label='UA')
plt.plot(x, github_gad_d2, marker='*', label='GAD')
plt.xlabel("Checkpoint Interval (k)")
plt.ylabel(r'$\left\|| W_t - W_{t^\prime} \right\||_2$')
plt.ylim(0, 1.6)
plt.xlim(2, 12)
# plt.yticks(np.arange(0.4, 1.6, 0.2))
# plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("github_d2_dist.pdf", dpi=300, format='pdf')
plt.show()

