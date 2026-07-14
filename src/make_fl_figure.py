"""
make_fl_figure.py — 生成 FL 收斂曲線圖
從 outputs/fl_training_log.txt 讀取每輪數據，輸出雙軸折線圖。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import re

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH   = os.path.join(BASE_DIR, "outputs", "fl_training_log.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

plt.rcParams.update({
    'font.family':     'Times New Roman',
    'font.size':       13,
    'axes.titlesize':  13,
    'axes.labelsize':  12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
})

# 從 log 解析數據
rounds, losses, accuracies = [], [], []
with open(LOG_PATH, 'r') as f:
    for line in f:
        m = re.search(r'Round (\d+) \| Loss: ([\d.]+) \| Accuracy: ([\d.]+)', line)
        if m:
            rounds.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            accuracies.append(float(m.group(3)) * 100)  # 轉成 %

if not rounds:
    print("找不到數據，請確認 fl_training_log.txt 有內容")
    exit(1)

print(f"讀取到 {len(rounds)} 輪數據：Round {rounds[0]}–{rounds[-1]}")

fig, ax1 = plt.subplots(figsize=(7, 4))

# 左軸：Accuracy
color_acc = '#2563eb'
ax1.plot(rounds, accuracies, 'o-', color=color_acc, lw=2, markersize=5,
         label='Global Accuracy (%)')
ax1.set_xlabel('Communication Round')
ax1.set_ylabel('Global Accuracy (%)', color=color_acc)
ax1.tick_params(axis='y', labelcolor=color_acc)
ax1.set_ylim(95, 101)
ax1.set_xticks(rounds)
ax1.grid(True, ls='--', alpha=0.4)

# 右軸：Loss
ax2 = ax1.twinx()
color_loss = '#dc2626'
ax2.plot(rounds, losses, 's--', color=color_loss, lw=1.5, markersize=4,
         label='Global Loss')
ax2.set_ylabel('Loss', color=color_loss)
ax2.tick_params(axis='y', labelcolor=color_loss)
ax2.set_ylim(0.38, 0.43)

# 合併圖例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

plt.tight_layout()

out_png = os.path.join(OUTPUT_DIR, "fl_convergence_curve.png")
out_pdf = os.path.join(OUTPUT_DIR, "fl_convergence_curve.pdf")
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()

print(f"已存：{out_png}")
print(f"已存：{out_pdf}")
