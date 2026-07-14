"""
make_eval_figures.py
載入已訓練的 best 模型，重新評估測試集，產生：
  - confusion_matrix_best.png
  - roc_curve_best.png
  - pr_curve_best.png
  - class_metrics_curve_best.png（從 outputs/xxx_metrics_mlptransformer.txt 讀取）
所有圖：Times New Roman 字體，Motor Oil / Olive Oil / Palm Oil / Lard / Water 標籤
"""
import os
import re
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_mlp_transformer_v2.pth")
TEST_DIR   = os.path.join(BASE_DIR, "data_paper", "real_source", "real_test")
OUT_DIR    = os.path.join(BASE_DIR, "outputs")
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES  = ["motor_oil", "olive_oil", "palm_oil", "lard", "water"]
DISPLAY  = {"motor_oil": "Motor Oil", "olive_oil": "Olive Oil",
            "palm_oil": "Palm Oil", "lard": "Lard", "water": "Water"}
LABELS   = [DISPLAY[c] for c in CLASSES]

# ── 模型定義 ─────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_size=1280, hidden_size=512, output_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class TransformerBranch(nn.Module):
    def __init__(self, output_size=128):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 512, kernel_size=40, stride=40)
        el = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(el, num_layers=3)
        self.fc = nn.Linear(512, output_size)
    def forward(self, x):
        x = self.patch_embed(x).flatten(2).permute(0, 2, 1)
        return self.fc(self.transformer(x).mean(1))

class TransformerFusionModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.mlp = MLP()
        self.transformer_branch = TransformerBranch()
        self.fc = nn.Linear(256, num_classes)
    def forward(self, txt, img):
        return self.fc(torch.cat([self.mlp(txt), self.transformer_branch(img)], 1))

# ── 資料集 ───────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((400, 1280)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

class RealDataset(Dataset):
    def __init__(self, root):
        self.s = []
        for ci, c in enumerate(CLASSES):
            cd = os.path.join(root, c)
            if not os.path.isdir(cd): continue
            for f in sorted(os.listdir(cd)):
                if not f.endswith("_sg.txt"): continue
                base = f[:-7]
                bp = os.path.join(cd, f"{base}.bmp")
                if os.path.exists(bp):
                    self.s.append((os.path.join(cd, f), bp, ci))
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        t, b, l = self.s[i]
        x = np.loadtxt(t, dtype=np.float32)
        if len(x) != 1280:
            x = np.interp(np.linspace(0, len(x)-1, 1280), np.arange(len(x)), x).astype(np.float32)
        return torch.tensor(x / 255.0), transform(Image.open(b).convert('RGB')), l

# ── 評估 ─────────────────────────────────────────────────────────────────────
def evaluate(model, loader):
    all_labels, all_preds, all_probs = [], [], []
    model.eval()
    with torch.no_grad():
        for txt, img, lab in loader:
            out = model(txt.to(device), img.to(device))
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(lab.numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# ── 混淆矩陣 ─────────────────────────────────────────────────────────────────
def save_confusion_matrix(y, p):
    cm = confusion_matrix(y, p)
    acc = (y == p).mean() * 100
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABELS, yticklabels=LABELS,
                annot_kws={"size": 13}, square=True, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Confusion Matrix  (Acc = {acc:.2f}%)', fontsize=13)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_best.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_best.pdf"), bbox_inches='tight')
    plt.close()
    print(f"  confusion_matrix_best.png  Acc={acc:.2f}%")

# ── ROC 曲線 ─────────────────────────────────────────────────────────────────
def save_roc(y, probs):
    plt.figure(figsize=(7, 4))
    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y == i, probs[:, i])
        plt.plot(fpr, tpr, label=f'{DISPLAY[c]} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_curve_best.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, "roc_curve_best.pdf"), bbox_inches='tight')
    plt.close()
    print("  roc_curve_best.png")

# ── PR 曲線 ──────────────────────────────────────────────────────────────────
def save_pr(y, probs):
    plt.figure(figsize=(7, 4))
    for i, c in enumerate(CLASSES):
        prec, rec, _ = precision_recall_curve(y == i, probs[:, i])
        plt.plot(rec, prec, label=f'{DISPLAY[c]} (AUC = {auc(rec, prec):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pr_curve_best.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, "pr_curve_best.pdf"), bbox_inches='tight')
    plt.close()
    print("  pr_curve_best.png")

# ── class_metrics_curve（從 txt 讀取）────────────────────────────────────────
def save_class_metrics_curve():
    data = {}
    for c in CLASSES:
        path = os.path.join(OUT_DIR, f"{c}_metrics_mlptransformer.txt")
        if not os.path.exists(path):
            print(f"  ⚠ 找不到 {path}，跳過"); return
        epochs, accs, precs, recs, f1s = [], [], [], [], []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('Epoch') or line.startswith('---') or line.startswith('Avg'):
                    continue
                parts = re.split(r'\s*\|\s*', line)
                if len(parts) < 5: continue
                try:
                    epochs.append(int(parts[0]))
                    accs.append(float(parts[1].replace('%', '')))
                    precs.append(float(parts[2].replace('%', '')))
                    recs.append(float(parts[3].replace('%', '')))
                    f1s.append(float(parts[4].replace('%', '')))
                except ValueError:
                    continue
        data[c] = {'epoch': epochs, 'acc': accs, 'prec': precs, 'rec': recs, 'f1': f1s}

    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    titles  = ['Class Accuracy', 'Class Precision', 'Class Recall', 'Class F1 Score']
    keys    = ['acc', 'prec', 'rec', 'f1']
    ylabels = ['Class Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1 Score (%)']
    for ax, title, key, ylabel in zip(axes.flat, titles, keys, ylabels):
        for c in CLASSES:
            d = data[c]
            ax.plot(d['epoch'], d[key], label=f'{DISPLAY[c]}')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "class_metrics_curve_best.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, "class_metrics_curve_best.pdf"), bbox_inches='tight')
    plt.close()
    print("  class_metrics_curve_best.png")

# ── 各指標獨立圖（4 張，從 txt 讀取）────────────────────────────────────────
def save_individual_class_metrics():
    data = {}
    for c in CLASSES:
        path = os.path.join(OUT_DIR, f"{c}_metrics_mlptransformer.txt")
        if not os.path.exists(path):
            print(f"  ⚠ 找不到 {path}，跳過"); return
        epochs, accs, precs, recs, f1s = [], [], [], [], []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('Epoch') or line.startswith('---') or line.startswith('Avg'):
                    continue
                parts = re.split(r'\s*\|\s*', line)
                if len(parts) < 5: continue
                try:
                    epochs.append(int(parts[0]))
                    accs.append(float(parts[1].replace('%', '')))
                    precs.append(float(parts[2].replace('%', '')))
                    recs.append(float(parts[3].replace('%', '')))
                    f1s.append(float(parts[4].replace('%', '')))
                except ValueError:
                    continue
        data[c] = {'epoch': epochs, 'acc': accs, 'prec': precs, 'rec': recs, 'f1': f1s}

    for key, title, ylabel, fname in [
        ('acc',  'Class Accuracy',   'Class Accuracy (%)',  'class_acc_curve_mlptransformer'),
        ('prec', 'Class Precision',  'Precision (%)',       'class_prec_curve_mlptransformer'),
        ('rec',  'Class Recall',     'Recall (%)',          'class_rec_curve_mlptransformer'),
        ('f1',   'Class F1 Score',   'F1 Score (%)',        'class_f1_curve_mlptransformer'),
    ]:
        plt.figure(figsize=(7, 4))
        for c in CLASSES:
            plt.plot(data[c]['epoch'], data[c][key], label=DISPLAY[c])
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{fname}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(OUT_DIR, f"{fname}.pdf"), bbox_inches='tight')
        plt.close()
        print(f"  {fname}.png")


# ── 主程式 ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("載入模型...")
    model = TransformerFusionModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    model.eval()

    print("評估測試集...")
    ds = RealDataset(TEST_DIR)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    y, p, probs = evaluate(model, loader)

    print("產生圖表...")
    save_confusion_matrix(y, p)
    save_roc(y, probs)
    save_pr(y, probs)
    save_class_metrics_curve()
    save_individual_class_metrics()
    print("完成！")
