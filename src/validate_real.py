"""
validate_real.py  ── 第二層驗證
載入訓練好的 MLP-Transformer 模型，用「原始真實資料」(augmented/) 測試，
評估模型對真實世界的泛化能力（論文最終數字）。
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})
import seaborn as sns
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score, accuracy_score)
from torchvision import transforms

# ── 路徑 ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR   = os.path.join(BASE_DIR, "data", "real_val_mixed")   # 真實驗證資料（6/10 混合，water 用 5/31）
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_mlp_transformer_v2.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = ["motor_oil", "olive_oil", "palm_oil", "lard", "water"]
REAL_TEST_PER_CLASS = 50   # 每類取前 50 張原始真實樣本做第二層驗證
NUM_CLASSES = len(CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置：{device}")

# ── Dataset（真實資料）──────────────────────────────────────────────────────
class RealDataset(Dataset):
    """讀 real_test/{class}/*_sg.txt + 對應 *.bmp（支援 timestamp 及 _cpN 命名）"""
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples   = []
        for label, cls in enumerate(CLASSES):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                print(f"警告：找不到 {cls_dir}"); continue
            orig_samples = []
            for f in sorted(os.listdir(cls_dir)):
                if not f.endswith("_sg.txt"): continue
                if "_cp" in f: continue                  # 跳過複製補齊的檔案
                base  = f[:-7]                          # 去掉 _sg.txt
                txt_p = os.path.join(cls_dir, f)
                img_p = os.path.join(cls_dir, f"{base}.bmp")
                if os.path.exists(img_p):
                    orig_samples.append((txt_p, img_p, label))
            self.samples.extend(orig_samples[:REAL_TEST_PER_CLASS])
        print(f"真實測試集大小：{len(self.samples)}")
        for i, cls in enumerate(CLASSES):
            n = sum(1 for _, _, l in self.samples if l == i)
            print(f"  {cls}: {n} 樣本")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        txt_p, img_p, label = self.samples[idx]
        with open(txt_p) as f:
            txt = np.array([float(x) for x in f.read().strip().split()], dtype=np.float32)
        if len(txt) != 1280:
            txt = np.interp(np.linspace(0, len(txt)-1, 1280),
                            np.arange(len(txt)), txt).astype(np.float32)
        txt = txt / 255.0   # 固定縮放（與訓練一致，保留振幅差異）
        txt = torch.tensor(txt)
        img = Image.open(img_p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return txt, img, label

# ── 模型架構（須與 MLP+Tran_test.py 完全一致）────────────────────────────────
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
        self.patch_size  = 40
        self.embed_dim   = 512
        self.patch_embed = nn.Conv2d(3, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        encoder_layer    = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fc          = nn.Linear(self.embed_dim, output_size)
    def forward(self, x):
        x = self.patch_embed(x).flatten(2).permute(0, 2, 1)
        return self.fc(self.transformer(x).mean(dim=1))

class TransformerFusionModel(nn.Module):
    def __init__(self, mlp_input_size=1280, num_classes=NUM_CLASSES):
        super().__init__()
        self.mlp = MLP(input_size=mlp_input_size, hidden_size=512, output_size=128)
        self.transformer_branch = TransformerBranch(output_size=128)
        self.fc = nn.Linear(128 + 128, num_classes)
    def forward(self, txt, img):
        return self.fc(torch.cat([self.mlp(txt), self.transformer_branch(img)], dim=1))

# ── 載入模型 ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((400, 1280)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"找不到模型：{MODEL_PATH}（請先訓練）")

model = TransformerFusionModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"已載入模型：{MODEL_PATH}\n")

dataset = RealDataset(REAL_DIR, transform)
loader  = DataLoader(dataset, batch_size=8, shuffle=False)

# ── 推論 ─────────────────────────────────────────────────────────────────────
all_preds, all_labels = [], []
with torch.no_grad():
    for txt, img, labels in loader:
        txt, img = txt.to(device), img.to(device)
        out = model(txt, img)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ── 評估指標 ─────────────────────────────────────────────────────────────────
overall_acc = 100 * accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
prec = 100 * precision_score(all_labels, all_preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
rec  = 100 * recall_score(all_labels, all_preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))
f1   = 100 * f1_score(all_labels, all_preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))

print("="*60)
print(f"【第二層驗證 — 真實資料】總體準確率：{overall_acc:.2f}%")
print("="*60)
print(f"{'類別':<12}{'準確率':>8}{'精確率':>8}{'召回率':>8}{'F1':>8}")
print("-"*60)
for i, cls in enumerate(CLASSES):
    cls_acc = 100 * cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
    print(f"{cls:<12}{cls_acc:>7.1f}%{prec[i]:>7.1f}%{rec[i]:>7.1f}%{f1[i]:>7.1f}%")
print("-"*60)
print(f"{'平均':<12}{overall_acc:>7.1f}%{prec.mean():>7.1f}%{rec.mean():>7.1f}%{f1.mean():>7.1f}%")

# ── 存結果 ───────────────────────────────────────────────────────────────────
# 文字報告
with open(os.path.join(OUTPUT_DIR, "validate_real_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"第二層驗證（真實資料）總體準確率：{overall_acc:.2f}%\n\n")
    f.write(f"{'類別':<12}{'準確率':>8}{'精確率':>8}{'召回率':>8}{'F1':>8}\n")
    for i, cls in enumerate(CLASSES):
        cls_acc = 100 * cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        f.write(f"{cls:<12}{cls_acc:>7.1f}%{prec[i]:>7.1f}%{rec[i]:>7.1f}%{f1[i]:>7.1f}%\n")

# 混淆矩陣圖
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title(f"Real-Data Validation  (Acc={overall_acc:.1f}%)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "validate_real_confusion.png"), dpi=150)
plt.close()

print(f"\n結果已存：")
print(f"  {OUTPUT_DIR}/validate_real_report.txt")
print(f"  {OUTPUT_DIR}/validate_real_confusion.png")
