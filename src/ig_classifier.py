"""
ig_classifier.py ── 解釋 MLP-Transformer 分類器
用 Integrated Gradients 分析：輸入光譜的哪些波段最影響分類決策。
對 5 類各取一個真實樣本，輸出每類的波段重要性圖 + 合併比較圖。
"""
import torch
import torch.nn as nn
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
from captum.attr import IntegratedGradients
from torchvision import transforms
from PIL import Image
import os
import glob

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR   = os.path.join(BASE_DIR, "data_paper", "real_source", "real_test")
MODEL_PATH = os.path.join(BASE_DIR, "models", "mlp_transformer_v2.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = ["motor_oil", "olive_oil", "palm_oil", "lard", "water"]
DISPLAY = {"motor_oil": "Motor Oil", "olive_oil": "Olive Oil",
           "palm_oil": "Palm Oil", "lard": "Lard", "water": "Water"}
NUM_CLASSES = len(CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置：{device}")

# ── 模型架構 ──────────────────────────────────────────────────────────────────
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
        self.mlp = MLP(input_size=mlp_input_size)
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

model = TransformerFusionModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model.eval()
print(f"已載入模型：{MODEL_PATH}\n")

# ── IG 計算（固定影像，對光譜做 IG）─────────────────────────────────────────
def compute_ig(txt_tensor, img_tensor, target_class_idx):
    img_fixed = img_tensor.detach()

    def forward_fn(txt_input):
        bs = txt_input.shape[0]
        return model(txt_input, img_fixed.expand(bs, -1, -1, -1))

    ig = IntegratedGradients(forward_fn)
    baseline = torch.zeros_like(txt_tensor)
    attributions, delta = ig.attribute(
        txt_tensor,
        baselines=baseline,
        target=target_class_idx,
        return_convergence_delta=True,
        n_steps=200
    )
    print(f"  IG 收斂誤差 Delta: {delta.item():.4f}")
    return attributions.squeeze().detach().cpu().numpy()

# ── 處理每一類 ────────────────────────────────────────────────────────────────
wavelengths = np.linspace(300, 1000, 1280)
N_BINS = 40
all_binned   = {}
all_raw_txt  = {}

for cls_idx, cls_name in enumerate(CLASSES):
    cls_dir = os.path.join(REAL_DIR, cls_name)
    # 取第一個原始樣本（不含 _cp）
    sg_files = sorted([f for f in os.listdir(cls_dir)
                       if f.endswith("_sg.txt") and "_cp" not in f])
    if not sg_files:
        print(f"[{cls_name}] 找不到原始樣本，跳過"); continue

    sg_name  = sg_files[0]
    base     = sg_name[:-7]
    txt_path = os.path.join(cls_dir, sg_name)
    bmp_path = os.path.join(cls_dir, f"{base}.bmp")

    if not os.path.exists(bmp_path):
        print(f"[{cls_name}] 找不到 BMP，跳過"); continue

    print(f"[{cls_name}] 樣本：{sg_name}")

    # 讀光譜
    raw = np.loadtxt(txt_path, dtype=np.float32)
    if len(raw) != 1280:
        raw = np.interp(np.linspace(0, len(raw)-1, 1280), np.arange(len(raw)), raw).astype(np.float32)
    all_raw_txt[cls_name] = raw
    txt_tensor = torch.tensor(raw / 255.0).unsqueeze(0).to(device).requires_grad_(True)

    # 讀影像
    img = Image.open(bmp_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 預測
    with torch.no_grad():
        logits = model(txt_tensor.detach(), img_tensor)
        pred_idx  = logits.argmax(1).item()
        pred_name = CLASSES[pred_idx]
    print(f"  預測：{pred_name}（真實：{cls_name}）")

    # IG（對真實類別解釋）
    attr = compute_ig(txt_tensor, img_tensor, cls_idx)

    # 分箱
    idx_bins = np.array_split(np.arange(1280), N_BINS)
    centers  = np.array([wavelengths[b].mean() for b in idx_bins])
    binned   = np.array([np.abs(attr[b]).mean() for b in idx_bins])
    all_binned[cls_name] = (centers, binned)

    # 單類圖
    fig, axes = plt.subplots(2, 1, figsize=(7, 5))

    # 上：原始光譜
    axes[0].plot(wavelengths, raw, color='#2c7bb6', linewidth=1.2)
    axes[0].set_title(f'Input Spectrum — {DISPLAY[cls_name]}')
    axes[0].set_xlabel('Wavelength (nm)')
    axes[0].set_ylabel('Intensity')
    axes[0].grid(True, ls='--', alpha=0.4)

    # 下：波段重要性
    top5 = np.argsort(binned)[-5:]
    colors = ['crimson' if i in top5 else 'steelblue' for i in range(len(centers))]
    axes[1].bar(centers, binned, width=700/N_BINS*0.9, color=colors)
    axes[1].set_title(f'IG Band Importance — {DISPLAY[cls_name]}  (predicted: {DISPLAY[pred_name]})')
    axes[1].set_xlabel('Wavelength (nm)')
    axes[1].set_ylabel('|Attribution|')
    axes[1].grid(True, ls='--', alpha=0.4)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"ig_cls_{cls_name}.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  已存：{out}\n")

# ── 5 類合併比較圖 ────────────────────────────────────────────────────────────
if len(all_binned) == NUM_CLASSES:
    fig, axes = plt.subplots(NUM_CLASSES, 1, figsize=(7, 2.5 * NUM_CLASSES), sharex=True)
    for ax, cls_name in zip(axes, CLASSES):
        centers, binned = all_binned[cls_name]
        top5 = np.argsort(binned)[-5:]
        colors = ['crimson' if i in top5 else 'steelblue' for i in range(len(centers))]
        ax.bar(centers, binned / binned.max(), width=700/N_BINS*0.9, color=colors)
        ax.set_ylabel('Norm. |Attr.|', fontsize=10)
        ax.set_title(DISPLAY[cls_name], fontsize=12)
        ax.grid(True, ls='--', alpha=0.4)
    axes[-1].set_xlabel('Wavelength (nm)')
    plt.tight_layout()
    out_all = os.path.join(OUTPUT_DIR, "ig_cls_all_comparison.png")
    plt.savefig(out_all, dpi=300, bbox_inches='tight')
    plt.savefig(out_all.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"合併圖已存：{out_all}")

print("\n完成！")
