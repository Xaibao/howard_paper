"""
LIME.py ── 可解釋性分析
對訓練好的 MLP-Transformer 模型，用 LIME 解釋「影像分支」哪些區域影響辨識結果。
解釋對象：真實原始資料（augmented/）
"""
import torch
import torch.nn as nn
from torchvision import transforms
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
from lime import lime_image
from skimage.segmentation import mark_boundaries

# ── 路徑 ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR   = os.path.join(BASE_DIR, "data", "real", "real_test")
MODEL_PATH = os.path.join(BASE_DIR, "models", "mlp_transformer_v2.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = ["motor_oil", "olive_oil", "palm_oil", "lard", "water"]
NUM_CLASSES = len(CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置：{device}")

# ── 模型架構（須與 MLP+Tran.py 一致：無 EarlyConv）────────────────────────────
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
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"已載入模型：{MODEL_PATH}")

# ── 選一個真實樣本來解釋 ──────────────────────────────────────────────────────
# 可改 TARGET_CLASS / SAMPLE_IDX 解釋不同樣本
TARGET_CLASS = "palm_oil"   # 解釋哪一類
SAMPLE_IDX   = 0            # 該類第幾個樣本

cls_dir  = os.path.join(REAL_DIR, TARGET_CLASS)
sg_files = sorted([f for f in os.listdir(cls_dir) if f.endswith("_sg.txt")])
sg_name  = sg_files[SAMPLE_IDX]
base     = sg_name[:-7]                          # 去掉 _sg.txt
bmp_path = os.path.join(cls_dir, f"{base}.bmp")

# 載入該樣本的波形（固定縮放 /255，與訓練一致）
txt = np.loadtxt(os.path.join(cls_dir, sg_name), dtype=np.float32)
if len(txt) != 1280:
    txt = np.interp(np.linspace(0, len(txt)-1, 1280), np.arange(len(txt)), txt).astype(np.float32)
txt = torch.tensor(txt / 255.0).to(device)

print(f"\n解釋樣本：{bmp_path}")
print(f"真實類別：{TARGET_CLASS}")

# ── LIME 預測器（橋接雙輸入模型）─────────────────────────────────────────────
def lime_predictor(numpy_images):
    # LIME 擾動的影像都對應同一條光譜
    txt_batch = txt.unsqueeze(0).repeat(numpy_images.shape[0], 1)
    lime_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    imgs_uint8 = (numpy_images * 255).astype(np.uint8)
    img_batch  = torch.stack([lime_tf(im) for im in imgs_uint8]).to(device)
    with torch.no_grad():
        out = model(txt_batch, img_batch)
        return torch.softmax(out, dim=1).cpu().numpy()

# ── 執行 LIME ────────────────────────────────────────────────────────────────
print("生成 LIME 解釋中...（需一點時間）")
explainer = lime_image.LimeImageExplainer()
display_image = np.array(Image.open(bmp_path).convert('RGB').resize((1280, 400)))

explanation = explainer.explain_instance(
    display_image, lime_predictor, top_labels=1, hide_color=0, num_samples=1000
)

pred_class = CLASSES[explanation.top_labels[0]]
print(f"模型預測：{pred_class}")

# ── 可視化（原圖 + LIME 疊圖）───────────────────────────────────────────────
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False
)

fig, axes = plt.subplots(2, 1, figsize=(7, 5))
axes[0].imshow(display_image)
axes[0].set_title(f"Original — True: {TARGET_CLASS}", fontsize=13)
axes[0].axis('off')
axes[1].imshow(mark_boundaries(temp / 255.0, mask))
axes[1].set_title(f"LIME Explanation — Predicted: {pred_class}", fontsize=13)
axes[1].axis('off')
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, f"lime_{TARGET_CLASS}.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"\n完成！LIME 解釋圖已存：{out_path}")
