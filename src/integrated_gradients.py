"""
integrated_gradients.py ── 解釋 Generated AI（CGAN 生成器）
用 Integrated Gradients 分析：輸入光譜的哪些波段（特徵）最影響 CGAN 生成的光譜圖。
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
from torchvision import transforms as T
import torch.utils.checkpoint as checkpoint
import os
import glob

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NOISE_DIM     = 100
CONDITION_DIM = 1280

# ── Generator（須與 CGAN_spectrum.py 完全一致）──────────────────────────────
class Generator(nn.Module):
    def __init__(self, noise_dim=100, condition_dim=1280):
        super().__init__()
        self.project = nn.Linear(noise_dim + condition_dim, 64 * 4 * 6)
        self.vertical_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(15, 1), padding=(7, 0)),
            nn.BatchNorm2d(64), nn.ReLU(True)
        )
        self.pixel_shuffle_block = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2), nn.BatchNorm2d(64), nn.LeakyReLU(0.2)
        )
        self.upscale_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.ConvTranspose2d(16,  8, 5, 2, 1), nn.BatchNorm2d(8),  nn.ReLU(True),
            nn.ConvTranspose2d( 8,  1, 5, 2, 1), nn.Sigmoid()
        )

    def forward(self, noise, condition):
        x = self.project(torch.cat([noise, condition], dim=1)).view(-1, 64, 4, 6)
        # IG 需要梯度，用 use_reentrant=False
        x = checkpoint.checkpoint(self.vertical_block,      x, use_reentrant=False)
        x = checkpoint.checkpoint(self.pixel_shuffle_block, x, use_reentrant=False)
        x = checkpoint.checkpoint(self.upscale_layers,      x, use_reentrant=False)
        x = T.functional.resize(x, (800, 1280), interpolation=T.InterpolationMode.BILINEAR)
        return x

# ── IG 解釋 ──────────────────────────────────────────────────────────────────
def explain_with_ig(model, condition_tensor, noise_tensor, device):
    model.eval()
    def forward_wrapper(condition_input):
        bs = condition_input.shape[0]
        expanded_noise = noise_tensor.expand(bs, -1)
        out = model(expanded_noise, condition_input)
        return out.sum(dim=(1, 2, 3))   # 每張生成圖的總強度當純量輸出

    ig = IntegratedGradients(forward_wrapper)
    baseline = torch.zeros_like(condition_tensor, device=device)
    attributions, delta = ig.attribute(condition_tensor, baselines=baseline,
                                       return_convergence_delta=True, n_steps=200)
    print(f"IG 收斂誤差 Delta: {delta.item():.4f}")
    with torch.no_grad():
        generated = model(noise_tensor, condition_tensor).detach().cpu().numpy()
    return generated, attributions.cpu().numpy()

# ── 視覺化 ───────────────────────────────────────────────────────────────────
_DISPLAY = {"motor_oil":"Motor Oil","olive_oil":"Olive Oil",
            "palm_oil":"Palm Oil","lard":"Lard","water":"Water"}

def visualize(input_data, generated, attributions, tag):
    input_data   = input_data.flatten()
    attributions = attributions.flatten()
    wl = np.linspace(300, 1000, len(input_data))
    disp_tag = _DISPLAY.get(tag, tag)

    # 分箱資料（共用）
    n_bins   = 40
    idx_bins = np.array_split(np.arange(len(attributions)), n_bins)
    centers  = np.array([wl[b].mean() for b in idx_bins])
    binned   = np.array([np.abs(attributions[b]).mean() for b in idx_bins])
    top      = np.argsort(binned)[-5:]

    # ── 圖 1：Generated Spectrogram ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.5))
    spec = generated.squeeze()
    im = ax.imshow(spec, aspect='auto', cmap='viridis', origin='lower')
    ax.set_title(f'Generated Spectrogram — {disp_tag}')
    ax.set_xlabel('Width'); ax.set_ylabel('Height')
    fig.colorbar(im, ax=ax, label='Intensity')
    plt.tight_layout()
    out1 = os.path.join(OUTPUT_DIR, f"ig_{tag}_spectrogram.png")
    plt.savefig(out1, dpi=300, bbox_inches='tight')
    plt.savefig(out1.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"已儲存：{out1}")

    # ── 圖 2：Input Condition Spectrum ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(wl, input_data, color='blue')
    ax.set_title('Input Condition Spectrum')
    ax.set_xlabel('Wavelength (nm)'); ax.set_ylabel('Intensity')
    ax.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, f"ig_{tag}_spectrum.png")
    plt.savefig(out2, dpi=300, bbox_inches='tight')
    plt.savefig(out2.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"已儲存：{out2}")

    # ── 圖 3：Band Importance ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(centers, binned, width=700/n_bins*0.9, color='steelblue')
    ax.bar(centers[top], binned[top], width=700/n_bins*0.9, color='crimson')
    ax.set_title(f'Integrated Gradients — Band Importance ({disp_tag})')
    ax.set_xlabel('Wavelength (nm)'); ax.set_ylabel('|Attribution|')
    ax.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    out3 = os.path.join(OUTPUT_DIR, f"ig_{tag}_band_importance.png")
    plt.savefig(out3, dpi=300, bbox_inches='tight')
    plt.savefig(out3.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"已儲存：{out3}")

# ── 主程式 ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    TARGET_CLASS = "palm_oil"   # 解釋哪一類的生成器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")

    model_path = os.path.join(BASE_DIR, "models", f"CGAN_spectrum_{TARGET_CLASS}_A_train.pth")
    _cond_files = sorted(glob.glob(os.path.join(BASE_DIR, "data", "real", "real_test", TARGET_CLASS, "*_sg.txt")))
    cond_path   = _cond_files[0] if _cond_files else ""

    if not os.path.exists(model_path):
        print(f"找不到模型：{model_path}"); exit()
    if not cond_path:
        print(f"找不到條件光譜：data/real/real_test/{TARGET_CLASS}/"); exit()

    model = Generator(NOISE_DIM, CONDITION_DIM).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"已載入：{model_path}")

    data = np.loadtxt(cond_path, dtype=np.float32)
    if len(data) != 1280:
        data = np.interp(np.linspace(0, len(data)-1, 1280), np.arange(len(data)), data).astype(np.float32)
    condition = torch.from_numpy(data).view(1, -1).to(device)
    noise = torch.randn(1, NOISE_DIM, device=device)

    print("計算 Integrated Gradients...")
    generated, attr = explain_with_ig(model, condition, noise, device)
    visualize(condition.cpu().numpy(), generated, attr, TARGET_CLASS)
    print("完成！")
