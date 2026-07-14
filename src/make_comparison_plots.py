"""
make_comparison_plots.py
產生論文用的對比圖：
  curve_comparison.png   ── 波形 真實 vs CGAN生成
  bmp_comparison.png     ── 影像 真實 vs CGAN生成
  food_oils_compare.png  ── 三種食用油平均光譜疊圖（Supplementary）
  curve_difference.png   ── 真實 vs 生成 差距圖（mean ± std + residual）
"""
import os
import numpy as np
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
from PIL import Image

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR   = os.path.join(BASE_DIR, "data", "real", "real_test")
GEN_DIR    = os.path.join(BASE_DIR, "data", "dataset", "train")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES  = ["motor_oil", "olive_oil", "palm_oil", "lard", "water"]
_DISPLAY = {"motor_oil":"Motor Oil","olive_oil":"Olive Oil",
            "palm_oil":"Palm Oil","lard":"Lard","water":"Water"}
wl = np.linspace(300, 1000, 1280)
plt.rcParams.update({'font.size': 12})

# ── 1. 各類別獨立波形圖（真實 × 5 + 生成 × 5）────────────────────────────────
def individual_curves():
    for cls in CLASSES:
        od = os.path.join(REAL_DIR, cls)
        gd = os.path.join(GEN_DIR, cls, "txt")
        origs = [np.loadtxt(os.path.join(od, f)) for f in sorted(os.listdir(od))
                 if f.endswith('_sg.txt') and '_cp' not in f][:8]
        gens  = [np.loadtxt(os.path.join(gd, f)) for f in sorted(os.listdir(gd))
                 if f.endswith('.txt')][:8]

        # 真實曲線
        plt.figure(figsize=(7, 4))
        for o in origs: plt.plot(wl, o, alpha=0.7, lw=1.2)
        plt.title(f'{_DISPLAY[cls]} — Real Spectrum')
        plt.xlabel('Wavelength (nm)'); plt.ylabel('Intensity'); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{cls}_real_curve.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, f"{cls}_real_curve.pdf"), bbox_inches='tight')
        plt.close()

        # 生成曲線
        plt.figure(figsize=(7, 4))
        for g in gens: plt.plot(wl, g, alpha=0.7, lw=1.2)
        plt.title(f'{_DISPLAY[cls]} — CGAN Generated Spectrum')
        plt.xlabel('Wavelength (nm)'); plt.ylabel('Intensity'); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{cls}_generated_curve.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, f"{cls}_generated_curve.pdf"), bbox_inches='tight')
        plt.close()

        print(f"  {cls}_real_curve.png / {cls}_generated_curve.png")

# ── 2. 影像對比 ──────────────────────────────────────────────────────────────
def bmp_comparison():
    fig, axes = plt.subplots(5, 4, figsize=(7, 5))
    for r, cls in enumerate(CLASSES):
        od, gd = os.path.join(REAL_DIR, cls), os.path.join(GEN_DIR, cls, "bmp")
        of = sorted([f for f in os.listdir(od) if f.endswith('.bmp')])[:2]
        gf = sorted(os.listdir(gd))[:2]
        for c, f in enumerate(of):
            axes[r,c].imshow(np.array(Image.open(os.path.join(od, f)).convert('L')), cmap='gray', aspect='auto')
            axes[r,c].set_title(f'{_DISPLAY[cls]} — Real', fontsize=10); axes[r,c].axis('off')
        for c, f in enumerate(gf):
            axes[r,c+2].imshow(np.array(Image.open(os.path.join(gd, f)).convert('L')), cmap='gray', aspect='auto')
            axes[r,c+2].set_title(f'{_DISPLAY[cls]} — Generated', fontsize=10); axes[r,c+2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "bmp_comparison.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "bmp_comparison.pdf"), bbox_inches='tight')
    plt.close()
    print("  bmp_comparison.png")

# ── 3. 食用油相似性（Supplementary）─────────────────────────────────────────
def food_oils_compare():
    classes = ['motor_oil', 'olive_oil', 'palm_oil', 'lard', 'water']
    labels  = {'motor_oil': 'Motor Oil', 'olive_oil': 'Olive Oil',
               'palm_oil': 'Palm Oil', 'lard': 'Lard', 'water': 'Water'}
    colors  = {'motor_oil': '#dc2626', 'olive_oil': '#16a34a',
               'palm_oil': '#d97706',  'lard': '#7c3aed', 'water': '#2563eb'}
    plt.figure(figsize=(7, 4))
    for cls in classes:
        od = os.path.join(REAL_DIR, cls)
        d = [np.loadtxt(os.path.join(od, f)) for f in sorted(os.listdir(od))
             if f.endswith('_sg.txt') and '_cp' not in f]
        if not d:
            continue
        plt.plot(wl, np.mean(d, axis=0), label=labels[cls], color=colors[cls], lw=2)
    plt.xlabel('Wavelength (nm)'); plt.ylabel('Intensity')
    plt.title('Mean Spectra of Five Water Quality Classes')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "food_oils_compare.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "food_oils_compare.pdf"), bbox_inches='tight')
    plt.close()
    print("  food_oils_compare.png")

# ── 4. 真實 vs 生成 差距圖 ───────────────────────────────────────────────────
def curve_difference():
    """
    每個類別一列，三個子圖：
      左：真實平均（mean ± std 陰影）
      中：生成平均（mean ± std 陰影）
      右：殘差 = mean_real - mean_generated（正值=真實較高，負值=生成較高）
    """
    CLASS_COLORS = {
        'motor_oil': '#dc2626',
        'olive_oil': '#16a34a',
        'palm_oil':  '#d97706',
        'lard':      '#7c3aed',
        'water':     '#2563eb',
    }

    fig, axes = plt.subplots(5, 3, figsize=(7, 8))
    fig.suptitle('Real vs. CGAN-Generated Spectral Curves — Mean ± Std & Residual',
                 fontsize=14, y=1.01)

    for r, cls in enumerate(CLASSES):
        color = CLASS_COLORS[cls]
        od = os.path.join(REAL_DIR, cls)
        gd = os.path.join(GEN_DIR, cls, "txt")

        # 讀取全部真實和生成曲線
        real_curves = np.array([
            np.loadtxt(os.path.join(od, f))
            for f in sorted(os.listdir(od)) if f.endswith('_sg.txt')
        ])
        gen_curves = np.array([
            np.loadtxt(os.path.join(gd, f))
            for f in sorted(os.listdir(gd)) if f.endswith('.txt')
        ])

        real_mean, real_std = real_curves.mean(axis=0), real_curves.std(axis=0)
        gen_mean,  gen_std  = gen_curves.mean(axis=0),  gen_curves.std(axis=0)
        residual = real_mean - gen_mean

        ax0, ax1, ax2 = axes[r, 0], axes[r, 1], axes[r, 2]

        # 左：真實
        ax0.plot(wl, real_mean, color=color, lw=1.5, label='Mean')
        ax0.fill_between(wl, real_mean - real_std, real_mean + real_std,
                         alpha=0.25, color=color, label='±1 Std')
        ax0.set_title(f'{cls} — Real (n={len(real_curves)})', fontsize=10)
        ax0.set_ylabel('Intensity'); ax0.grid(alpha=0.3); ax0.legend(fontsize=10)

        # 中：生成
        ax1.plot(wl, gen_mean, color=color, lw=1.5, linestyle='--', label='Mean')
        ax1.fill_between(wl, gen_mean - gen_std, gen_mean + gen_std,
                         alpha=0.25, color=color, label='±1 Std')
        ax1.set_title(f'{cls} — CGAN Generated (n={len(gen_curves)})', fontsize=10)
        ax1.set_ylabel('Intensity'); ax1.grid(alpha=0.3); ax1.legend(fontsize=10)

        # 右：殘差
        ax2.axhline(0, color='gray', lw=0.8, linestyle=':')
        ax2.fill_between(wl, residual, 0,
                         where=(residual >= 0), alpha=0.5, color='tomato',   label='Real > Generated')
        ax2.fill_between(wl, residual, 0,
                         where=(residual <  0), alpha=0.5, color='steelblue', label='Generated > Real')
        ax2.plot(wl, residual, color='black', lw=0.8)
        mae = np.abs(residual).mean()
        ax2.set_title(f'{cls} — Residual (MAE={mae:.2f})', fontsize=10)
        ax2.set_ylabel('Δ Intensity'); ax2.grid(alpha=0.3); ax2.legend(fontsize=10)

        for ax in [ax0, ax1, ax2]:
            ax.set_xlabel('Wavelength (nm)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "curve_difference.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "curve_difference.pdf"), bbox_inches='tight')
    plt.close()
    print("  curve_difference.png")


# ── 5. 各類別獨立殘差圖（1×3，每類一張）────────────────────────────────────
def individual_curve_difference():
    CLASS_COLORS = {
        'motor_oil': '#dc2626', 'olive_oil': '#16a34a',
        'palm_oil':  '#d97706', 'lard':      '#7c3aed', 'water': '#2563eb',
    }
    for cls in CLASSES:
        color = CLASS_COLORS[cls]
        od = os.path.join(REAL_DIR, cls)
        gd = os.path.join(GEN_DIR, cls, "txt")
        real_curves = np.array([np.loadtxt(os.path.join(od, f))
                                 for f in sorted(os.listdir(od)) if f.endswith('_sg.txt')])
        gen_curves  = np.array([np.loadtxt(os.path.join(gd, f))
                                 for f in sorted(os.listdir(gd)) if f.endswith('.txt')])
        r_m, r_s = real_curves.mean(0), real_curves.std(0)
        g_m, g_s = gen_curves.mean(0),  gen_curves.std(0)
        res = r_m - g_m

        plt.rcParams.update({
            'font.size':       13,
            'axes.titlesize':  13,
            'axes.labelsize':  12,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 11,
        })
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(7, 10))
        ax0.plot(wl, r_m, color=color, lw=1.5, label='Mean')
        ax0.fill_between(wl, r_m-r_s, r_m+r_s, alpha=0.25, color=color, label='±1 Std')
        ax0.set_title(f'{_DISPLAY[cls]} — Real (n={len(real_curves)})')
        ax0.set_xlabel('Wavelength (nm)'); ax0.set_ylabel('Intensity')
        ax0.grid(alpha=0.3); ax0.legend()

        ax1.plot(wl, g_m, color=color, lw=1.5, linestyle='--', label='Mean')
        ax1.fill_between(wl, g_m-g_s, g_m+g_s, alpha=0.25, color=color, label='±1 Std')
        ax1.set_title(f'{_DISPLAY[cls]} — CGAN Generated (n={len(gen_curves)})')
        ax1.set_xlabel('Wavelength (nm)'); ax1.set_ylabel('Intensity')
        ax1.grid(alpha=0.3); ax1.legend()

        mae = np.abs(res).mean()
        ax2.axhline(0, color='gray', lw=0.8, linestyle=':')
        ax2.fill_between(wl, res, 0, where=(res>=0), alpha=0.5, color='tomato',    label='Real > Generated')
        ax2.fill_between(wl, res, 0, where=(res<0),  alpha=0.5, color='steelblue', label='Generated > Real')
        ax2.plot(wl, res, color='black', lw=0.8)
        ax2.set_title(f'{_DISPLAY[cls]} — Residual (MAE={mae:.2f})')
        ax2.set_xlabel('Wavelength (nm)'); ax2.set_ylabel('Δ Intensity')
        ax2.grid(alpha=0.3); ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{cls}_curve_difference.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, f"{cls}_curve_difference.pdf"), bbox_inches='tight')
        plt.close()
        print(f"  {cls}_curve_difference.png")


# ── 6. 各類別獨立 BMP 對比圖（1×4，每類一張）────────────────────────────────
def individual_bmp_comparison():
    for cls in CLASSES:
        od = os.path.join(REAL_DIR, cls)
        gd = os.path.join(GEN_DIR, cls, "bmp")
        of = sorted([f for f in os.listdir(od) if f.endswith('.bmp')])[:2]
        gf = sorted(os.listdir(gd))[:2]
        fps    = [os.path.join(od, f) for f in of] + [os.path.join(gd, f) for f in gf]
        titles = [f'{_DISPLAY[cls]} — Real', f'{_DISPLAY[cls]} — Real',
                  f'{_DISPLAY[cls]} — Generated', f'{_DISPLAY[cls]} — Generated']

        fig, axes = plt.subplots(1, 4, figsize=(7, 2.2))
        for ax, fp, t in zip(axes, fps, titles):
            ax.imshow(np.array(Image.open(fp).convert('L')), cmap='gray', aspect='auto')
            ax.set_title(t, fontsize=9)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{cls}_bmp_comparison.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, f"{cls}_bmp_comparison.pdf"), bbox_inches='tight')
        plt.close()
        print(f"  {cls}_bmp_comparison.png")


if __name__ == '__main__':
    print("產生對比圖...")
    individual_curves()            # 各類別獨立波形（真實×5 + 生成×5）
    bmp_comparison()               # 合併 BMP 對比（保留備用）
    individual_bmp_comparison()    # 各類別獨立 BMP（1×4）
    food_oils_compare()
    curve_difference()             # 合併殘差圖（保留備用）
    individual_curve_difference()  # 各類別獨立殘差圖（1×3）
    print("完成！")
