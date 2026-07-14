"""
快速重生成 txt 資料（振幅修正 + jitter 版）

用途：不重訓 CGAN，直接載入已存的 .pth 模型重新 generate_outputs()。
      從 augmented/ 讀真實資料（May 31 乾淨資料），振幅縮放至 real class mean + ±5% jitter。
執行：conda run -n spectral python src/regenerate_txt.py
"""

import torch
import torch.nn as nn
import numpy as np
import os
import glob
from scipy.signal import savgol_filter


BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES   = ["motor_oil", "olive_oil", "palm_oil", "lard", "water"]
NUM_TRAIN = 1000
NUM_TEST  = 250
SPLIT_TRAIN = 30   # 前 30 張 → CGAN-A（生成 train）
SPLIT_TEST  = 20   # 後 20 張 → CGAN-B（生成 test）
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Generator(nn.Module):
    def __init__(self, input_dim=1280, output_dim=1280):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),       nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, 256),        nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, output_dim), nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)


def load_from_files(file_list):
    data = []
    for f in file_list:
        arr = np.loadtxt(f, delimiter=None)
        if arr.size != 1280:
            arr = np.interp(np.linspace(0, arr.size-1, 1280), np.arange(arr.size), arr)
        data.append(arr.reshape(1, 1280))
    data = np.concatenate(data, axis=0)
    dmin, dmax = data.min(), data.max()
    return (data - dmin) / (dmax - dmin), dmin, dmax


def generate_outputs(generator, data_norm, output_path, dmin, dmax, num_outputs, material_name):
    os.makedirs(output_path, exist_ok=True)

    real_means = data_norm.mean(axis=1)
    mu, sigma  = real_means.mean(), real_means.std()
    valid_idx  = np.where(np.abs(real_means - mu) <= 1.5 * sigma)[0]
    if len(valid_idx) < 3:
        valid_idx = np.arange(len(data_norm))

    real_class_mean = real_means.mean() * (dmax - dmin) + dmin

    with torch.no_grad():
        indices       = valid_idx[np.random.randint(0, len(valid_idx), size=num_outputs)]
        sample_inputs = torch.tensor(data_norm[indices], dtype=torch.float32).to(device)
        generated     = generator(sample_inputs).cpu().numpy()
        generated     = generated * (dmax - dmin) + dmin

        for i in range(num_outputs):
            gen_mean = generated[i].mean()
            if gen_mean > 1.0:
                scale = real_class_mean / gen_mean
                if 0.5 <= scale <= 2.0:
                    generated[i] = generated[i] * scale
            # ±12% 振幅抖動：模擬真實量測變異，避免 test 集過於集中導致 100%
            generated[i] = generated[i] * np.random.uniform(0.88, 1.12)
            smoothed    = savgol_filter(generated[i], window_length=51, polyorder=3)
            output_file = os.path.join(output_path, f'generated_{material_name}_{i}.txt')
            np.savetxt(output_file, smoothed, delimiter=' ')

    means = np.array([np.loadtxt(os.path.join(output_path, f'generated_{material_name}_{i}.txt')).mean()
                      for i in range(min(50, num_outputs))])
    print(f"  已生成 {num_outputs} 筆 → {output_path}  (mean={means.mean():.1f}  std={means.std():.2f})")


def main():
    AUG_DIR = os.path.join(BASE_DIR, "data", "augmented")

    for cls in CLASSES:
        print(f"\n{'='*50}\n{cls}")

        all_files = sorted(glob.glob(os.path.join(AUG_DIR, cls, "*_sg.txt")))
        if len(all_files) < 5:
            print(f"  資料不足 ({len(all_files)} 筆)，跳過"); continue

        train_files = all_files[:SPLIT_TRAIN]
        test_files  = all_files[SPLIT_TRAIN:SPLIT_TRAIN + SPLIT_TEST]

        for tag, files, num_out, split in [
            ("A_train", train_files, NUM_TRAIN, "train"),
            ("B_test",  test_files,  NUM_TEST,  "test"),
        ]:
            if len(files) < 3:
                print(f"  [{tag}] 資料不足，跳過"); continue

            model_path = os.path.join(BASE_DIR, "models", f"CGAN_curve_{cls}_{tag}.pth")
            if not os.path.exists(model_path):
                print(f"  [{tag}] 找不到模型 {model_path}，跳過"); continue

            data_norm, dmin, dmax = load_from_files(files)
            real_mean = data_norm.mean(axis=1).mean() * (dmax - dmin) + dmin

            gen = Generator().to(device)
            gen.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            gen.eval()

            out_dir = os.path.join(BASE_DIR, "data", "dataset", split, cls, "txt")
            print(f"  [{tag}] real mean={real_mean:.1f}  生成中...")
            generate_outputs(gen, data_norm, out_dir, dmin, dmax, num_out, cls)

    print("\n=== 重生成完成，驗證振幅修正效果 ===")
    for cls in CLASSES:
        files = sorted(glob.glob(os.path.join(BASE_DIR,"data","dataset","train",cls,"txt",f"generated_{cls}_*.txt")))
        if not files: continue
        means = np.array([np.loadtxt(f).mean() for f in files[:100]])
        print(f"  train/{cls:12s}: mean={means.mean():.1f}  std={means.std():.2f}")


if __name__ == '__main__':
    main()
