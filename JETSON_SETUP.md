# JETSON_SETUP.md — Jetson Orin Nano 部署指南

> 目標：把 MLP-Transformer 模型部署到 Jetson Orin Nano，實現 < 1 秒邊緣推論
> Jetson Tailscale IP：100.102.243.109
> 3060 Tailscale IP：100.108.78.9

---

## Step 1：從 3060 設定 SSH 免密鑰登入

在 **3060** 上執行：

```bash
# 生成 SSH key（如果還沒有）
ls ~/.ssh/id_rsa.pub 2>/dev/null || ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa

# 複製 key 到 Jetson（輸入一次 Jetson 的密碼）
ssh-copy-id howard@100.102.243.109

# 測試免密碼 SSH
ssh howard@100.102.243.109 "echo OK"
```

---

## Step 2：確認 Jetson 系統環境

```bash
ssh howard@100.102.243.109
```

進入 Jetson 後執行：

```bash
# 確認是 Jetson
cat /etc/nv_tegra_release

# 確認 Python
python3 --version     # 目標：3.8+

# 確認有沒有 pip
pip3 --version

# 確認 GPU（Jetson 用 NVIDIA Tegra）
nvidia-smi 2>/dev/null || tegrastats
```

---

## Step 3：在 Jetson 安裝 Python 套件

```bash
# 基本套件
pip3 install numpy matplotlib pillow scikit-learn

# PyTorch for Jetson（Jetson 不能用一般 pip 裝 torch，要用官方 wheel）
# 先確認 JetPack 版本
cat /etc/nv_tegra_release | grep "R"
```

**根據 JetPack 版本選擇 PyTorch wheel**：

| JetPack | PyTorch wheel |
|---------|--------------|
| 6.x | https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/ |
| 5.x | https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/ |

```bash
# 下載後安裝（以 JetPack 6 為例）
pip3 install torch-*.whl torchvision-*.whl
```

---

## Step 4：從 3060 傳送模型與推論程式

在 **3060** 上執行：

```bash
BASE=/home/t1204-3060/Howard/spectral_monitor
JETSON=howard@100.102.243.109

# 建立 Jetson 上的目錄
ssh $JETSON "mkdir -p ~/spectral_monitor/models ~/spectral_monitor/src"

# 傳送模型（epoch 100，論文使用版本）
scp $BASE/models/mlp_transformer_v2.pth $JETSON:~/spectral_monitor/models/

# 傳送推論腳本
scp $BASE/src/edge_inference.py $JETSON:~/spectral_monitor/src/

# （如果有測試樣本也一起傳）
scp $BASE/data_paper/real_source/real_test/motor_oil/20250101_000001_sg.txt \
    $JETSON:~/spectral_monitor/test_sample.txt 2>/dev/null || true
```

---

## Step 5：建立 Edge 推論腳本

在 **3060** 上建立 `src/edge_inference.py`，然後 scp 到 Jetson：

```python
"""
edge_inference.py — Jetson Orin Nano 邊緣推論腳本
量測推論時間，確認 < 1 秒
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import time
import os

MODEL_PATH = os.path.expanduser("~/spectral_monitor/models/mlp_transformer_v2.pth")
CLASSES    = ["motor_oil", "olive_oil", "palm_oil", "lard", "water"]
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"裝置：{device}")

# ── 模型架構（與 3060 訓練版本一致）──────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_size=1280, hidden_size=512, output_size=128):
        super().__init__()
        self.fc1     = nn.Linear(input_size, hidden_size)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2     = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class TransformerBranch(nn.Module):
    def __init__(self, output_size=128):
        super().__init__()
        self.patch_size  = 40
        self.embed_dim   = 512
        self.patch_embed = nn.Conv2d(3, self.embed_dim,
                                     kernel_size=self.patch_size, stride=self.patch_size)
        encoder_layer    = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=8,
                                                       batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fc          = nn.Linear(self.embed_dim, output_size)
    def forward(self, x):
        x = self.patch_embed(x).flatten(2).permute(0, 2, 1)
        return self.fc(self.transformer(x).mean(dim=1))

class TransformerFusionModel(nn.Module):
    def __init__(self, mlp_input_size=1280, num_classes=5):
        super().__init__()
        self.mlp               = MLP(input_size=mlp_input_size)
        self.transformer_branch = TransformerBranch(output_size=128)
        self.fc                 = nn.Linear(256, num_classes)
    def forward(self, txt, img):
        return self.fc(torch.cat([self.mlp(txt), self.transformer_branch(img)], dim=1))

# ── 載入模型 ──────────────────────────────────────────────────────────────────
model = TransformerFusionModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model.eval()
print(f"模型載入完成：{MODEL_PATH}")

transform = transforms.Compose([
    transforms.Resize((400, 1280)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def infer(txt_path, bmp_path):
    """單次推論，回傳（類別, 信心度%, 推論時間ms）"""
    raw = np.loadtxt(txt_path, dtype=np.float32)
    if len(raw) != 1280:
        raw = np.interp(np.linspace(0, len(raw)-1, 1280),
                        np.arange(len(raw)), raw).astype(np.float32)
    txt_t = torch.tensor(raw / 255.0).unsqueeze(0).to(device)

    img   = Image.open(bmp_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(txt_t, img_t)
        probs  = torch.softmax(logits, dim=1)
    t1 = time.perf_counter()

    pred = probs.argmax().item()
    conf = probs.max().item() * 100
    ms   = (t1 - t0) * 1000
    return CLASSES[pred], conf, ms

# ── 測試推論（用假資料熱身 + 正式量測）──────────────────────────────────────
print("\n[熱身] 第一次推論（含模型 JIT 初始化，不計入）")
dummy_txt = torch.zeros(1, 1280).to(device)
dummy_img = torch.zeros(1, 3, 400, 1280).to(device)
with torch.no_grad():
    _ = model(dummy_txt, dummy_img)

print("[正式量測] 10 次推論取平均：")
times = []
for i in range(10):
    dummy_txt = torch.rand(1, 1280).to(device)
    dummy_img = torch.rand(1, 3, 400, 1280).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy_txt, dummy_img)
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000
    times.append(ms)
    print(f"  Run {i+1:2d}: {ms:.1f} ms")

print(f"\n平均推論時間：{np.mean(times):.1f} ms")
print(f"最大推論時間：{np.max(times):.1f} ms")
print(f"目標：< 1000 ms → {'✓ 達標' if np.max(times) < 1000 else '✗ 未達標'}")
```

---

## Step 6：在 Jetson 上執行推論測試

```bash
ssh howard@100.102.243.109
cd ~/spectral_monitor
python3 src/edge_inference.py
```

**預期輸出**：
```
裝置：cuda
模型載入完成：...
[正式量測] 10 次推論取平均：
  Run  1: 35.2 ms
  ...
平均推論時間：38.5 ms
目標：< 1000 ms → ✓ 達標
```

把這個輸出截圖或複製起來，**平均推論時間就是論文寫的數字**。

---

## Step 7：把推論時間記錄下來

跑完後執行：

```bash
python3 src/edge_inference.py 2>&1 | tee ~/spectral_monitor/inference_timing.txt
```

然後從 3060 拿回來：

```bash
scp howard@100.102.243.109:~/spectral_monitor/inference_timing.txt \
    /home/t1204-3060/Howard/spectral_monitor/outputs/jetson_inference_timing.txt
```

---

## 常見問題

**Q：Jetson 上 torch.cuda.is_available() 回傳 False？**
→ 確認 PyTorch 版本是 Jetson 官方 wheel，不是一般 pip 版本。

**Q：模型載入很慢？**
→ 第一次載入需要 10-30 秒，之後快取後很快。

**Q：推論時間超過 1 秒？**
→ 確認使用 cuda device，不要用 cpu。

---

## 完成後更新論文數據

把推論時間填入論文對應位置：
- Methods 章節：「Edge inference latency: X ms (average), X ms (maximum)」
- Table X：Inference Time 欄位

同時更新 `reviewer_responses.txt` 中關於 Adaptive AI 的 Inference Time 數字。
