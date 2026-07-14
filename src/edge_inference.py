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

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "mlp_transformer_v2.pth")
CLASSES = ["motor_oil", "olive_oil", "palm_oil", "lard", "water"]
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"裝置：{device}")

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
        self.mlp                = MLP(input_size=mlp_input_size)
        self.transformer_branch = TransformerBranch(output_size=128)
        self.fc                 = nn.Linear(256, num_classes)
    def forward(self, txt, img):
        return self.fc(torch.cat([self.mlp(txt), self.transformer_branch(img)], dim=1))

model = TransformerFusionModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model.eval()
print(f"模型載入完成：{MODEL_PATH}\n")

transform = transforms.Compose([
    transforms.Resize((400, 1280)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 熱身（不計入計時）
print("[熱身] 第一次推論（含 CUDA 初始化，不計入）")
with torch.no_grad():
    _ = model(torch.zeros(1, 1280).to(device), torch.zeros(1, 3, 400, 1280).to(device))

# 正式量測 10 次
print("[正式量測] 10 次推論：")
times = []
for i in range(10):
    txt_t = torch.rand(1, 1280).to(device)
    img_t = torch.rand(1, 3, 400, 1280).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(txt_t, img_t)
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000
    times.append(ms)
    print(f"  Run {i+1:2d}: {ms:.2f} ms")

avg_ms = np.mean(times)
max_ms = np.max(times)
print(f"\n平均推論時間：{avg_ms:.2f} ms")
print(f"最大推論時間：{max_ms:.2f} ms")
print(f"目標 < 1000 ms → {'✓ 達標' if max_ms < 1000 else '✗ 未達標'}")
print(f"\n→ 論文填入數字：Inference latency = {avg_ms:.1f} ms (avg), {max_ms:.1f} ms (max)")
