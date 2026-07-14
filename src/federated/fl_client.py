"""
FL Client — 跑在 Jetson Orin Nano（現場 Edge）或 4060（模擬）
載入本地真實水樣資料，本地訓練後只上傳 model weights，不傳原始資料

啟動（Jetson 或其他機器）：
  conda activate spectral
  python src/federated/fl_client.py \
      --server 192.168.50.212:8080 \
      --client-id taipei_01 \
      --classes motor_oil water

模擬兩個 client（在同一台機器測試）：
  # 終端機 1（模擬台北站）
  python src/federated/fl_client.py --server localhost:8080 --client-id taipei --classes motor_oil water
  # 終端機 2（模擬高雄站）
  python src/federated/fl_client.py --server localhost:8080 --client-id kaohsiung --classes olive_oil palm_oil lard
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import flwr as fl
from flwr.common import NDArrays, Scalar

from federated.fl_model import get_model, IMG_TRANSFORM, CLASSES

BASE_DIR = Path(__file__).parent.parent.parent
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL_EPOCHS = 5
BATCH_SIZE   = 4


# ── 本地真實資料集（data/augmented/）──────────────────────────
class LocalRealDataset(Dataset):
    """
    載入 data/augmented/{class}/ 下的真實水樣資料
    只載入 client 負責的 classes（代表該縣市的監測資料）
    """
    def __init__(self, augmented_dir: Path, target_classes: List[str]):
        self.txt_files = []
        self.img_files = []
        self.labels    = []

        for class_name in target_classes:
            if class_name not in CLASSES:
                print(f"[Client] 警告：未知類別 {class_name}，跳過")
                continue
            label    = CLASSES.index(class_name)
            class_dir = augmented_dir / class_name
            if not class_dir.exists():
                print(f"[Client] 找不到 {class_dir}，跳過")
                continue

            for bmp in sorted(class_dir.glob("*.bmp")):
                stem   = bmp.stem                             # e.g. motor_oil_0001
                sg_txt = class_dir / f"{stem}_sg.txt"
                if sg_txt.exists():
                    self.txt_files.append(sg_txt)
                    self.img_files.append(bmp)
                    self.labels.append(label)

        print(f"[Client] 本地資料集：{len(self.txt_files)} 筆")
        for cls in target_classes:
            lbl = CLASSES.index(cls) if cls in CLASSES else -1
            cnt = self.labels.count(lbl)
            print(f"  {cls}: {cnt} 筆")

        if len(self.txt_files) == 0:
            raise ValueError(f"本地資料集為空，請確認 {augmented_dir} 路徑正確")

    def __len__(self):
        return len(self.txt_files)

    def __getitem__(self, idx):
        with open(self.txt_files[idx]) as f:
            txt = np.array([float(x) for x in f.read().strip().split()], dtype=np.float32)
        txt = torch.tensor(txt / 255.0)

        img = Image.open(self.img_files[idx]).convert("RGB")
        img = IMG_TRANSFORM(img)

        return txt, img, self.labels[idx]


# ── Flower Client ────────────────────────────────────────────
class FOGClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, target_classes: List[str]):
        self.client_id = client_id
        self.model     = get_model(device=DEVICE)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)

        # 載入本地資料，80/20 切 train/val
        augmented_dir = BASE_DIR / "data" / "augmented"
        dataset       = LocalRealDataset(augmented_dir, target_classes)
        n_train       = max(1, int(0.8 * len(dataset)))
        n_val         = len(dataset) - n_train
        train_ds, val_ds = random_split(dataset, [n_train, n_val])

        self.train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

        print(f"[{client_id}] 訓練集: {n_train} | 驗證集: {n_val} | 裝置: {DEVICE}")

    def get_parameters(self, config) -> NDArrays:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: NDArrays):
        state = self.model.state_dict()
        for k, w in zip(state.keys(), parameters):
            state[k] = torch.tensor(w).to(DEVICE)
        self.model.load_state_dict(state)

    def fit(self, parameters: NDArrays, config) -> Tuple[NDArrays, int, Dict]:
        self.set_parameters(parameters)
        self.model.train()

        total_loss = 0.0
        total_samples = 0
        for epoch in range(LOCAL_EPOCHS):
            epoch_loss = 0.0
            for txt, img, labels in self.train_loader:
                txt, img, labels = txt.to(DEVICE), img.to(DEVICE), labels.to(DEVICE)
                self.optimizer.zero_grad()
                outputs = self.model(txt, img)
                loss    = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(labels)
                total_samples += len(labels)

            total_loss += epoch_loss
            print(f"  [{self.client_id}] Epoch {epoch+1}/{LOCAL_EPOCHS} Loss: {epoch_loss/len(self.train_loader.dataset):.4f}")

        avg_loss = total_loss / (LOCAL_EPOCHS * max(total_samples, 1))
        return self.get_parameters(config={}), total_samples, {"train_loss": float(avg_loss)}

    def evaluate(self, parameters: NDArrays, config) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        self.model.eval()

        correct = 0
        total   = 0
        val_loss = 0.0
        with torch.no_grad():
            for txt, img, labels in self.val_loader:
                txt, img, labels = txt.to(DEVICE), img.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(txt, img)
                loss    = self.criterion(outputs, labels)
                val_loss += loss.item() * len(labels)
                preds   = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += len(labels)

        accuracy = correct / max(total, 1)
        avg_loss = val_loss / max(total, 1)
        print(f"  [{self.client_id}] Eval → Accuracy: {accuracy:.4f} | Loss: {avg_loss:.4f}")
        return float(avg_loss), total, {"accuracy": float(accuracy)}


def parse_args():
    parser = argparse.ArgumentParser(description="FOG FL Client")
    parser.add_argument("--server", default="localhost:8080",
                        help="FL server 地址（預設 localhost:8080）")
    parser.add_argument("--client-id", default="client_01",
                        help="客戶端 ID（代表監測站名稱）")
    parser.add_argument("--classes", nargs="+",
                        default=["motor_oil", "olive_oil", "palm_oil", "lard", "water"],
                        choices=CLASSES,
                        help="此客戶端負責的污染物類別（代表本地資料）")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"=== FOG FL Client ===")
    print(f"監測站 ID : {args.client_id}")
    print(f"FL Server : {args.server}")
    print(f"本地類別  : {args.classes}")
    print(f"本地訓練  : {LOCAL_EPOCHS} epochs/輪")

    client = FOGClient(
        client_id=args.client_id,
        target_classes=args.classes,
    )

    fl.client.start_numpy_client(
        server_address=args.server,
        client=client,
    )

    print(f"[{args.client_id}] FL 訓練完成")


if __name__ == "__main__":
    main()
