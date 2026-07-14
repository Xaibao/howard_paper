import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import re
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
from sklearn.metrics import precision_recall_curve, roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix
from torchvision import transforms

# 設置設備（優先使用 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DISPLAY = {"motor_oil":"Motor Oil","olive_oil":"Olive Oil",
            "palm_oil":"Palm Oil","lard":"Lard","water":"Water"}

# 真實資料 Dataset（timestamp 命名，flat 結構）
class RealDataset(Dataset):
    def __init__(self, root_dir, transform=None, per_class=9999):
        self.transform = transform
        self.samples = []
        classes = ["motor_oil", "olive_oil", "palm_oil", "lard", "water"]
        for label, cls in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                print(f"警告：找不到 {cls_dir}"); continue
            found = []
            for f in sorted(os.listdir(cls_dir)):
                if not f.endswith("_sg.txt"): continue
                base = f[:-7]
                txt_p = os.path.join(cls_dir, f)
                img_p = os.path.join(cls_dir, f"{base}.bmp")
                if os.path.exists(img_p):
                    found.append((txt_p, img_p, label))
            self.samples.extend(found[:per_class])

        print(f"數據集大小：{len(self.samples)}")
        print("類別分佈：")
        for i, cls in enumerate(classes):
            n = sum(1 for _, _, l in self.samples if l == i)
            print(f"{cls}: {n} 樣本")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        txt_p, img_p, label = self.samples[idx]
        with open(txt_p) as f:
            txt = np.array([float(x) for x in f.read().strip().split()], dtype=np.float32)
        if len(txt) != 1280:
            txt = np.interp(np.linspace(0, len(txt)-1, 1280), np.arange(len(txt)), txt).astype(np.float32)
        txt = torch.tensor(txt / 255.0)
        img = Image.open(img_p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return txt, img, label

# 自定義數據集，用於載入配對的 txt 和 bmp 檔案
class OilDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.txt_files = []
        self.img_files = []
        self.labels    = []
        self.classes   = ["motor_oil", "olive_oil", "palm_oil", "lard", "water"]

        for class_idx, class_name in enumerate(self.classes):
            txt_dir = os.path.join(root_dir, class_name, "txt")
            img_dir = os.path.join(root_dir, class_name, "bmp")
            if not os.path.isdir(txt_dir) or not os.path.isdir(img_dir):
                print(f"警告：找不到 {class_name}/txt 或 /bmp"); continue

            pat = re.compile(rf"^generated_{re.escape(class_name)}_(\d+)\.txt$")
            for f in sorted(os.listdir(txt_dir)):
                m = pat.match(f)
                if not m: continue
                n     = m.group(1)
                txt_p = os.path.join(txt_dir, f)
                img_p = os.path.join(img_dir, f"generated_spectrum_{class_name}_{n}.bmp")
                if os.path.exists(img_p):
                    self.txt_files.append(txt_p)
                    self.img_files.append(img_p)
                    self.labels.append(class_idx)
                else:
                    print(f"配對失敗：{f}")

        print(f"數據集大小：{len(self.txt_files)}")
        print("類別分佈：")
        for i, class_name in enumerate(self.classes):
            class_count = sum(1 for label in self.labels if label == i)
            print(f"{class_name}: {class_count} 樣本")
        if len(self.txt_files) == 0:
            raise ValueError("數據集為空，請檢查檔案路徑和命名")

    def __len__(self):
        return len(self.txt_files)

    def __getitem__(self, idx):
        # 載入文本數據（1280 維光強度陣列）
        with open(self.txt_files[idx], 'r') as f:
            txt_data = np.array([float(x) for x in f.read().strip().split()], dtype=np.float32)
            if len(txt_data) != 1280:
                raise ValueError(f"Text file {self.txt_files[idx]} does not contain exactly 1280 values.")
        # 固定縮放（保留振幅差異，用於區分相似油脂；不做逐樣本標準化）
        txt_data = txt_data / 255.0
        txt_data = torch.tensor(txt_data)

        # 載入圖像數據
        img = Image.open(self.img_files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return txt_data, img, label


# MLP 模型，用於處理 1280 維文本數據
class MLP(nn.Module):
    def __init__(self, input_size=1280, hidden_size=512, output_size=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Transformer 分支模型，用於處理 1280x400 像素圖像
# Transformer 分支：處理影像數據
class TransformerBranch(nn.Module):
    def __init__(self, output_size=128):
        super(TransformerBranch, self).__init__()
        self.patch_size = 40 
        self.embed_dim = 512 
        
        # 使用卷積進行 Patch Embedding
        self.patch_embed = nn.Conv2d(3, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        # Transformer 層，確保 batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.fc = nn.Linear(self.embed_dim, output_size)

    def forward(self, x):
        x = self.patch_embed(x)  # (Batch, 512, 10, 32)
        x = x.flatten(2)         # (Batch, 512, 320)
        x = x.permute(0, 2, 1)   # (Batch, 320, 512)
        x = self.transformer(x)
        x = x.mean(dim=1)        # 全局平均池化
        x = self.fc(x)
        return x
# class CNN(nn.Module):
#     def __init__(self, output_size=128):
#         super(CNN, self).__init__()
#         self.resnet = resnet18(pretrained=True)
#         self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_size)
#
#     def forward(self, x):
#         return self.resnet(x)

# 融合模型：結合 MLP 與 Transformer
class TransformerFusionModel(nn.Module):
    def __init__(self, mlp_input_size=1280, num_classes=5):
        super(TransformerFusionModel, self).__init__()
        self.mlp = MLP(input_size=mlp_input_size)
        self.transformer_branch = TransformerBranch(output_size=128) # 這裡要對應 TransformerBranch 的定義
        self.fc = nn.Linear(128 + 128, num_classes)

    def forward(self, txt, img):
        txt_features = self.mlp(txt)
        img_features = self.transformer_branch(img)
        fused = torch.cat((txt_features, img_features), dim=1)
        output = self.fc(fused)
        return output

transform = transforms.Compose([
    transforms.Resize((400, 1280)),           # 縮放至模型要求的尺寸
    transforms.ToTensor(),                   # 轉為 Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 標準化
])

# 初始化數據集和數據載入器
train_dataset = OilDataset(os.path.join(BASE_DIR, "data", "dataset", "train"), transform=transform)
test_dataset  = RealDataset(
    root_dir  = os.path.join(BASE_DIR, "data_paper", "real_source", "real_test"),
    transform = transform,
    per_class = 250
)
train_loader  = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=2)
test_loader   = DataLoader(test_dataset,  batch_size=8, shuffle=False, num_workers=2)

# 初始化模型、損失函數和優化器

model = TransformerFusionModel().to(device) # 改成你定義的新名稱
# 加重 CanolaOil 的懲罰權重
weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# 儲存評估指標
train_losses = []
test_losses = []
test_accuracies = []
test_class_accuracies = []
test_precisions = []
test_recalls = []
test_f1_scores = []
test_maes = []

# 訓練與測試迴圈
num_epochs = 100
best_acc = 0.0
for epoch in range(num_epochs):
    # 訓練階段
    model.train()
    running_loss = 0.0
    for txt_data, img_data, labels in train_loader:
        txt_data, img_data, labels = txt_data.to(device), img_data.to(device), labels.to(device)
        # 量測雜訊：模擬感測器隨機變異
        txt_data = txt_data + torch.randn_like(txt_data) * 0.02
        # 5% 機率注入隨機光譜＋隨機標籤，模擬真實部署時遇到非目標物質
        corrupt_mask = torch.rand(txt_data.size(0), device=device) < 0.05
        if corrupt_mask.any():
            n = corrupt_mask.sum().item()
            txt_data[corrupt_mask] = torch.rand(n, 1280, device=device)
            labels[corrupt_mask] = torch.randint(0, 5, (n,), device=device)
        optimizer.zero_grad()
        outputs = model(txt_data, img_data)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 測試階段
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    mae_sum = 0.0
    with torch.no_grad():
        for txt_data, img_data, labels in test_loader:
            txt_data, img_data, labels = txt_data.to(device), img_data.to(device), labels.to(device)
            # 5% 測試時也注入隨機樣本，與訓練一致
            corrupt_mask = torch.rand(txt_data.size(0), device=device) < 0.05
            if corrupt_mask.any():
                n = corrupt_mask.sum().item()
                txt_data[corrupt_mask] = torch.rand(n, 1280, device=device)
                labels[corrupt_mask] = torch.randint(0, 5, (n,), device=device)
            outputs = model(txt_data, img_data)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            probs = torch.softmax(outputs, dim=1)
            true_one_hot = torch.zeros_like(probs).scatter_(1, labels.unsqueeze(1), 1.0)
            mae_sum += torch.mean(torch.abs(probs - true_one_hot)).item() * labels.size(0)
            all_probs.extend(probs.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    test_accuracy = 100 * correct / total if total > 0 else 0
    test_accuracies.append(test_accuracy)

    # 計算每個類別的指標
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3, 4])
    print("混淆矩陣：")
    print(cm)
    class_accuracy = np.zeros(len(train_dataset.classes))
    for i in range(len(train_dataset.classes)):
        if cm.sum(axis=1)[i] > 0:
            class_accuracy[i] = 100 * cm.diagonal()[i] / cm.sum(axis=1)[i]
        else:
            class_accuracy[i] = 0  # 若類別無樣本，設置 Accuracy 為 0
    class_precision = 100 * precision_score(all_labels, all_preds, average=None, zero_division=0, labels=[0, 1, 2, 3, 4])
    class_recall = 100 * recall_score(all_labels, all_preds, average=None, zero_division=0, labels=[0, 1, 2, 3, 4])
    class_f1 = 100 * f1_score(all_labels, all_preds, average=None, zero_division=0, labels=[0, 1, 2, 3, 4])

    test_class_accuracies.append(class_accuracy)
    test_precisions.append(class_precision)
    test_recalls.append(class_recall)
    test_f1_scores.append(class_f1)
    test_mae = mae_sum / total if total > 0 else 0
    test_maes.append(test_mae)

    # 輸出每個類別的指標
    print(f"第 {epoch + 1} 輪：")
    print(f"訓練損失：{avg_train_loss:.4f}")
    print(f"測試損失：{avg_test_loss:.4f}")
    print(f"總體測試準確率：{test_accuracy:.2f}%")
    for i, class_name in enumerate(train_dataset.classes):
        print(
            f"{class_name} - 準確率：{class_accuracy[i]:.2f}%, 精確率：{class_precision[i]:.2f}%, 召回率：{class_recall[i]:.2f}%, F1 分數：{class_f1[i]:.2f}%")
    print(f"測試 MAE：{test_mae:.4f}")
    scheduler.step()
    if test_accuracy > best_acc:
        best_acc = test_accuracy
        torch.save(model.state_dict(), os.path.join(BASE_DIR, "models", "best_mlp_transformer_v2.pth"))
        print(f"  ★ 最佳模型更新：{best_acc:.2f}%（已存 best_mlp_transformer_v2.pth）")

# 為每個類別生成 .txt 檔案
for i, class_name in enumerate(train_dataset.classes):
    with open(os.path.join(BASE_DIR, "outputs", f"{class_name}_metrics_mlptransformer.txt"), "w", encoding="utf-8") as f:
        f.write("Epoch | Accuracy | Precision | Recall | F1 Score\n")
        f.write("-" * 50 + "\n")
        for epoch in range(len(test_class_accuracies)):
            f.write(f"{epoch + 1:<5} | {test_class_accuracies[epoch][i]:.2f}% | {test_precisions[epoch][i]:.2f}% | {test_recalls[epoch][i]:.2f}% | {test_f1_scores[epoch][i]:.2f}%\n")
        # 計算平均值
        avg_accuracy = np.mean([acc[i] for acc in test_class_accuracies])
        avg_precision = np.mean([prec[i] for prec in test_precisions])
        avg_recall = np.mean([rec[i] for rec in test_recalls])
        avg_f1 = np.mean([f1[i] for f1 in test_f1_scores])
        f.write("-" * 50 + "\n")
        f.write(f"Avg   | {avg_accuracy:.2f}% | {avg_precision:.2f}% | {avg_recall:.2f}% | {avg_f1:.2f}%\n")

# 繪製每個類別的 Accuracy、Precision、Recall 和 F1 Score 折線圖
plt.figure(figsize=(7, 6))
for i, class_name in enumerate(train_dataset.classes):
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(test_class_accuracies) + 1), [acc[i] for acc in test_class_accuracies], label=f'{_DISPLAY.get(class_name, class_name)} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Class Accuracy (%)')
    plt.title('Class Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(test_precisions) + 1), [prec[i] for prec in test_precisions], label=f'{_DISPLAY.get(class_name, class_name)} Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision (%)')
    plt.title('Class Precision')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(test_recalls) + 1), [rec[i] for rec in test_recalls], label=f'{_DISPLAY.get(class_name, class_name)} Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall (%)')
    plt.title('Class Recall')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(range(1, len(test_f1_scores) + 1), [f1[i] for f1 in test_f1_scores], label=f'{_DISPLAY.get(class_name, class_name)} F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score (%)')
    plt.title('Class F1 Score')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "outputs", "class_metrics_curve_mlptransformer.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, "outputs", "class_metrics_curve_mlptransformer.pdf"), bbox_inches='tight')
plt.close()


# 繪製 Train Loss 和 Test Loss 折線圖
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(train_losses) + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(BASE_DIR, "outputs", "loss_curve_mlptransformer.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, "outputs", "loss_curve_mlptransformer.pdf"), bbox_inches='tight')
plt.close()

# 繪製 MAE 折線圖
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(train_losses) + 1), test_maes, label='Test MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Test MAE Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(BASE_DIR, "outputs", "mae_curve_mlptransformer.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, "outputs", "mae_curve_mlptransformer.pdf"), bbox_inches='tight')
plt.close()

# 繪製總體 Accuracy、Precision、Recall 和 F1 Score 折線圖
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Accuracy', color='#1f77b4')
plt.plot(range(1, len(test_precisions) + 1), [np.mean(prec) for prec in test_precisions], label='Precision', color='#ff7f0e')
plt.plot(range(1, len(test_recalls) + 1), [np.mean(rec) for rec in test_recalls], label='Recall', color='#2ca02c')
plt.plot(range(1, len(test_f1_scores) + 1), [np.mean(f1) for f1 in test_f1_scores], label='F1 Score', color='#d62728')
plt.xlabel('Epoch')
plt.ylabel('Percentage (%)')
plt.title('Overall Accuracy, Precision, Recall, and F1 Score')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(BASE_DIR, "outputs", "metrics_curve_mlptransformer.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, "outputs", "metrics_curve_mlptransformer.pdf"), bbox_inches='tight')
plt.close()

# 繪製 PR 曲線（每個類別）
plt.figure(figsize=(7, 4))
for i in range(len(train_dataset.classes)):
    precision, recall, _ = precision_recall_curve(np.array(all_labels) == i, np.array(all_probs)[:, i])
    plt.plot(recall, precision, label=f'{_DISPLAY.get(train_dataset.classes[i], train_dataset.classes[i])} (AUC = {auc(recall, precision):.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(BASE_DIR, "outputs", "pr_curve_mlptransformer.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, "outputs", "pr_curve_mlptransformer.pdf"), bbox_inches='tight')
plt.close()

# 繪製 ROC 曲線（每個類別）
plt.figure(figsize=(7, 4))
for i in range(len(train_dataset.classes)):
    fpr, tpr, _ = roc_curve(np.array(all_labels) == i, np.array(all_probs)[:, i])
    plt.plot(fpr, tpr, label=f'{_DISPLAY.get(train_dataset.classes[i], train_dataset.classes[i])} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # 對角線
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(BASE_DIR, "outputs", "roc_curve_mlptransformer.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, "outputs", "roc_curve_mlptransformer.pdf"), bbox_inches='tight')
plt.close()

# 繪製混淆矩陣
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(7, 5.5))
_disp_labels = [_DISPLAY.get(c, c) for c in train_dataset.classes]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=_disp_labels, yticklabels=_disp_labels,
            annot_kws={"size": 12}, square=True)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(BASE_DIR, "outputs", "confusion_matrix_mlptransformer.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(BASE_DIR, "outputs", "confusion_matrix_mlptransformer.pdf"), bbox_inches='tight')
plt.close()

# 測試單一測試樣本的函數
def test_single_sample(txt_path, img_path, model, transform, classes):
    model.eval()
    # 載入並預處理文本數據
    with open(txt_path, 'r') as f:
        txt_data = np.array([float(x) for x in f.read().strip().split()], dtype=np.float32)
        if len(txt_data) != 1280:
            raise ValueError(f"Text file {txt_path} does not contain exactly 1280 values.")
    txt_data = (txt_data - txt_data.mean()) / (txt_data.std() + 1e-8)
    txt_data = torch.tensor(txt_data).unsqueeze(0).to(device)

    # 載入並預處理圖像數據
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(txt_data, img)
        _, predicted = torch.max(output, 1)
        predicted_class = classes[predicted.item()]

    return predicted_class

# 保存模型（最終epoch → mlp_transformer_v2.pth；最佳epoch → best_mlp_transformer_v2.pth）
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
torch.save(model.state_dict(), os.path.join(BASE_DIR, "models", "mlp_transformer_v2.pth"))
print(f"最終模型已存：mlp_transformer_v2.pth（epoch 100）")
print(f"最佳模型在：best_mlp_transformer_v2.pth（{best_acc:.2f}%）")


