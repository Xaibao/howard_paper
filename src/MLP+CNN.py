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
from torchvision.models import resnet18

# 設置設備（優先使用 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自定義數據集，用於載入配對的 txt 和 bmp 檔案
class OilDataset(Dataset):
    def __init__(self, txt_dir, img_dir, transform=None, is_train=True):
        self.txt_dir = txt_dir
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train
        self.txt_files = []
        self.img_files = []
        self.labels = []
        self.classes = ["canolaoil", "oliveoil", "palmoil", "sunfloweroil"]

        # 載入檔案路徑和標籤
        for class_idx, class_name in enumerate(self.classes):
            if is_train:
                txt_pattern = re.compile(f"(generated_{class_name}_\\d+\\.txt|{class_name}_\\d+\\.txt)")
                img_pattern = re.compile(f"(generated_spectrum_{class_name}_\\d+\\.bmp|{class_name}_\\d+\\.bmp)")
            else:
                txt_pattern = re.compile(f"({class_name}_\\d+\\.txt|generated_{class_name}_\\d+\\.txt)")
                img_pattern = re.compile(f"({class_name}_\\d+\\.bmp|generated_spectrum_{class_name}_\\d+\\.bmp)")

            txt_files = [f for f in os.listdir(txt_dir) if txt_pattern.match(f)]
            img_files = [f for f in os.listdir(img_dir) if img_pattern.match(f)]

            # 確保檔案配對
            for txt in txt_files:
                idx = txt.split('_')[-1].split('.')[0]
                if is_train and 'generated' in txt:
                    img_name = f"generated_spectrum_{class_name}_{idx}.bmp"
                else:
                    img_name = f"{class_name}_{idx.zfill(4)}.bmp"
                    if img_name not in img_files:  # 嘗試無前導零
                        img_name = f"{class_name}_{idx}.bmp"
                if img_name in img_files:
                    self.txt_files.append(os.path.join(txt_dir, txt))
                    self.img_files.append(os.path.join(img_dir, img_name))
                    self.labels.append(class_idx)
                else:
                    print(f"配對失敗：{txt} 無對應圖像 {img_name}")

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
        # 標準化文本數據
        txt_data = (txt_data - txt_data.mean()) / (txt_data.std() + 1e-8)
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


# CNN 模型，用於處理 1280x400 像素圖像
class CNN(nn.Module):
    def __init__(self, output_size=128):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 160 * 50, output_size)  # 1280x400 -> 160x50 after three poolings

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)  # 640x200
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)  # 320x100
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)  # 160x50
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
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

# 融合模型，結合 MLP 和 CNN
class FusionModel(nn.Module):
    def __init__(self, mlp_input_size=1280, num_classes=4):
        super(FusionModel, self).__init__()
        self.mlp = MLP(input_size=mlp_input_size)
        self.cnn = CNN()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 + 128, num_classes)  # 拼接 MLP 和 CNN 的特徵

    def forward(self, txt, img):
        txt_features = self.mlp(txt)
        img_features = self.cnn(img)
        fused = torch.cat((txt_features, img_features), dim=1)
        output = self.fc(fused)
        return output


# 圖像數據的轉換
transform = transforms.Compose([
    transforms.Resize((400, 1280)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 初始化數據集和數據載入器
train_dataset = OilDataset(txt_dir=r"./train/txt", img_dir=r"./train/bmp", transform=transform, is_train=True) #訓練集路徑
test_dataset = OilDataset(txt_dir=r"./test/txt", img_dir=r"./test/bmp", transform=transform, is_train=False) #測試集路徑
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # 批次大小 16，適應 8GB 記憶體
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 初始化模型、損失函數和優化器
model = FusionModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # 降低學習率並添加權重衰減
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)  # 學習率調度器

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
best_loss = float('inf')
for epoch in range(num_epochs):
    # 訓練階段
    model.train()
    running_loss = 0.0
    for txt_data, img_data, labels in train_loader:
        txt_data, img_data, labels = txt_data.to(device), img_data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(txt_data, img_data)
        loss = criterion(outputs, labels)
        loss.backward()
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
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    print("混淆矩陣：")
    print(cm)
    class_accuracy = np.zeros(len(train_dataset.classes))
    for i in range(len(train_dataset.classes)):
        if cm.sum(axis=1)[i] > 0:
            class_accuracy[i] = 100 * cm.diagonal()[i] / cm.sum(axis=1)[i]
        else:
            class_accuracy[i] = 0  # 若類別無樣本，設置 Accuracy 為 0
    class_precision = 100 * precision_score(all_labels, all_preds, average=None, zero_division=0, labels=[0, 1, 2, 3])
    class_recall = 100 * recall_score(all_labels, all_preds, average=None, zero_division=0, labels=[0, 1, 2, 3])
    class_f1 = 100 * f1_score(all_labels, all_preds, average=None, zero_division=0, labels=[0, 1, 2, 3])

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

# 為每個類別生成 .txt 檔案
for i, class_name in enumerate(train_dataset.classes):
    with open(f"{class_name}_metrics_resnet18.txt", "w", encoding="utf-8") as f:
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
plt.figure(figsize=(12, 8))
for i, class_name in enumerate(train_dataset.classes):
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(test_class_accuracies) + 1), [acc[i] for acc in test_class_accuracies], label=f'{class_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Class Accuracy (%)')
    plt.title('Class Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(test_precisions) + 1), [prec[i] for prec in test_precisions], label=f'{class_name} Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision (%)')
    plt.title('Class Precision')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(test_recalls) + 1), [rec[i] for rec in test_recalls], label=f'{class_name} Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall (%)')
    plt.title('Class Recall')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(range(1, len(test_f1_scores) + 1), [f1[i] for f1 in test_f1_scores], label=f'{class_name} F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score (%)')
    plt.title('Class F1 Score')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig('class_metrics_curve_resnet18.png', dpi=300)
plt.close()


# 繪製 Train Loss 和 Test Loss 折線圖
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve_resnet18.png', dpi=300)
plt.close()

# 繪製 MAE 折線圖
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), test_maes, label='Test MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Test MAE Curve')
plt.legend()
plt.grid(True)
plt.savefig('mae_curve_resnet18.png', dpi=300)
plt.close()

# 繪製總體 Accuracy、Precision、Recall 和 F1 Score 折線圖
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Accuracy', color='#1f77b4')
plt.plot(range(1, len(test_precisions) + 1), [np.mean(prec) for prec in test_precisions], label='Precision', color='#ff7f0e')
plt.plot(range(1, len(test_recalls) + 1), [np.mean(rec) for rec in test_recalls], label='Recall', color='#2ca02c')
plt.plot(range(1, len(test_f1_scores) + 1), [np.mean(f1) for f1 in test_f1_scores], label='F1 Score', color='#d62728')
plt.xlabel('Epoch')
plt.ylabel('Percentage (%)')
plt.title('Overall Accuracy, Precision, Recall, and F1 Score')
plt.legend()
plt.grid(True)
plt.savefig('metrics_curve_resnet18.png', dpi=300)
plt.close()

# 繪製 PR 曲線（每個類別）
plt.figure(figsize=(10, 5))
for i in range(len(train_dataset.classes)):
    precision, recall, _ = precision_recall_curve(np.array(all_labels) == i, np.array(all_probs)[:, i])
    plt.plot(recall, precision, label=f'{train_dataset.classes[i]} (AUC = {auc(recall, precision):.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.savefig('pr_curve_resnet18.png', dpi=300)
plt.close()

# 繪製 ROC 曲線（每個類別）
plt.figure(figsize=(10, 5))
for i in range(len(train_dataset.classes)):
    fpr, tpr, _ = roc_curve(np.array(all_labels) == i, np.array(all_probs)[:, i])
    plt.plot(fpr, tpr, label=f'{train_dataset.classes[i]} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # 對角線
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.savefig('roc_curve_resnet18.png', dpi=300)
plt.close()

# 繪製混淆矩陣
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_resnet18.png', dpi=300)
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

# 保存模型
torch.save(model.state_dict(), "mlp_cnn_fusion_model_resnet18.pth")

