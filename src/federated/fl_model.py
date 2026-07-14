"""
FL 共用模組：TransformerFusionModel 定義
Server 和 Client 都 import 這個，確保架構一致
"""
import torch
import torch.nn as nn
from torchvision import transforms

CLASSES = ["motor_oil", "olive_oil", "palm_oil", "lard", "water"]
NUM_CLASSES = len(CLASSES)

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((400, 1280)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


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
        self.patch_embed = nn.Conv2d(3, 512, kernel_size=40, stride=40)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fc          = nn.Linear(512, output_size)

    def forward(self, x):
        x = self.patch_embed(x)                          # (B, 512, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)               # (B, patches, 512)
        x = self.transformer(x)
        x = x.mean(dim=1)                               # global avg pool
        return self.fc(x)


class TransformerFusionModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.mlp               = MLP()
        self.transformer_branch = TransformerBranch()
        self.fc                = nn.Linear(128 + 128, num_classes)

    def forward(self, txt, img):
        return self.fc(torch.cat([self.mlp(txt), self.transformer_branch(img)], dim=1))


def get_model(weights_path: str = None, device: str = "cpu") -> TransformerFusionModel:
    model = TransformerFusionModel().to(device)
    if weights_path:
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
    return model
