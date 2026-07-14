import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
from torchvision import transforms
from torch.amp import GradScaler, autocast
import torch.utils.checkpoint as checkpoint
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
from scipy.ndimage import gaussian_filter

# 設置記憶體分配策略
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 設備設置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES   = ["motor_oil", "olive_oil", "palm_oil", "lard", "water"]
NUM_TRAIN = 1000
NUM_TEST  = 250

# 數據集
class SpectrogramDataset(Dataset):
    def __init__(self, image_dir, intensity_dir, material_name, transform=None, file_list=None):
        self.image_dir     = image_dir
        self.intensity_dir = intensity_dir
        self.material_name = material_name
        self.transform     = transform
        if file_list is not None:
            self.image_files = sorted(file_list)
        else:
            self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.bmp')])
        print(f"  找到 {len(self.image_files)} 張圖片")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')
        original_size = image.size
        if original_size != (1280, 800):
            print(f"警告：{img_path} 的原始尺寸為 {original_size}，預期為 (1280, 800)")
            image = image.resize((1280, 800), Image.Resampling.BILINEAR)
        
        if self.transform:
            image = self.transform(image)
        
        if image.shape[1] != 800 or image.shape[2] != 1280:
            print(f"錯誤：{img_path} 的轉換後尺寸為 {image.shape}，預期為 (1, 800, 1280)")
            image = transforms.functional.resize(image, (800, 1280), interpolation=transforms.InterpolationMode.BILINEAR)

        # bmp stem → 對應 _sg.txt（支援 timestamp 和舊命名）
        stem           = os.path.splitext(img_file)[0]   # e.g. 20260531_103112 or motor_oil_0001
        intensity_file = f"{stem}_sg.txt"
        intensity_path = os.path.join(self.intensity_dir, intensity_file)
        intensity = np.loadtxt(intensity_path, dtype=np.float32)
        expected_dim = 1280
        if len(intensity) != expected_dim:
            print(f"錯誤：{intensity_path} 的光強度陣列長度為 {len(intensity)}，預期為 {expected_dim}")
            if len(intensity) > expected_dim:
                intensity = intensity[:expected_dim]
            else:
                intensity = np.pad(intensity, (0, expected_dim - len(intensity)), mode='constant')
        
        intensity = torch.tensor(intensity, dtype=torch.float32)
        return image, intensity

# 轉換管道
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('L')),
    transforms.Lambda(lambda img: img.resize((1280, 800), Image.Resampling.BILINEAR)),
    transforms.ToTensor(),
])

# 生成器
class Generator(nn.Module):
    def __init__(self, noise_dim=100, condition_dim=1280):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        
        # 初始投影：4x6
        self.project = nn.Linear(noise_dim + condition_dim, 64 * 4 * 6)
        
        # 1. 垂直特徵強化塊：在低解析度時就定義好垂直帶
        self.vertical_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(15, 1), padding=(7, 0)), 
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        # 2. 亞像素卷積放大 (PixelShuffle)：4x6 -> 8x12
        # 要放大 2 倍，輸入通道必須是 輸出通道 * (2^2) = 64 * 4 = 256
        self.pixel_shuffle_block = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        # 3. 剩餘放大層：8x12 -> ... -> 64x96
        self.upscale_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16x24
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 32x48
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=1),   # 64x96
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, kernel_size=5, stride=2, padding=1),    # 128x192
            nn.Sigmoid()
        )
    
    def forward(self, noise, condition):
        x = torch.cat([noise, condition], dim=1)
        x = self.project(x).view(-1, 64, 4, 6)
        
        # 使用 checkpoint 節省記憶體，按順序執行
        x = checkpoint.checkpoint(self.vertical_block,      x, use_reentrant=False)
        x = checkpoint.checkpoint(self.pixel_shuffle_block, x, use_reentrant=False)
        x = checkpoint.checkpoint(self.upscale_layers,      x, use_reentrant=False)
        
        # 最後統一調整到目標尺寸 800x1280
        x = transforms.functional.resize(x, (800, 1280), interpolation=transforms.InterpolationMode.BILINEAR)
        return x

# 鑑別器
class Discriminator(nn.Module):
    def __init__(self, condition_dim=1280):
        super(Discriminator, self).__init__()
        self.condition_dim = condition_dim
        self.condition_project = nn.Linear(condition_dim, 50 * 80)
        self.condition_upsample = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # 100x160
            nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # 200x320
            nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),  # 400x640
            nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)   # 800x1280
        )
        self.model = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=4, stride=2, padding=1),  # 400x640
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # 200x320
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 100x160
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, kernel_size=4, stride=2, padding=1),   # 50x80
            nn.Flatten(),
            nn.Linear(50 * 80, 1)
        )
    
    def forward(self, image, condition):
        condition_map = self.condition_project(condition).view(-1, 1, 50, 80)
        condition_map = self.condition_upsample(condition_map)
        if condition_map.shape[2:] != image.shape[2:]:
            raise RuntimeError(f"尺寸不匹配：condition_map {condition_map.shape[2:]} vs image {image.shape[2:]}")
        x = torch.cat([image, condition_map], dim=1)
        x = checkpoint.checkpoint_sequential(self.model, segments=2, input=x, use_reentrant=False)
        return x

noise_dim     = 100
condition_dim = 1280
num_epochs    = 100
SPLIT_TRAIN   = 30   # 前 30 張真實圖訓練 CGAN-A（生成 train），其餘訓練 CGAN-B（生成 test）

def run_one_cgan(cls_dir, file_list, material_name, gen_dir, num_outputs, tag):
    """用指定真實圖訓練一個獨立 CGAN，生成影像"""
    dataset    = SpectrogramDataset(cls_dir, cls_dir, material_name, transform, file_list=file_list)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)

    real_stats = [(dataset[i][0].numpy().std(), dataset[i][0].numpy().max()) for i in range(len(dataset))]
    THRESHOLD_STD = np.mean([s[0] for s in real_stats]) * 0.3
    THRESHOLD_MAX = np.mean([s[1] for s in real_stats]) * 0.5

    generator     = Generator(noise_dim, condition_dim).to(device)
    discriminator = Discriminator(condition_dim).to(device)
    optimizer_G   = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_D   = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    adversarial_loss = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')
    g_losses, d_losses = [], []

    for epoch in range(num_epochs):
        epoch_g_loss, epoch_d_loss, batch_count = 0.0, 0.0, 0
        for real_images, conditions in dataloader:
            bs          = real_images.size(0)
            real_images = real_images.to(device, non_blocking=True)
            conditions  = conditions.to(device, non_blocking=True)
            real_labels = torch.full((bs, 1), 0.9, device=device)
            fake_labels = torch.zeros(bs, 1, device=device)

            optimizer_D.zero_grad(set_to_none=True)
            with autocast('cuda'):
                noise     = torch.randn(bs, noise_dim, device=device)
                fake_imgs = generator(noise, conditions)
                d_loss    = (adversarial_loss(discriminator(real_images, conditions), real_labels) +
                             adversarial_loss(discriminator(fake_imgs.detach(), conditions), fake_labels)) / 2
            scaler.scale(d_loss).backward(); scaler.step(optimizer_D); scaler.update()

            optimizer_G.zero_grad(set_to_none=True)
            with autocast('cuda'):
                g_loss = adversarial_loss(discriminator(fake_imgs, conditions), real_labels)
            scaler.scale(g_loss).backward(); scaler.step(optimizer_G); scaler.update()

            epoch_g_loss += g_loss.item(); epoch_d_loss += d_loss.item(); batch_count += 1
            torch.cuda.empty_cache()

        g_losses.append(epoch_g_loss / batch_count)
        d_losses.append(epoch_d_loss / batch_count)
        if (epoch + 1) % 25 == 0:
            print(f"  [{tag}] Epoch [{epoch+1}/{num_epochs}] D:{epoch_d_loss/batch_count:.4f}  G:{epoch_g_loss/batch_count:.4f}")

    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), g_losses, label='Generator Loss', color='blue')
    plt.plot(range(1, num_epochs+1), d_losses, label='Discriminator Loss', color='orange')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(f'CGAN Spectrum Loss - {material_name} ({tag})')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(BASE_DIR, "outputs", f"CGAN_spectrum_{material_name}_{tag}.png"), dpi=150)
    plt.close()

    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    torch.save(generator.state_dict(),
               os.path.join(BASE_DIR, "models", f"CGAN_spectrum_{material_name}_{tag}.pth"))

    os.makedirs(gen_dir, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        for i in range(num_outputs):
            noise   = torch.randn(1, noise_dim, device=device)
            _, cond = dataset[np.random.randint(len(dataset))]
            cond    = cond.unsqueeze(0).to(device)
            with autocast('cuda'):
                fake = generator(noise, cond)
            img_np = (fake.squeeze().cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img_np, mode='L').save(
                os.path.join(gen_dir, f"generated_spectrum_{material_name}_{i}.bmp"))
            if (i+1) % 200 == 0:
                print(f"    [{tag}] 已生成 {i+1}/{num_outputs}")
            torch.cuda.empty_cache()

def main():
    # 改用 augmented/ 避免 cgan_source 06/10 的 lard↔palm 標籤對調問題
    AUG_DIR   = os.path.join(BASE_DIR, "data", "augmented")
    SPLIT_TRAIN = 30
    SPLIT_TEST  = 20

    for material_name in CLASSES:
        print(f"\n{'='*50}\n處理類別：{material_name}")

        all_bmp = sorted([f for f in os.listdir(os.path.join(AUG_DIR, material_name)) if f.endswith('.bmp')])
        train_files = all_bmp[:SPLIT_TRAIN]
        test_files  = all_bmp[SPLIT_TRAIN:SPLIT_TRAIN + SPLIT_TEST]
        train_src   = os.path.join(AUG_DIR, material_name)
        test_src    = train_src   # same folder, different file lists

        if len(train_files) < 5 or len(test_files) < 5:
            print(f"  !! 資料不足（train:{len(train_files)} test:{len(test_files)}），跳過"); continue

        print(f"  CGAN-A:{len(train_files)}張(生成train)  CGAN-B:{len(test_files)}張(生成test)")

        train_dir = os.path.join(BASE_DIR, "data", "dataset", "train", material_name, "bmp")
        run_one_cgan(train_src, train_files, material_name, train_dir, NUM_TRAIN, "A_train")

        test_dir = os.path.join(BASE_DIR, "data", "dataset", "test", material_name, "bmp")
        run_one_cgan(test_src, test_files, material_name, test_dir, NUM_TEST, "B_test")

    print("\n全部完成！")

if __name__ == '__main__':
    main()
