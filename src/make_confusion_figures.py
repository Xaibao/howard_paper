"""
make_confusion_figures.py
用同一個訓練好的模型，對「合成測試集」和「真實資料」各算一次混淆矩陣，
以完全一致的風格輸出，供論文 fig:confusion 雙面板使用。
另存一張並排雙面板 (a)(b)。
"""
import os, re
import numpy as np
import torch
import torch.nn as nn
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
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_mlp_transformer_v2.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CLASSES = ["motor_oil", "olive_oil", "palm_oil", "lard", "water"]
LABELS  = ["Motor Oil", "Olive Oil", "Palm Oil", "Lard", "Water"]
NC = len(CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 模型（無 EarlyConv，與 MLP+Tran.py 一致）──
class MLP(nn.Module):
    def __init__(self, i=1280, h=512, o=128):
        super().__init__(); self.fc1=nn.Linear(i,h); self.relu=nn.ReLU(); self.dropout=nn.Dropout(0.3); self.fc2=nn.Linear(h,o)
    def forward(self,x): return self.fc2(self.dropout(self.relu(self.fc1(x))))
class TransformerBranch(nn.Module):
    def __init__(self,o=128):
        super().__init__(); self.patch_embed=nn.Conv2d(3,512,40,40)
        el=nn.TransformerEncoderLayer(d_model=512,nhead=8,batch_first=True)
        self.transformer=nn.TransformerEncoder(el,num_layers=3); self.fc=nn.Linear(512,o)
    def forward(self,x):
        x=self.patch_embed(x).flatten(2).permute(0,2,1); return self.fc(self.transformer(x).mean(1))
class TransformerFusionModel(nn.Module):
    def __init__(self,nc=NC):
        super().__init__(); self.mlp=MLP(); self.transformer_branch=TransformerBranch(); self.fc=nn.Linear(256,nc)
    def forward(self,txt,img): return self.fc(torch.cat([self.mlp(txt),self.transformer_branch(img)],1))

transform = transforms.Compose([
    transforms.Resize((400,1280)), transforms.ToTensor(),
    transforms.Normalize((0.5,)*3,(0.5,)*3)])

# ── 兩種資料集 ──
class SynDataset(Dataset):
    def __init__(self, root):
        self.s=[]
        for ci,c in enumerate(CLASSES):
            td,bd=os.path.join(root,c,"txt"),os.path.join(root,c,"bmp")
            pat=re.compile(rf"^generated_{re.escape(c)}_(\d+)\.txt$")
            for f in sorted(os.listdir(td)):
                m=pat.match(f)
                if not m: continue
                bp=os.path.join(bd,f"generated_spectrum_{c}_{m.group(1)}.bmp")
                if os.path.exists(bp): self.s.append((os.path.join(td,f),bp,ci))
    def __len__(self): return len(self.s)
    def __getitem__(self,i):
        t,b,l=self.s[i]
        x=np.loadtxt(t,dtype=np.float32)
        if len(x)!=1280: x=np.interp(np.linspace(0,len(x)-1,1280),np.arange(len(x)),x).astype(np.float32)
        return torch.tensor(x/255.0), transform(Image.open(b).convert('RGB')), l

class RealDataset(Dataset):
    def __init__(self, root):
        self.s=[]
        for ci,c in enumerate(CLASSES):
            cd=os.path.join(root,c)
            if not os.path.isdir(cd): continue
            for f in sorted(os.listdir(cd)):
                if not f.endswith("_sg.txt"): continue
                base=f[:-7]
                bp=os.path.join(cd,f"{base}.bmp")
                if os.path.exists(bp): self.s.append((os.path.join(cd,f),bp,ci))
    def __len__(self): return len(self.s)
    def __getitem__(self,i):
        t,b,l=self.s[i]
        x=np.loadtxt(t,dtype=np.float32)
        if len(x)!=1280: x=np.interp(np.linspace(0,len(x)-1,1280),np.arange(len(x)),x).astype(np.float32)
        return torch.tensor(x/255.0), transform(Image.open(b).convert('RGB')), l

def evaluate(ds):
    loader=DataLoader(ds,batch_size=8,shuffle=False)
    P,Y=[],[]
    with torch.no_grad():
        for txt,img,lab in loader:
            out=model(txt.to(device),img.to(device))
            P.extend(out.argmax(1).cpu().numpy()); Y.extend(lab.numpy())
    return np.array(Y),np.array(P)

def plot_cm(ax, y, p, title):
    cm=confusion_matrix(y,p,labels=list(range(NC)))
    acc=100*accuracy_score(y,p)
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',cbar=True,
                xticklabels=LABELS,yticklabels=LABELS,ax=ax,
                annot_kws={"size":12}, square=True)
    ax.set_xlabel('Predicted',fontsize=12); ax.set_ylabel('True',fontsize=12)
    ax.set_title(f'{title}  (Acc = {acc:.1f}%)',fontsize=13)
    ax.tick_params(labelsize=10)
    return acc

if __name__=='__main__':
    model=TransformerFusionModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH,map_location=device)); model.eval()
    print("已載入模型")

    yr,pr=evaluate(RealDataset(os.path.join(BASE_DIR,"data_paper","real_source","real_test")))

    fig,ax=plt.subplots(1,1,figsize=(7,5.5))
    a=plot_cm(ax,yr,pr,'Held-out Real Spectra')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"fig_confusion_panel_best.png"),dpi=300,bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR,"fig_confusion_panel_best.pdf"),bbox_inches='tight')
    plt.close()
    print(f"真實 Acc={a:.2f}%")
    print(f"已存 fig_confusion_panel_best.png")
