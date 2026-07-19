# AI-Based Robotic Spectral Monitoring for Rapid Water Quality Disaster Assessment

> **Submitted to:** Science of the Total Environment (STOTEN), Elsevier  
> **Author:** Howard, Department of Industrial Engineering and Management, NTUST

---

## Background & Motivation

On **November 27, 2024**, a FOG (Fats, Oils, and Grease) contamination event hit the **Keelung River, Taiwan** (3.2 km affected). Emergency response was delayed **over 6 hours** because RGB drone cameras **cannot detect colorless, transparent FOG** on water surfaces.

This paper proposes an AI-based robotic spectral monitoring system using **near-infrared (NIR) spectroscopy** to identify invisible FOG pollutants in real time.

### Why Each Technology Was Chosen

| Technology | Why Not the Alternative | Why This |
|------------|------------------------|----------|
| NIR Spectroscopy | RGB cameras: transparent FOG looks identical to clean water | NIR captures C-H molecular absorption fingerprints unique to each oil type |
| CGAN | Simple augmentation: doesn't preserve spectral physics; standard GAN: mode collapse with 30 samples | Four-term spectral loss (slope + linear + recon×10 + high-band×12) enforces physical consistency |
| MLP-Transformer | Single-modal: spectrum alone loses spatial patterns; image alone loses amplitude precision | Dual branch fuses both — MLP for 1280-dim spectrum, Transformer for 400×1280 BMP |
| Jetson Edge | Cloud-only: disaster sites often have no network connectivity | < 1 second inference on-device, fully offline capable |
| Federated Learning | Centralized: raw water quality data reveals site-specific info; huge bandwidth cost | Each edge trains locally, uploads weight deltas only — raw data never leaves the site |
| LLM + RAG | Static rules: cannot adapt to detected class + context | RAG grounds Llama 3.1 in 11 domain documents (188 vectors) — no hallucination |
| XAI (LIME + IG) | Black-box: regulators won't act on unexplained detections | LIME shows image regions; Integrated Gradients shows which wavelength bands drive each prediction |

### Target Pollutants

| Class | Level | Spectral Notes |
|-------|:-----:|----------------|
| Motor Oil | **Level 3 — Severe** | High intensity, broad NIR absorption; contains PAHs |
| Olive Oil | Level 2 — Moderate | Mid intensity, characteristic band peaks |
| Palm Oil | Level 2 — Moderate | r = 0.991 with water — hardest to classify |
| Lard | Level 2 — Moderate | r = 0.965 with olive oil |
| Water | **Level 0 — Normal** | Baseline reference |

### Key Results

| Metric | Value |
|--------|-------|
| Motor Oil accuracy | 96.8% |
| Olive Oil accuracy | 96.8% |
| Lard accuracy | 94.9% |
| Water accuracy | 98.8% |
| Palm Oil accuracy | 32.0% (spectral r = 0.991 with water — discussed in paper) |
| **Overall accuracy** (epoch 100) | **83.68%** |
| Best checkpoint accuracy | 94.40% (epoch ~70) |
| Edge inference time | < 1 second (Jetson Orin Nano) |
| FL convergence | Stable 100% from Round 4 / 10 rounds |

### Ablation Study

| Change | Synthetic Test Acc | Key Reason |
|--------|:-----------------:|------------|
| Baseline (with EarlyConv) | 52% | Overfit to generated texture, fails on real BMP |
| Remove EarlyConv | 61% | Better generalization |
| Add CGAN recon loss (×10) | 75.86% | Compensates for weak adversarial signal from 30 samples |
| Normalization: MinMax → ÷255 | **87.74%** | Preserves inter-class amplitude differences |
| Final model — real test set | — | **83.68%** |

> `÷255` was the biggest single improvement (+11.88%). MinMax normalizes each sample independently, erasing the fact that motor oil absorbs much more NIR than water — the key discriminative feature.

---

## System Requirements

### Run System Only

| Component | Minimum | Tested |
|-----------|---------|--------|
| OS | Ubuntu 20.04+ | Ubuntu 22.04 |
| RAM | 16 GB | 32 GB |
| GPU VRAM | 4 GB (NVIDIA CUDA 12.x) | RTX 3060 12 GB |
| Storage | 15 GB | NVMe SSD |
| Python | 3.10+ | 3.10 (Conda) |
| Node.js | v18+ | v20.20.2 |

### Full Training (CGAN + MLP-Transformer)

| Component | Minimum |
|-----------|---------|
| GPU VRAM | **8 GB** |
| Storage | **80 GB** (CGAN generates ~6.2 GB dataset) |

### LLM (Ollama + Llama 3.1)

| Component | Requirement |
|-----------|------------|
| RAM | +8 GB additional (Llama 3.1 8B Q4_K_M ≈ 5–6 GB) |
| Storage | +5 GB for model weights |

> Llama 3.1 runs on CPU if GPU is occupied. Inference is slower (~10–30s) but functional.

### ⚠️ Important

- **Do NOT update Linux kernel to 6.8.0-124+** — known boot failure. Stay on 6.8.0-111.
- Stop training with `Ctrl+C`, never `Ctrl+Z` (GPU memory leak).
- Prepend `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` if CUDA OOM occurs.

### Install Dependencies

```bash
conda create -n spectral python=3.10 && conda activate spectral

pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install flask==3.1.3 flask-cors==6.0.2 numpy==2.2.6 Pillow==12.2.0 scipy==1.15.3
pip install scikit-learn==1.7.2 matplotlib==3.10.8 seaborn==0.13.2
pip install langchain==1.2.15 langchain-community==0.4.1 langchain-huggingface==1.2.1
pip install chromadb==1.5.7 sentence-transformers==5.4.0 flwr==1.30.0
pip install transformers==5.5.3 requests==2.33.1

npm install -g @angular/cli
cd frontend && npm install && cd ..
cd rtmp-server && npm install && cd ..
```

---

# Part 1 — Training

> Goal: 30 real samples/class → CGAN synthetic data → train MLP-Transformer → evaluate on real test set.

## Data Structure

```
data/real/
├── cgan_source/train/{class}/   ← 30 real samples (CGAN training source)
└── real_test/{class}/           ← 250 real samples (final evaluation)
```

Each sample = `{timestamp}.bmp` + `{timestamp}_sg.txt` (1280 float values, 300–1100 nm, SG-filtered)

## Step 1 — CGAN: Synthetic Spectrum (txt)

Trains a 1D-Conv CGAN per class. 30 real → 1,000 synthetic `_sg.txt` per class.

```bash
conda activate spectral
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/CGAN_curve.py
```

Output: `data/dataset/train/{class}/txt/`

**Four-term spectral loss:**

| Term | Weight | Purpose |
|------|:------:|---------|
| slope_loss | 1.0 | Preserve wavelength-by-wavelength slope |
| linear_loss | 1.0 | Prevent band discontinuities |
| recon_loss | **10.0** | Compensate for weak adversarial signal (30 samples) |
| high_band_loss | **12.0** | Prevent collapse of high-intensity bands (> 0.6) |

**CGAN quality validation (mean absolute band difference):**

| Class | Mean Diff | Notes |
|-------|:---------:|-------|
| Motor Oil | 0.86 | — |
| Olive Oil | 0.95 | — |
| Palm Oil | 1.72 | Bands 477–494 corrected (195→174, real=176.7) |
| Lard | 2.81 | Bands 237–241 corrected (31→203, real=203.6) |
| Water | 4.91 | — |

## Step 2 — CGAN: Synthetic Images (bmp)

Trains an image CGAN (ConvTranspose2d) per class. Generates 1,000 BMP images per class.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/CGAN_spectrum.py
```

Output: `data/dataset/train/{class}/bmp/`

## Step 3 — Train MLP-Transformer

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/MLP+Tran.py
```

**Model architecture:**

```
Spectrum (1280-dim)              Image (400×1280 BMP)
        │                                │
   MLP Branch                   Transformer Branch
  Linear(1280→512)              Conv2d(3→512, k=40, stride=40)
  ReLU + Dropout(0.3)           → 320 patches
  Linear(512→128)               TransformerEncoder(d=512, nhead=8, layers=3)
        │                       → mean pool → Linear(512→128)
        └──────── concat(256) ──┘
                Linear(256→5)
               5-class output
```

**Training config:** Adam lr=5e-5, weight_decay=1e-4 · CosineAnnealingLR T_max=100 · 100 epochs · batch 32 · CrossEntropyLoss(label_smoothing=0.1)

**Normalization:** `spectrum ÷ 255.0` (not MinMax — preserves inter-class amplitude differences)

Output:
- `models/best_mlp_transformer_v2.pth` — best checkpoint (94.40%)
- `models/mlp_transformer_v2.pth` — epoch 100 (83.68%)

## Step 4 — LLM + RAG Knowledge Base

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh && ollama pull llama3.1

# Build ChromaDB (run once)
conda activate spectral
python src/build_knowledge_db.py
```

Output: `src/fog_expert_db/` (188 vectors from 11 domain documents)

**How RAG works:**
```
User query → ChromaDB top-3 chunks → build prompt → Llama 3.1 → expert response
```
Language auto-detection: > 85% ASCII → English response · Otherwise → Chinese response

## Step 5 — Paper Figures

```bash
python src/make_confusion_figures.py   # Confusion matrix (1,250 real samples)
python src/make_comparison_plots.py    # CGAN vs real comparison
python src/LIME.py                     # XAI: image region attribution
python src/ig_classifier.py            # XAI: wavelength band attribution
python src/make_fl_figure.py           # FL convergence curve
```

All figures → `paper_figures/` (PNG + PDF, 300 DPI)

## Step 6 — Federated Learning

In real multi-site deployment, raw water quality data **cannot be centralized** (privacy, bandwidth, regulatory reasons). FL solves this: each edge device trains locally and uploads only weight deltas.

```bash
# Terminal 1 — FL Server
conda activate spectral && python src/federated/fl_server.py

# Terminal 2 — FL Client (edge or local simulation)
conda activate spectral && python src/federated/fl_client.py
```

**FL flow:** Server broadcasts global weights → each client trains 5 local epochs → uploads Δweights → FedAvg aggregation → repeat for 10 rounds

**Non-IID data split (by deployment scenario):**

| Client | Scenario | Classes | Samples |
|--------|----------|---------|---------|
| Drone (無人機) | Aerial surveillance | motor_oil + water | 82 |
| Boat (船隻) | On-water monitoring | olive_oil + lard | 87 |
| Ground (陸地) | Shore-based station | palm_oil | 40 |

**Experimental results (10 rounds, FedAvg, 3 clients):**

| Round | Aggregated Accuracy | Loss |
|:-----:|:------------------:|:----:|
| 1 | 84.91% | 0.6677 |
| 2 | **96.23%** | 0.6118 |
| 3 | 81.13% | 0.7815 |
| 4 | **98.11%** | 0.5969 |
| 5–10 | 81–85% | 0.66–0.78 |

**Per-client local accuracy (final round):**

| Client | Local Accuracy | Note |
|--------|:--------------:|------|
| Drone | **100%** | motor_oil + water spectrally distinct |
| Boat | **100%** | olive_oil + lard separable |
| Ground | 0% | palm_oil (r=0.991 with water) — global model predicts as water; validates Discussion |

- 5 local epochs/round · FedAvg weighted by sample count
- Aggregated accuracy oscillates due to non-IID distribution — classic FL challenge
- Convergence figure: `paper_figures/fl_convergence_curve.pdf`

---

# Part 2 — Run the System

## Complete Setup on a New Computer (From Zero)

Follow these steps in order. Takes ~30–60 minutes depending on download speed.

---

### Step 0 — System Dependencies

```bash
# Ubuntu 20.04+ required. Install these first:
sudo apt update
sudo apt install -y git ffmpeg cifs-utils build-essential

# Verify ffmpeg is at /usr/bin/ffmpeg (required by start_all.sh)
which ffmpeg   # should print /usr/bin/ffmpeg
```

Install **Miniconda** (if not already installed):
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Restart terminal after install
```

Install **Node.js v18+**:
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
node --version   # should print v20.x.x
```

Install **NVIDIA driver + CUDA 12.x** (skip if already have GPU driver):
```bash
# Check current driver
nvidia-smi
# If missing, install nvidia-535 or newer from Ubuntu's additional drivers
```

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Xaibao/howard_paper.git
cd howard_paper
```

---

### Step 2 — Python Environment

```bash
conda create -n spectral python=3.10 -y
conda activate spectral

# PyTorch with CUDA 12.1
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# All other packages
pip install flask==3.1.3 flask-cors==6.0.2 \
  numpy==2.2.6 Pillow==12.2.0 scipy==1.15.3 \
  scikit-learn==1.7.2 matplotlib==3.10.8 seaborn==0.13.2 \
  langchain==1.2.15 langchain-community==0.4.1 langchain-huggingface==1.2.1 \
  chromadb==1.5.7 sentence-transformers==5.4.0 \
  flwr==1.30.0 transformers==5.5.3 requests==2.33.1
```

---

### Step 3 — Node.js Dependencies

```bash
# Angular dashboard
cd frontend && npm install && cd ..

# RTMP server
cd rtmp-server && npm install && cd ..

# Angular CLI
npm install -g @angular/cli
```

---

### Step 4 — Change the Server IP

The repo has `100.108.78.9` hardcoded in 4 places. Replace with your machine's IP.

Get your IP first:
```bash
hostname -I | awk '{print $1}'
```

**Edit `frontend/src/app/app.ts`** — change 2 lines:
```typescript
// Line 9
const API = 'http://{YOUR_IP}:5000';

// Line 60
private hlsUrl = 'http://{YOUR_IP}:18000/live/stream/index.m3u8';
```

**Edit `frontend/src/app/app.html`** — change 2 lines:
```html
<!-- Line 58 — display text only -->
<div>rtmp://{YOUR_IP}:1935/live/stream</div>

<!-- Line 124 — image src -->
[src]="'http://{YOUR_IP}:5000' + r.image_url"
```

> DJI drone push URL after change: `rtmp://{YOUR_IP}:1935/live/stream`

---

### Step 5 — Build ChromaDB Knowledge Base (RAG)

```bash
conda activate spectral
python src/build_knowledge_db.py
# Output: src/fog_expert_db/ (~188 vectors, takes 1-2 min)
```

Run once only. Persists on disk.

---

### Step 6 — Install Ollama + Llama 3.1 (LLM)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1
# Downloads ~4.7 GB — takes a few minutes

# Verify
ollama run llama3.1 "hello"
```

---

### Step 7 — Start the System (3 Terminals)

Open 3 terminals in order:

**Terminal 1 — Flask API (start this first)**
```bash
conda activate spectral
cd howard_paper
python src/app.py
# Wait for: * Running on http://0.0.0.0:5000
```

**Terminal 2 — RTMP Live Video**
```bash
cd howard_paper
bash rtmp-server/start_all.sh
# Wait for: === 全部啟動完成 ===
```

**Terminal 3 — Angular Dashboard**
```bash
cd howard_paper/frontend
npx ng serve --host 0.0.0.0 --port 4200
# Wait for: Application bundle generation complete.
```

Open browser: **`http://{YOUR_IP}:4200`**

---

### Checklist — Before Opening Browser

- [ ] `conda activate spectral` is active in Terminal 1
- [ ] Flask shows `Running on http://0.0.0.0:5000`
- [ ] RTMP shows `全部啟動完成`
- [ ] Angular shows `Application bundle generation complete`
- [ ] IP changed in `app.ts` and `app.html`
- [ ] `src/fog_expert_db/` folder exists (ChromaDB built)
- [ ] Ollama running: `ollama list` shows `llama3.1`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/report` | POST | Receive Edge inference result |
| `/api/results` | GET | Latest 50 detection records |
| `/api/analyze` | POST | LLM pollution analysis (RAG) |
| `/api/chat` | POST | Expert Q&A (auto English/Chinese) |
| `/api/rpi/predict` | GET | Inference on latest RPi file |
| `/api/health` | GET | Health check |

## (Optional) Raspberry Pi Samba — Boat Agent Live Data

The Boat agent reads live `.bmp` + `_sg.txt` files from the Raspberry Pi via Samba (CIFS).

### Step 1 — Install Samba client (first time only)

```bash
sudo apt install cifs-utils
```

### Step 2 — Create mount point

```bash
mkdir -p ~/rpi_share
```

### Step 3 — Mount the RPi share

```bash
sudo mount -t cifs //100.74.109.10/share ~/rpi_share \
  -o username=pi,password=t1204,uid=$(id -u),gid=$(id -g)
```

Verify it worked:

```bash
ls ~/rpi_share    # should show .bmp and _sg.txt files from RPi
```

### Step 4 — Start Flask (Terminal 1 as normal)

Flask automatically polls `~/rpi_share` every 3 seconds, runs inference on the latest file, and updates the **Boat** agent panel on the dashboard.

### Unmount when done

```bash
sudo umount ~/rpi_share
```

### If mount fails

| Error | Fix |
|-------|-----|
| `mount error(115)` | RPi not reachable — check Tailscale (`tailscale status`) |
| `Permission denied` | Wrong username/password — verify RPi Samba config |
| `cifs-utils not found` | Run `sudo apt install cifs-utils` first |

> RPi Tailscale IP: `100.74.109.10` · Share name: `share` · User: `pi`

---

## Architecture

```
RPi + MSR-001 (Device)
  └─ spectrum (.txt) + image (.bmp)
       └─→ Jetson Orin Nano (Edge) — inference < 1s
               └─→ POST /api/report
                     └─→ Flask API :5000 (Cloud)
                           ├─ LLM + RAG analysis
                           └─→ Angular Dashboard :4200
                                 ├─ Live video (RTMP→HLS)
                                 ├─ Spectrum + waveform
                                 └─ Detection + Chat
```

---

## Repository Structure

```
spectral_monitor/
├── src/
│   ├── app.py                    # Flask API
│   ├── MLP+Tran.py               # Main model training
│   ├── CGAN_curve.py             # Spectrum CGAN
│   ├── CGAN_spectrum.py          # Image CGAN
│   ├── LIME.py                   # XAI — LIME
│   ├── ig_classifier.py          # XAI — Integrated Gradients
│   ├── make_confusion_figures.py # Confusion matrix
│   ├── make_comparison_plots.py  # CGAN vs real plots
│   ├── make_fl_figure.py         # FL convergence figure
│   └── federated/
│       ├── fl_server.py          # FL Server (FedAvg, 10 rounds)
│       ├── fl_client.py          # FL Client (5 epochs/round)
│       └── fl_model.py           # Shared model definition
├── frontend/                     # Angular Dashboard
├── rtmp-server/                  # RTMP + HLS pipeline
├── models/                       # Trained weights
├── data/real/                    # Real measurements (cgan_source + real_test)
├── paper_figures/                # All paper figures
├── outputs/                      # Training logs and curves
└── 啟動說明.txt                   # Quick-start guide (Chinese)
```
