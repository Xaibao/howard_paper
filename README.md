# Artificial Intelligence Based Robotic Spectral Monitoring for Rapid Water Quality Disaster Assessment

> **Submitted to:** Science of the Total Environment (STOTEN), Elsevier  
> **Author:** Howard, Department of Industrial Engineering and Management, NTUST

This repository contains the full implementation for reproducing the paper, including data augmentation (CGAN), model training (MLP-Transformer), XAI analysis, Federated Learning, and the Cloud Dashboard.

---

## Table of Contents

- [Research Value & Motivation](#research-value--motivation)
- [System Requirements](#system-requirements)
- [Part 1 — Training (Reproduce the Model)](#part-1--training-reproduce-the-model)
- [Part 2 — Run the System (Dashboard)](#part-2--run-the-system-dashboard)
- [Architecture Overview](#architecture-overview)
- [Repository Structure](#repository-structure)

---

## Research Value & Motivation

### Background — Why Spectral Monitoring?

On **November 27, 2024**, a major FOG (Fats, Oils, and Grease) contamination event occurred in the Keelung River, Taiwan, affecting a 3.2 km stretch. Emergency response was delayed by over 6 hours because **conventional drone surveillance — relying entirely on visual cameras — failed to detect the colorless, transparent FOG pollutants on the water surface**.

This exposes a fundamental gap in current water quality disaster response: visual methods cannot distinguish transparent oil films from clean water.

### What This Research Proposes

This paper introduces the **first AI-based robotic spectral monitoring system** that replaces visual detection with near-infrared (NIR) spectroscopy, enabling rapid identification of invisible FOG pollutants in real time.

| Problem | This Work's Solution |
|---------|---------------------|
| Transparent FOG undetectable visually | NIR spectrum identifies molecular signatures |
| Limited real measurement data (30/class) | CGAN synthesizes 1,000 training samples/class |
| Slow cloud-only inference | MLP-Transformer on Jetson Edge: < 1 second |
| Multi-site data privacy concerns | Federated Learning — raw data never leaves the site |
| Black-box AI | XAI (LIME + Integrated Gradients) explains every prediction |

### Target Pollutants (5 Classes)

| Class | Chinese | Pollution Level | Spectral Characteristic |
|-------|---------|:--------------:|------------------------|
| Motor Oil | 機油 | Level 3 — Severe | High intensity, broad absorption band |
| Olive Oil | 橄欖油 | Level 2 — Moderate | Mid intensity, specific band peaks |
| Palm Oil | 棕櫚油 | Level 2 — Moderate | Highly correlated with water (r = 0.991) |
| Lard | 豬油 | Level 2 — Moderate | Correlated with olive oil (r = 0.965) |
| Water | 清水 | Level 0 — Normal | Baseline reference |

### Key Contributions

1. **Data augmentation under scarcity**: CGAN with multi-term spectral loss (slope + linear + reconstruction + high-band) generates physically consistent synthetic spectra from only 30 real samples per class.
2. **Dual-modal fusion**: MLP branch (1280-dim spectrum) + Transformer branch (400×1280 BMP image) combined for robust classification.
3. **Full-stack deployment**: Device → Edge → Cloud architecture with live dashboard, LLM expert analysis, and RAG-powered Q&A.
4. **Interpretability**: LIME highlights spatial regions in BMP; Integrated Gradients identifies which wavelength bands (nm) drive each classification.
5. **Privacy-preserving multi-site FL**: FedAvg convergence stable from Round 4/10; each site's raw data remains local.

### Use Cases

| Scenario | How This System Applies |
|----------|------------------------|
| **River emergency response** | Autonomous robot patrols river after an incident report; edge inference identifies pollutant type and level within 1 second; cloud dashboard alerts responders with treatment recommendations |
| **Factory discharge monitoring** | Deployed at industrial outflow points; FL allows multiple factories to share a global model without exposing proprietary data |
| **Remote or disconnected areas** | Edge inference runs offline — no internet required for real-time identification |
| **Environmental regulatory compliance** | Continuous spectral logging with timestamped records provides auditable evidence for regulators |
| **Multi-river simultaneous monitoring** | FL aggregates learnings from different river sites, improving accuracy for rare pollutant events |

---

## System Requirements

Two hardware profiles are listed below — one for full training + system, one for running the system only.

### Minimum Requirements — Run System Only (No Training)

| Component | Minimum | Tested |
|-----------|---------|--------|
| OS | Ubuntu 20.04+ | Ubuntu 22.04 |
| CPU | 4-core | Intel/AMD 8-core |
| RAM | 16 GB | 32 GB |
| GPU VRAM | 4 GB (NVIDIA, CUDA 12.x) | RTX 3060 12 GB |
| Storage | 15 GB free | NVMe SSD |
| CUDA Driver | 530+ | nvidia-535 |
| Node.js | v18+ | v20.20.2 |
| Python | 3.10+ | 3.10 (Conda) |

> The 15 GB covers: conda environment (~8 GB), model weights (~1 GB), ChromaDB, Angular build.

### Minimum Requirements — Full Training (CGAN + MLP-Transformer)

| Component | Minimum | Tested |
|-----------|---------|--------|
| OS | Ubuntu 20.04+ | Ubuntu 22.04 |
| CPU | 8-core | Intel/AMD 8-core |
| RAM | 32 GB | 32 GB |
| GPU VRAM | **8 GB** (NVIDIA, CUDA 12.x) | RTX 3060 12 GB |
| Storage | **80 GB** free | NVMe SSD |
| CUDA Driver | 530+ | nvidia-535 |

> Storage breakdown: CGAN-generated dataset (~6.2 GB) + data source (~335 MB) + conda env (~8 GB) + model weights (~1 GB) + outputs/figures (~2 GB) + buffer.

### LLM Requirements (Ollama + Llama 3.1)

| Component | Minimum | Notes |
|-----------|---------|-------|
| RAM | **+8 GB** additional | Llama 3.1 8B Q4_K_M needs ~5–6 GB RAM |
| GPU VRAM | Optional (CPU-only OK) | GPU offload speeds up inference |
| Storage | +5 GB | For Llama 3.1 model weights |

> Llama 3.1 runs on CPU if GPU VRAM is fully used by the PyTorch model. Inference will be slower (~10–30s per response) but functional.

### ⚠️ Important Notes

- **Do NOT update the Linux kernel to 6.8.0-124+** — known boot failure on this hardware configuration. Stay on 6.8.0-111.
- Use `Ctrl+C` to stop training, never `Ctrl+Z` (causes GPU memory leak).
- Add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before training commands if you encounter CUDA OOM errors.

### Install Dependencies

```bash
# 1. Create conda environment
conda create -n spectral python=3.10
conda activate spectral

# 2. Install PyTorch (CUDA 12.1)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 3. Install Python packages
pip install flask==3.1.3 flask-cors==6.0.2
pip install numpy==2.2.6 Pillow==12.2.0 scipy==1.15.3
pip install scikit-learn==1.7.2 matplotlib==3.10.8 seaborn==0.13.2
pip install langchain==1.2.15 langchain-community==0.4.1 langchain-huggingface==1.2.1
pip install chromadb==1.5.7 sentence-transformers==5.4.0
pip install anthropic==0.111.0 flwr==1.30.0
pip install transformers==5.5.3 requests==2.33.1

# 4. Install Node.js and Angular CLI
npm install -g @angular/cli

# 5. Install frontend packages
cd frontend && npm install && cd ..

# 6. Install RTMP server
cd rtmp-server && npm install && cd ..
```

### Environment Variables

Create a `.env` file or export before running:

```bash
export ANTHROPIC_API_KEY="your-claude-api-key"
```

---

---

# Part 1 — Training (Reproduce the Model)

> **Goal:** Start from 30 real measured samples per class → generate synthetic data → train MLP-Transformer → evaluate on real test set.

## Data Preparation

Place your real measured data in the following structure:

```
data_paper/real_source/
├── cgan_source/train/
│   ├── motor_oil/    ← 30 real samples (.bmp + _sg.txt each)
│   ├── olive_oil/
│   ├── palm_oil/
│   ├── lard/
│   └── water/
└── real_test/
    ├── motor_oil/    ← 250 real test samples
    ├── olive_oil/
    ├── palm_oil/
    ├── lard/
    └── water/
```

Each sample consists of two files:
- `{timestamp}.bmp` — spectral visualization image (1280×800)
- `{timestamp}_sg.txt` — Savitzky-Golay filtered spectrum (1280 float values, 300–1100 nm)

## Step 1 — CGAN: Generate Synthetic Spectrum Data (txt)

Trains a 1D-Conv CGAN for each class. Generates 1,000 synthetic `_sg.txt` files per class.

```bash
conda activate spectral
cd /path/to/spectral_monitor
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/CGAN_curve.py
```

Output: `data/dataset/train/{class}/txt/generated_{class}_*.txt`

**CGAN Loss:**
- `slope_loss` — preserves spectral slope
- `1.0 × linear_loss` — linear consistency
- `10.0 × recon_loss` — reconstruction accuracy
- `12.0 × high_band_loss` — prevents high-intensity band collapse

## Step 2 — CGAN: Generate Synthetic Spectral Images (bmp)

Trains an image CGAN (ConvTranspose2d) for each class. Generates 1,000 BMP images per class.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/CGAN_spectrum.py
```

Output: `data/dataset/train/{class}/bmp/generated_spectrum_{class}_*.bmp`

## Step 3 — Train MLP-Transformer

Trains the main fusion model on CGAN-generated data, evaluated on real test set.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/MLP+Tran.py
```

**Model Architecture:**

```
Spectrum (1280-dim txt)          Image (400×1280 BMP)
        │                                │
   MLP Branch                  Transformer Branch
  Linear(1280→512)             Conv2d patch (40×40 stride)
  ReLU + Dropout(0.3)          → 320 patches
  Linear(512→128)              TransformerEncoder (3 layers, nhead=8)
        │                      → mean pool → Linear(512→128)
        └──────── concat ──────┘
                (256-dim)
              Linear(256→5)
            5-class output
```

**Training Config:**
- Loss: CrossEntropyLoss (label_smoothing=0.1)
- Optimizer: Adam (lr=5e-5, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=100)
- Epochs: 100 | Batch size: 32

Output:
- `models/best_mlp_transformer_v2.pth` — best validation accuracy checkpoint
- `models/mlp_transformer_v2.pth` — final epoch 100 model

## Step 4 — Build RAG Knowledge Base (LLM + RAG)

The system uses **Retrieval-Augmented Generation (RAG)** to ground LLM responses in domain-specific pollution literature.

### Components

| Component | Details |
|-----------|---------|
| LLM | Ollama + Llama 3.1 (local, port 11434) |
| Embedding Model | `all-MiniLM-L6-v2` (HuggingFace, auto-downloaded) |
| Vector Store | ChromaDB (persisted at `src/fog_expert_db/`) |
| Documents | 11 files in `src/knowledge_base/` (PDF + TXT) |
| Vectors | 188 vectors after chunking |

### Knowledge Base Documents

```
src/knowledge_base/
├── fog_contamination_overview.txt       # FOG pollution general overview
├── food_oils_contamination.txt          # Food oils in water bodies
├── motor_oil_contamination.txt          # Motor oil environmental impact
├── water_treatment_response.txt         # Emergency treatment methods
├── sensors-24-01833.pdf                 # Spectral sensing literature
├── Spectrochip Pleroma Spec V8.pdf      # MSR-001 spectrometer specs
├── ac3c01132_si_001.pdf                 # Supplementary chemistry data
└── (+ 4 Taiwan regulatory PDF documents)
```

### Install Ollama + Llama 3.1

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull Llama 3.1 model
ollama pull llama3.1

# Verify (should return a response)
ollama run llama3.1 "hello"
```

### Build ChromaDB Vector Database

Run once to index all knowledge base documents:

```bash
conda activate spectral
python src/build_knowledge_db.py
```

Output: `src/fog_expert_db/` (ChromaDB, ~188 vectors)

> **Note:** Only needs to be run once. The database is persisted on disk and loaded automatically when Flask starts.

### How RAG Works in the System

```
User clicks "Run Pollution Analysis"
          │
          ▼
Flask /api/analyze
  1. Query ChromaDB → top-3 most relevant document chunks
  2. Build prompt:
       [Detection Result] Motor Oil, 93.1%, Level 3
       [Reference Literature] <3 retrieved chunks>
       [Task] Generate 4-section analysis in English
  3. Call Ollama Llama 3.1 (local, port 11434)
  4. Return structured analysis to dashboard
```

**Language Auto-Detection:**
- Query > 85% ASCII characters → English prompt → English response
- Contains Chinese characters → Chinese prompt → Chinese response

---

## Step 5 — Generate Paper Figures

```bash
# Confusion matrix (real test set, 1250 samples)
python src/make_confusion_figures.py

# CGAN vs Real comparison plots
python src/make_comparison_plots.py

# XAI: LIME image explanation
python src/LIME.py

# XAI: Integrated Gradients (band importance)
python src/ig_classifier.py

# FL convergence curve
python src/make_fl_figure.py
```

All figures saved to `paper_figures/` (PNG + PDF).

## Step 5 — Federated Learning (Optional)

Simulates multi-site learning without sharing raw data.

```bash
# Terminal 1: Start FL Server
python src/federated/fl_server.py

# Terminal 2: Start FL Client (simulate edge device)
python src/federated/fl_client.py
```

FL runs for 10 rounds, 5 epochs each. Aggregation: FedAvg.
Results saved to `outputs/fl_training_log.txt`.

---

---

# Part 2 — Run the System (Dashboard)

> **Goal:** Start the full Cloud Dashboard with live video, real-time spectrum inference, and LLM expert analysis.

## Prerequisites

- Trained model at `models/best_mlp_transformer_v2.pth`
- ChromaDB built at `src/fog_expert_db/` (run `python src/build_knowledge_db.py` once)
- Ollama running with Llama 3.1: `ollama serve` + `ollama pull llama3.1`
- (Optional) Raspberry Pi connected via Samba for Boat agent live data

## Start the System — 3 Terminals

### Terminal 1 — Flask Cloud API

```bash
conda activate spectral
cd /path/to/spectral_monitor
python src/app.py
```

Wait for: `* Running on http://0.0.0.0:5000`

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/report` | POST | Receive Edge inference result |
| `/api/results` | GET | Latest 50 detection records |
| `/api/analyze` | POST | Full LLM pollution analysis (RAG) |
| `/api/chat` | POST | Expert Q&A (English/Chinese auto-detect) |
| `/api/rpi/predict` | GET | Run inference on latest RPi file |
| `/api/rpi/latest` | GET | Latest RPi file timestamp |
| `/api/health` | GET | Health check |

### Terminal 2 — RTMP Live Video

```bash
cd /path/to/spectral_monitor
bash rtmp-server/start_all.sh
```

Wait for: `=== 全部啟動完成 ===`

| Service | Port | Description |
|---------|------|-------------|
| RTMP Server | 1935 | Receives DJI drone stream |
| ffmpeg → HLS | — | Converts RTMP to HLS (stream copy, no re-encode) |
| HLS HTTP | 18000 | Serves HLS segments to browser |

Push stream URL: `rtmp://{server-ip}:1935/live/stream`

### Terminal 3 — Angular Dashboard

```bash
cd /path/to/spectral_monitor/frontend
npx ng serve --host 0.0.0.0 --port 4200
```

Wait for: `Application bundle generation complete.`

Open browser: **`http://{server-ip}:4200`**

## Dashboard Features

| Feature | Description |
|---------|-------------|
| Live Video | RTMP stream with FPS counter |
| Spectrum Data | BMP image + 1280-dim waveform chart |
| 3 Agents | Drone / Boat / Ground — independent data sources |
| Detection Result | Class, confidence %, pollution level |
| Run Pollution Analysis | LLM generates 4-section report (English) |
| Expert Q&A | RAG-powered chat (auto English/Chinese) |
| Monitor Log | Detection history table with images |

## (Optional) Raspberry Pi — Boat Agent Live Data

Mount the RPi Samba share for real-time Boat agent data:

```bash
sudo mount -t cifs //100.74.109.10/share /home/user/rpi_share \
  -o username=pi,password=your_password
```

The RPi sends `.bmp` + `_sg.txt` files to the share. The Flask API polls every 3 seconds, runs inference, and updates the dashboard automatically.

---

## Architecture Overview

```
Raspberry Pi + MSR-001
  └── Spectrum (.txt) + Image (.bmp)
        └──→ Jetson Orin Nano (Edge inference)
                └── POST /api/report
                      └──→ Flask Cloud API (port 5000)
                                ├── Store results
                                ├── LLM analysis (Claude AI + RAG)
                                └──→ Angular Dashboard (port 4200)
                                          ├── Live Video (RTMP → HLS)
                                          ├── Spectrum visualization
                                          └── Detection results + Chat
```

---

## Repository Structure

```
spectral_monitor/
├── src/
│   ├── app.py                    # Flask Cloud API
│   ├── MLP+Tran.py               # Main model training
│   ├── CGAN_curve.py             # Spectrum CGAN (txt generation)
│   ├── CGAN_spectrum.py          # Image CGAN (bmp generation)
│   ├── LIME.py                   # XAI — LIME
│   ├── ig_classifier.py          # XAI — Integrated Gradients
│   ├── make_confusion_figures.py # Paper confusion matrix
│   ├── make_comparison_plots.py  # CGAN vs real comparison
│   ├── make_fl_figure.py         # FL convergence figure
│   └── federated/
│       ├── fl_server.py          # FL Server (FedAvg)
│       └── fl_client.py          # FL Client
├── frontend/                     # Angular Dashboard
├── rtmp-server/                  # RTMP + HLS pipeline
├── models/                       # Trained weights (.pth)
├── data_paper/real_source/       # Real measured data (train + test)
├── paper_figures/                # All paper figures (PNG + PDF)
├── outputs/                      # Training curves and logs
├── 啟動說明.txt                   # Quick start guide (Chinese)
└── README.md                     # This file
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Overall Accuracy (Epoch 100) | 83.68% |
| Motor Oil | 96.8% |
| Olive Oil | 96.8% |
| Lard | 94.9% |
| Water | 98.8% |
| Palm Oil | 32.0% (palm ↔ water confusion, spectral correlation = 0.991) |
| FL Convergence | Stable 100% from Round 4 / 10 rounds |
| Edge Inference Time | < 1 second (Jetson Orin Nano) |
