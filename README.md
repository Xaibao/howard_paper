# Artificial Intelligence Based Robotic Spectral Monitoring for Rapid Water Quality Disaster Assessment

> **Submitted to:** Science of the Total Environment (STOTEN), Elsevier  
> **Author:** Howard, Department of Industrial Engineering and Management, NTUST

---

## Overview

Traditional UAV-based water quality monitoring relies on visual imaging, which fails to detect colorless FOG (Fats, Oils, Grease) pollutants. This system deploys autonomous robots equipped with near-infrared spectrometers to classify water pollution in real-time — even when the pollutant is visually indistinguishable from clean water.

**Motivation:** The 2024/11/27 Keelung River FOG pollution incident (3.2 km range, 6+ hour detection delay) highlighted critical gaps in existing visual-only monitoring systems.

### 5 Pollution Classes

| Class | Pollution Level | Spectral Characteristic |
|-------|----------------|------------------------|
| Motor Oil | Level 3 — Severe | High intensity, broadband absorption |
| Olive Oil | Level 2 — Moderate | Medium intensity, specific band peaks |
| Palm Oil | Level 2 — Moderate | Correlation with water: 0.991 |
| Lard | Level 2 — Moderate | Correlation with olive oil: 0.965 |
| Water | Level 0 — Normal | Baseline reference |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│  DEVICE LAYER                                           │
│  Raspberry Pi 3B+ + MSR-001 Near-Infrared Spectrometer │
│  • Collects 1280-dim spectrum (300–1100 nm)             │
│  • Generates BMP spectral visualization image           │
│  • Transmits to Edge via Tailscale VPN                  │
└──────────────────────┬──────────────────────────────────┘
                       │ spectrum (.txt) + image (.bmp)
                       ▼
┌─────────────────────────────────────────────────────────┐
│  EDGE LAYER                                             │
│  Jetson Orin Nano                                       │
│  • MLP-Transformer inference (< 1 second)              │
│  • XAI: LIME image explanation + IG band importance     │
│  • Federated Learning client (local training only)      │
│  • POST results → Cloud API                             │
└──────────────────────┬──────────────────────────────────┘
                       │ JSON {prediction, confidence, image}
                       ▼
┌─────────────────────────────────────────────────────────┐
│  CLOUD LAYER                                            │
│  RTX 3060 Server (Flask API + Angular Dashboard)        │
│  • /api/report   — receive Edge inference results       │
│  • /api/analyze  — LLM pollution analysis (RAG)         │
│  • /api/chat     — Expert Q&A with literature context   │
│  • ChromaDB RAG (188 vectors, 7 pollution documents)    │
│  • Claude AI (claude-haiku-4-5) for expert analysis     │
│  • FL Server: FedAvg aggregation                        │
│  • Angular Dashboard: live video + spectrum + results   │
└─────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
spectral_monitor/
├── src/                          # All Python scripts
│   ├── app.py                    # Flask Cloud API (port 5000)
│   ├── MLP+Tran.py               # Main model training script
│   ├── CGAN_curve.py             # Spectrum CGAN (txt data augmentation)
│   ├── CGAN_spectrum.py          # Image CGAN (BMP data augmentation)
│   ├── LIME.py                   # XAI — LIME image region explanation
│   ├── ig_classifier.py          # XAI — Integrated Gradients (band importance)
│   ├── integrated_gradients.py   # XAI — IG on CGAN generator
│   ├── make_comparison_plots.py  # CGAN vs real comparison figures
│   ├── make_confusion_figures.py # Confusion matrix figures (paper)
│   ├── make_eval_figures.py      # Model evaluation figures
│   ├── make_fl_figure.py         # FL convergence curve figure
│   ├── validate_real.py          # Real data validation script
│   ├── regenerate_txt.py         # Re-generate CGAN spectrum txt files
│   └── federated/
│       ├── fl_server.py          # Federated Learning server (FedAvg, port 8080)
│       ├── fl_client.py          # FL client (local training, uploads only weights)
│       └── fl_model.py           # Shared model definition for FL
│
├── frontend/                     # Angular Cloud Dashboard
│   └── src/app/
│       ├── app.ts                # Main component (agents, polling, chart logic)
│       ├── app.html              # Dashboard template
│       └── app.css               # Styles
│
├── rtmp-server/                  # RTMP Live Video Streaming
│   ├── server.js                 # Node.js RTMP server (port 1935)
│   └── start_all.sh              # Start RTMP + ffmpeg HLS + HTTP server
│
├── models/                       # Trained model weights (not in repo — too large)
│   ├── best_mlp_transformer_v2.pth   # Best model (epoch 70, 94.40% accuracy)
│   └── mlp_transformer_v2.pth        # Final model (epoch 100, 83.68% accuracy)
│
├── data/                         # Dataset (not in repo — too large)
│   ├── augmented/                # FL client training data (real measurements)
│   └── dataset/train|test/       # CGAN-generated training set (1000 samples/class)
│
├── data_paper/                   # Paper data sources
│   └── real_source/
│       ├── cgan_source/train/    # CGAN training source (30 real samples/class)
│       └── real_test/            # Real test set (250 samples/class, 1250 total)
│
├── paper_figures/                # All paper figures (PNG + PDF)
├── outputs/                      # Training curves, confusion matrices, XAI figures
├── runs/                         # Experiment backups
│
├── CLAUDE.md                     # Full technical documentation
├── RUN_GUIDE.md                  # How to run each component
├── JETSON_SETUP.md               # Jetson Orin Nano deployment guide
├── 4060_SETUP.md                 # Windows client setup guide
└── run_all.sh                    # Start all services at once
```

---

## Model Architecture — MLP-Transformer Fusion

The core model fuses **spectrum data** (1280-dim txt) and **spectral image** (BMP) through two parallel branches:

```
Spectrum Input (1280-dim)        Image Input (400×1280 BMP)
        │                                    │
   MLP Branch                    Transformer Branch
  Linear(1280→512)               Conv2d patch embed (40×40)
  ReLU + Dropout(0.3)            → 320 patches
  Linear(512→128)                TransformerEncoder (3 layers, 8 heads)
        │                        → mean pool → Linear(512→128)
        └──────── concat(256) ──────────────┘
                       │
                 Linear(256→5)
                       │
               5-class prediction
```

**Training Setup:**
- Loss: CrossEntropyLoss (label_smoothing=0.1)
- Optimizer: Adam (lr=5e-5, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=100)
- Epochs: 100, Batch size: 32
- Data: 5,000 CGAN-generated samples (train) + 1,250 real samples (test)

**Results (Epoch 100 — paper figure):**

| Class | Accuracy |
|-------|----------|
| Motor Oil | 96.8% (242/250) |
| Olive Oil | 96.8% (243/251) |
| Palm Oil | 32.0% (82/256) ⚠️ |
| Lard | 94.9% (240/253) |
| Water | 98.8% (239/242) |
| **Overall** | **83.68%** |

> Palm Oil ↔ Water confusion discussed in paper Discussion section (spectral correlation = 0.991).

---

## Data Augmentation — CGAN

Real measured samples are limited (30/class). Two CGANs generate synthetic training data:

| Script | Input | Output | Samples Generated |
|--------|-------|--------|-------------------|
| `CGAN_curve.py` | 30 real spectrum txt | 1000 synthetic txt/class | 5,000 total |
| `CGAN_spectrum.py` | 30 real BMP images | 1000 synthetic BMP/class | 5,000 total |

**CGAN Loss (curve):** `slope_loss + 1.0×linear_loss + 10.0×recon_loss + 12.0×high_band_loss`

---

## XAI — Explainable AI

| Script | Method | Explains |
|--------|--------|---------|
| `LIME.py` | LIME | Which image regions support/oppose classification |
| `ig_classifier.py` | Integrated Gradients | Which wavelength bands (nm) drive classification |
| `integrated_gradients.py` | IG on CGAN Generator | Which input bands most affect CGAN generation quality |

---

## Federated Learning

Multi-site deployment without sharing raw data:

```
FL Server (Cloud, port 8080)
  └── Broadcasts global model weights
        ├── Edge Client 1 → trains locally → uploads Δweights
        └── Edge Client 2 → trains locally → uploads Δweights
              ↑
        FedAvg aggregation: w_global = Σ(n_i / n_total × w_i)
```

**Results:** 10 rounds, 1 client (261 samples), 5 epochs/round → stable 100% accuracy from Round 4, loss 0.413 → 0.393.

---

## Cloud Dashboard Features

- **Live Video**: DJI Mini RTMP stream → HLS → hls.js player with FPS counter
- **Spectrum Data**: Real-time BMP image + 1280-dim SG-filtered waveform chart
- **3 Agent Views**: Drone / Boat / Ground — each with independent data source
- **Detection Result**: Prediction class, confidence, pollution level badge
- **LLM Expert Q&A**: RAG-powered (ChromaDB, 188 vectors) + Claude AI analysis
- **Monitor Log**: Historical detection records with images and timestamps

---

## Quick Start

```bash
# 1. Activate environment
conda activate spectral

# 2. Start Cloud API
python src/app.py

# 3. Start live video pipeline
bash rtmp-server/start_all.sh

# 4. Start Angular dashboard
cd frontend && npx ng serve --host 0.0.0.0 --port 4200
```

Dashboard: `http://<server-ip>:4200`  
API: `http://<server-ip>:5000`

---

## Network

| Device | Role | Tailscale IP |
|--------|------|-------------|
| Raspberry Pi 3B+ + MSR-001 | Device (data collection) | 100.74.109.10 |
| Jetson Orin Nano | Edge (inference + FL client) | 100.102.243.109 |
| RTX 3060 Ubuntu | Cloud (Flask API + FL server) | 100.108.78.9 |
| RTX 4060 Windows | Client (Angular dashboard) | 100.125.219.94 |

---

## Citation

If you use this work, please cite:

> Howard et al., "Artificial Intelligence Based Robotic Spectral Monitoring for Rapid Water Quality Disaster Assessment," *Science of the Total Environment*, 2025 (under review).
