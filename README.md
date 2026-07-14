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

### Background — The FOG Water Pollution Crisis

#### Triggering Event: Keelung River, November 27, 2024

On **November 27, 2024**, a major FOG (Fats, Oils, and Grease) contamination event struck the Keelung River (基隆河), Taiwan, spreading across a **3.2 km stretch** of waterway. Emergency environmental response was delayed by **over 6 hours** because the deployed drone surveillance team — relying entirely on RGB cameras — **failed to visually identify any contamination**. The FOG pollutants were colorless and transparent on the water surface, indistinguishable from clean water to the naked eye and to RGB imaging systems.

This incident directly exposed a critical capability gap in existing water quality disaster response systems.

#### Why FOG Contamination is Uniquely Dangerous

FOG pollutants form a surface film on water that:
- **Blocks oxygen exchange** between air and water, rapidly depleting dissolved oxygen and suffocating aquatic life
- **Blocks sunlight penetration**, suppressing photosynthesis of submerged aquatic vegetation
- **Clogs water treatment infrastructure** — pump intakes, filtration membranes, and pipes
- **Carries persistent toxic compounds**: motor oil contains polycyclic aromatic hydrocarbons (PAHs), which are carcinogenic and persist in sediment for years
- **Triggers secondary eutrophication**: biological FOG (lard, palm oil, olive oil) provides a massive nutrient load as it degrades, causing algal blooms

Despite these severe impacts, **no existing real-time robotic monitoring system** is capable of detecting transparent FOG pollutants from unmanned aerial or aquatic platforms.

---

### Why These Technologies Were Chosen

#### 1. NIR Spectroscopy — Not RGB Cameras

RGB cameras record visible light reflectance (400–700 nm). Transparent liquids at low concentration reflect essentially the same spectrum as clean water — there is **no visual signature** for FOG at the surface.

Near-infrared (NIR) spectroscopy (300–1100 nm) measures **molecular absorption fingerprints**: C-H bonds in hydrocarbon chains (oils) absorb NIR radiation at characteristic wavelengths that water does not. Even colorless, transparent oil films produce a **distinct, class-specific spectral signature** in NIR.

This is why the MSR-001 spectrometer was selected:
- Range: 300–1100 nm, 1280 spectral bands (resolution ~0.625 nm/band)
- Non-contact measurement: mounted on robot/drone, no water surface contact required
- Output: raw spectrum + BMP visualization simultaneously
- Savitzky-Golay (SG) filter applied on-device to reduce shot noise while preserving spectral peaks

#### 2. CGAN — Not Simple Data Augmentation

Field measurement of water pollutants is constrained by: equipment availability, personnel safety, seasonal variation, regulatory permits for intentional contamination. This paper obtained only **30 real NIR measurements per class** — far below the thousands typically required for deep learning.

Simple augmentation approaches (random noise, time-shift, interpolation) fail for spectral data because they do not preserve the **physical spectral characteristics** that make each class distinguishable. Standard GAN training on 30 samples causes **mode collapse** — the generator collapses all outputs to the mean spectral shape.

The custom CGAN in this paper addresses this with a **four-term spectral loss**:

| Loss Term | Weight | Purpose |
|-----------|:------:|---------|
| `slope_loss` | 1.0 | Preserve the spectral slope profile across wavelength bands |
| `linear_loss` | 1.0 | Maintain linear consistency between adjacent bands |
| `recon_loss` | 10.0 | Force generated spectrum to reconstruct real training samples |
| `high_band_loss` | 12.0 | Prevent collapse of high-intensity absorption bands (> 0.6 normalized) |

The high weight on `recon_loss` and `high_band_loss` compensates for the weak adversarial signal from only 30 real discriminator samples, enforcing physical plausibility even when the discriminator cannot effectively distinguish real from fake.

**CGAN Quality Validation** — mean absolute difference between generated and real spectra across all 1280 bands (post-correction):

| Class | Mean Difference | Corrections Applied |
|-------|:--------------:|---------------------|
| Motor Oil | 0.86 | None |
| Olive Oil | 0.95 | None |
| Palm Oil | 1.72 | Bands 477–494 manually corrected (195 → 174; real mean = 176.7) |
| Lard | 2.81 | Bands 237–241 manually corrected (31 → 203; real mean = 203.6) |
| Water | 4.91 | None |

#### 3. MLP-Transformer Fusion — Not Single-Modal

The MSR-001 produces two simultaneous outputs per measurement:
- A **1D spectrum** (`_sg.txt`): 1280 float values representing absorption intensity at each wavelength
- A **BMP image**: 400×1280 pixel spectral visualization (color-mapped intensity gradient)

Each modality carries **different discriminative information**:
- The 1D spectrum captures absolute absorption amplitude at each wavelength — critical for inter-class differentiation (motor oil has much higher overall intensity than water)
- The BMP image captures **spatial gradient patterns** in the spectral visualization — visible as band structures, color transitions, and texture differences that the 1D values alone do not express

Using only the spectrum (MLP branch) misses the spatial patterns. Using only the image (Transformer branch) loses the precise numeric amplitude relationships. The fusion concatenates both 128-dim representations for final classification.

#### 4. Jetson Orin Nano — Not Cloud-Only Inference

In a real disaster scenario, a robot deployed at a remote river site may have **no reliable internet connectivity** — cell networks are often disrupted during disasters, and remote river areas have limited coverage. Cloud-dependent inference would fail at exactly the moment it is most needed.

The Jetson Orin Nano provides **on-device GPU inference** (< 1 second per sample) with no network dependency. Results are sent to the cloud when connectivity is available, but identification does not wait.

#### 5. Federated Learning — Not Centralized Training

In deployment across multiple sites (multiple rivers, multiple factory outflows), raw spectral data from each site should **not** be centralized because:
- **Data privacy**: Spectral signatures from a factory discharge site may reveal proprietary production process information
- **Bandwidth**: BMP + txt files from dozens of sensors would require substantial continuous upload bandwidth
- **Regulation**: Environmental monitoring data in some jurisdictions must remain within local administrative boundaries

With FL (Flower framework, FedAvg), each Jetson trains locally on its site's data and uploads only gradient weight updates. The cloud aggregates a stronger global model without ever seeing raw measurements.

#### 6. LLM + RAG — Not Static Rule-Based Alerts

When a pollution event is detected, responders need more than a class label — they need to know **why it matters, how it spreads, what to deploy for containment**, and what regulatory thresholds apply. Static rule-based alert systems cannot adapt to the combination of detected class + local context.

RAG (Retrieval-Augmented Generation) grounds the LLM response in specific domain documents (FOG pollution literature, MSR-001 specs, Taiwan environmental regulations, treatment method references) so that the generated expert report is **anchored in cited knowledge** rather than hallucinated. ChromaDB provides fast vector similarity search across 188 indexed chunks to retrieve the top-3 most relevant passages for each query.

#### 7. XAI — Not Black Box

Environmental agencies and regulatory bodies require **justification** for automated detection results before taking costly emergency action. A black-box classifier that outputs "Motor Oil, 93.1%" will not be trusted without explanation.

Two XAI methods are applied:
- **LIME** (Local Interpretable Model-agnostic Explanations): segments the BMP image into semantic regions and perturbs them to identify which spatial areas are driving the prediction
- **Integrated Gradients (IG)**: computes the path integral of gradients from a zero-baseline to the real input, identifying which of the 1280 wavelength bands most contributed to the classification

IG results directly support the Discussion: palm oil's low accuracy (32%) is explained by IG showing the model activates **water-correlated wavelength bands** when classifying palm oil — consistent with the spectral correlation coefficient r = 0.991 between palm oil and water.

---

### Target Pollutants (5 Classes)

| Class | Chinese | Pollution Level | Spectral Characteristic | Ecological Risk |
|-------|---------|:--------------:|------------------------|----------------|
| Motor Oil | 機油 | **Level 3 — Severe** | High intensity, broad NIR absorption | PAH contamination, long-term sediment persistence |
| Olive Oil | 橄欖油 | Level 2 — Moderate | Mid intensity, characteristic band peaks | Eutrophication, dissolved oxygen depletion |
| Palm Oil | 棕櫚油 | Level 2 — Moderate | Very similar to water (r = 0.991) — hardest to classify | Algal bloom trigger when biodegrading |
| Lard | 豬油 | Level 2 — Moderate | Correlated with olive oil (r = 0.965) | Biological oxygen demand, aquatic suffocation |
| Water | 清水 | **Level 0 — Normal** | Baseline reference spectrum | — |

### Deployment Scenario — How the Robot System Works

The full system operates as a **5-step autonomous patrol loop**:

```
Step 1: Robot (RPi + MSR-001) patrols river autonomously, stops at sampling point
         │
Step 2: MSR-001 illuminates water surface, records 1280-band NIR spectrum
         Simultaneously generates BMP spectral visualization
         Savitzky-Golay filter removes shot noise on-device
         │
Step 3: Jetson Orin Nano receives spectrum + BMP via local network
         MLP-Transformer inference: < 1 second
         Output: class label + confidence % + pollution level (0/2/3)
         │
Step 4: Result uploaded to Cloud Flask API (/api/report)
         LLM + RAG automatically generates 4-section expert analysis
         Angular dashboard updates in real time for remote responders
         │
Step 5: Responders view: live video + spectral waveform + detection result +
         LLM pollution analysis (source tracing, ecological impact,
         containment recommendations, regulatory context)
         FL Client: local Jetson trains on new sample → uploads Δweights
```

### Key Contributions

1. **Data augmentation under extreme scarcity**: Only 30 real NIR measurements per class are available — far below what deep learning requires. A 1D-Conv CGAN with a four-term physics-constrained spectral loss (slope, linear, reconstruction ×10, high-band ×12) synthesizes 1,000 training samples per class. Post-generation quality validation shows mean absolute band differences as low as 0.86 (motor oil) after manual correction of systematic errors in two classes.

2. **Dual-modal MLP-Transformer fusion**: Simultaneously encodes the 1280-dim NIR spectrum (MLP branch: Linear 1280→512→128) and the 400×1280 BMP visualization (Transformer branch: Conv2d patch extraction → 320 patches → 3-layer TransformerEncoder nhead=8 → 128-dim), concatenates to 256-dim, classifies into 5 classes. Each modality provides complementary discriminative information that the other alone cannot capture.

3. **Real-time edge inference independent of network**: The trained MLP-Transformer runs on Jetson Orin Nano at < 1 second per sample. In disaster scenarios where cloud connectivity fails, field identification continues uninterrupted. Results are buffered locally and synced when connectivity resumes.

4. **Privacy-preserving multi-site Federated Learning**: Flower (flwr) framework with FedAvg — each edge device trains 5 local epochs per round on its own water samples, uploads only gradient weight deltas. Raw spectral data never leaves the measurement site. In the paper's 10-round experiment, the global model converged stably from Round 4, achieving 100% local validation accuracy with loss decreasing from 0.413 to 0.393.

5. **Dual XAI for regulatory interpretability**: LIME (image-region attribution) + Integrated Gradients (wavelength-band attribution) provide two independent explanations per prediction. IG results quantitatively confirm the Discussion's claim about palm oil misclassification: the model relies on water-correlated wavelength bands for palm oil, consistent with r = 0.991 spectral correlation.

6. **End-to-end cloud dashboard with domain-grounded LLM**: Flask API + Angular dashboard + RTMP/HLS live video, with RAG (ChromaDB, 188 vectors, 11 domain documents including Taiwan environmental regulations) grounding Llama 3.1 responses in cited literature. English/Chinese bilingual auto-detection. Supports 3 independent agent streams (Drone / Boat / Ground).

### Ablation Study — Design Decisions That Mattered

Each architectural choice was validated through incremental experiments on a synthetic held-out test set:

| Design Change | Synthetic Test Acc | Key Insight |
|--------------|:-----------------:|-------------|
| Baseline (with EarlyConv layer) | 52% | EarlyConv overfit to generated image texture; real BMP textures differ |
| Remove EarlyConv | 61% | Transformer branch generalizes better without early spatial bias |
| Add CGAN reconstruction loss (×10) | 75.86% | Reconstruction loss compensates for weak adversarial signal (30 samples) |
| Change normalization: MinMax → ÷255 | **87.74%** | ÷255 preserves inter-class amplitude differences; MinMax collapsed them |
| **Final model on real test set** | — | **83.68%** (epoch 100), best checkpoint: 94.40% |

> The normalization change from MinMax to ÷255 produced the single largest accuracy jump (+11.88%). MinMax normalization scaled each spectrum independently to [0,1], erasing the fact that motor oil has much higher absolute absorption than water — a key discriminative feature. Fixed-scale ÷255 preserves this inter-class amplitude relationship.

### Use Cases

| Scenario | How This System Applies |
|----------|------------------------|
| **River emergency response** | Robot dispatched immediately upon FOG spill report; edge inference identifies pollutant class and severity (Level 0/2/3) within 1 second per sample; cloud dashboard delivers LLM expert report to responders within minutes, including containment recommendations and regulatory thresholds |
| **Factory discharge monitoring** | Robots deployed continuously at industrial outflow points; FL allows multiple factories to contribute to a shared model without exposing proprietary production data to competitors or regulators |
| **Remote / disconnected areas** | Edge inference runs fully offline — flash floods and disaster conditions that disrupt cellular networks do not impair real-time identification |
| **Environmental regulatory compliance** | Every detection is timestamped and logged with spectrum + BMP evidence; provides auditable chain of custody for regulatory enforcement |
| **Multi-river collaborative monitoring** | FL aggregates learnings across geographically distributed river sites — a pollution pattern encountered at one site improves recognition across all deployed nodes without data sharing |

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

Trains a 1D-Conv CGAN independently for each of the 5 classes. Each class CGAN takes the 30 real `_sg.txt` spectra as training data and generates 1,000 synthetic spectra that preserve the physical characteristics of that class.

```bash
conda activate spectral
cd /path/to/spectral_monitor
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/CGAN_curve.py
```

Output: `data/dataset/train/{class}/txt/generated_{class}_*.txt`

**Generator Architecture:** Noise(100-dim) + Condition(1280-dim) → Linear(1380→1024) → ReLU → Linear(1024→1280) → Sigmoid → generated spectrum

**Discriminator Architecture:** spectrum(1280-dim) → Conv1D(1→16, k=15) → Conv1D(16→32, k=15) → Linear → real/fake

**Four-Term Spectral Loss (why each term exists):**

| Term | Weight | What it Enforces |
|------|:------:|-----------------|
| `slope_loss` | 1.0 | Generated spectrum must follow the same wavelength-by-wavelength slope trend as real spectra |
| `linear_loss` | 1.0 | Linear consistency — prevents abrupt discontinuities between adjacent bands |
| `recon_loss` | **10.0** | Generator must be able to reconstruct real training samples — compensates for weak adversarial signal from only 30 real samples |
| `high_band_loss` | **12.0** | Penalizes collapse of high-intensity bands (normalized value > 0.6) — prevents the generator from defaulting to a flat, low-energy output |

**CGAN Quality Validation (paper):**

| Class | Mean Abs. Difference | Correction Required |
|-------|:-------------------:|---------------------|
| Motor Oil | 0.86 | None |
| Olive Oil | 0.95 | None |
| Palm Oil | 1.72 | Bands 477–494 corrected: generated 195 → target 174 (real mean = 176.7) |
| Lard | 2.81 | Bands 237–241 corrected: generated 31 → target 203 (real mean = 203.6) |
| Water | 4.91 | None |

> Lard showed the largest systematic error — bands 237–241 were near-zero in generated spectra but should be ~203. This was caused by the high-band loss not covering the mid-range. Manual correction was applied post-generation.

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
- Data augmentation: Gaussian noise (σ=0.02) on spectrum + 5% random sample corruption

**Normalization (critical design choice):**
```python
# Spectrum: fixed-scale division — preserves inter-class amplitude differences
txt = txt / 255.0   # NOT MinMax — MinMax erases amplitude differences between classes

# Image: standard ImageNet-style
Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
```

> Switching from MinMax to ÷255 was the single most impactful design change, producing a +11.88% accuracy improvement. MinMax normalizes each sample independently, erasing the fact that motor oil has much higher NIR absorption than water — a key class-discriminating feature.

Output:
- `models/best_mlp_transformer_v2.pth` — best real-test accuracy checkpoint (epoch ~70, 94.40%)
- `models/mlp_transformer_v2.pth` — final epoch 100 model (83.68% on real test)

**Final Results on Real Test Set (1,250 samples, 250/class):**

| Class | Correct / Total | Accuracy | Note |
|-------|:--------------:|:--------:|------|
| Motor Oil | 242 / 250 | **96.8%** | |
| Olive Oil | 243 / 251 | **96.8%** | |
| Lard | 240 / 253 | **94.9%** | |
| Water | 239 / 242 | **98.8%** | |
| Palm Oil | 82 / 256 | 32.0% | Confused with Water (r=0.991) — see Discussion |
| **Overall** | | **83.68%** | Epoch 100 model |

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

---

## Step 6 — Federated Learning

### Why FL in This System?

In real deployment, multiple autonomous robots operate at **different river sites** (upstream factory outflow, city waterway, estuary). Each site collects water samples that may reflect local pollution characteristics (seasonal variation, industrial mix). Simply centralizing all raw water quality data is problematic:

- **Privacy**: Raw spectral data from a factory site may reveal proprietary production information.
- **Bandwidth**: Continuously uploading large BMP + txt files from remote sites is impractical.
- **Regulation**: Environmental monitoring data may be subject to jurisdiction-specific data residency requirements.

**Federated Learning solves all three**: each edge device trains locally on its own data and only sends weight updates to the cloud server. The server aggregates them into a global model (FedAvg) and distributes it back — **no raw data ever leaves the site**.

### FL Architecture

```
Cloud Server (Flask + Flower, port 8080)
  │  1. Broadcast global model weights
  │
  ├──→ Edge Device A (Jetson, River Site 1)
  │       local train 5 epochs on site-A samples
  │       upload Δweights only
  │
  ├──→ Edge Device B (Jetson, River Site 2)
  │       local train 5 epochs on site-B samples
  │       upload Δweights only
  │
  └──→ ... (scalable to N sites)
         │
         ▼
  FedAvg: w_global = Σ (n_i / n_total) × w_i
         │
         ▼
  Updated global model → broadcast to all sites → next round
```

### How to Run

```bash
# Terminal 1 — FL Server (Cloud, port 8080)
conda activate spectral
python src/federated/fl_server.py

# Terminal 2 — FL Client (Edge or local simulation)
conda activate spectral
python src/federated/fl_client.py
```

**FL Server** (`fl_server.py`): Uses Flower's `FedAvg` strategy, 10 rounds, aggregates weight updates from all connected clients each round.

**FL Client** (`fl_client.py`): Loads local training data, trains `TransformerFusionModel` for 5 epochs per round, reports metrics to server.

**FL Model** (`fl_model.py`): Defines the same `TransformerFusionModel` architecture as the main classifier — ensures server and client use identical model structure.

### Experimental Results (Paper)

| Round | Accuracy | Loss |
|:-----:|:--------:|:----:|
| 1 | 100.00% | 0.413 |
| 2 | 100.00% | 0.405 |
| 3 | 98.11% | 0.404 |
| 4 | 100.00% | 0.401 |
| 5–8 | 100.00% | 0.397–0.401 |
| 9 | 100.00% | **0.393** (lowest) |
| 10 | 100.00% | 0.399 |

- **Setup**: 1 simulated edge client, 261 local samples (208 train / 53 validation), 5 local epochs per round
- **Convergence**: Stable from Round 4; minor dip at Round 3 recovered immediately
- **Total training time**: 435.89 seconds for 10 rounds
- **Aggregation**: FedAvg (weighted by local dataset size)

Results log: `outputs/fl_training_log.txt`
Convergence figure: `paper_figures/fl_convergence_curve.pdf` (dual-axis: Accuracy + Loss)

### Generate FL Convergence Figure

```bash
python src/make_fl_figure.py
```

Output: `paper_figures/fl_convergence_curve.pdf` and `.png` (300 DPI, dual Y-axis).

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
