# CLAUDE.md — Spectral Monitor 完整技術文件

> 使用者：**Howard**（NTUST 工管碩士生），中文溝通，論文投稿 **STOTEN**，快畢業、時間緊。
> 分工：論文文字 Howard 用另一個 AI 寫；你負責程式、實驗、出圖、查證數據。
> **不要自動跑訓練腳本，只給指令讓 Howard 自己跑。**

---

## 0. 環境

```bash
conda activate spectral     # 必做！torch 裝在這
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/xxx.py  # 跑訓練必加
```

- GPU：RTX 3060 (12GB)，driver nvidia-535，CUDA 12.2，kernel 6.8.0-111-generic
  - ⚠️ kernel 6.8.0-124 開不了機，**別更新**
- 中斷用 **Ctrl+C**，絕不 Ctrl+Z（會卡 GPU 記憶體）
- 路徑：`/home/t1204-3060/Howard/spectral_monitor`

---

## 1. 論文背景

**標題**：Artificial Intelligence-Based Robotic Spectral Monitoring for Rapid Water Quality Disaster Assessment
**投稿**：Science of the Total Environment (STOTEN), Elsevier
**動機**：2024/11/27 基隆河污染事件，無色 FOG 污染物讓傳統無人機視覺偵測完全失效

**5 類辨識目標（FOG = Fats, Oils, Grease）**：
| 類別（程式用）| 論文顯示名稱 | 中文 | 污染等級 | 光譜特性 |
|------|------|------|---------|---------|
| motor_oil | Motor Oil | 機油 | Level 3 嚴重 | 高強度，寬帶吸收 |
| olive_oil | Olive Oil | 橄欖油 | Level 2 中度 | 中強度，特定波段峰值 |
| palm_oil | Palm Oil | 棕櫚油 | Level 2 中度 | 與 water 相關係數 0.991 |
| lard | Lard | 豬油 | Level 2 中度 | 與 olive_oil 相關係數 0.965 |
| water | Water | 清水 | Level 0 正常 | 基準線 |

> 所有論文圖的顯示名稱已統一（`_DISPLAY` 對應表加入所有繪圖腳本）

---

## 2. 論文情境（Paper Scenario）

**情境設定**：河川突發性 FOG 污染（工廠排廢油、餐廚廢水）發生時，傳統無人機僅能拍攝影像，無色油脂在視覺上無法與清水區分。本系統以搭載近紅外線光譜儀的自主巡邏機器人取代純視覺方案，實現即時水質辨識與污染溯源。

**觸發事件**：2024/11/27 基隆河 FOG 污染，污染範圍 3.2 公里，傳統偵測延誤超過 6 小時。

**部署場景（5 步驟）**：
1. 機器人沿河道自主巡邏，每隔固定距離停下採樣
2. MSR-001 光譜儀照射水面，採集 1280 維近紅外光譜 + BMP 影像
3. Jetson Edge 即時推論，判斷污染類別與等級（< 1 秒）
4. 結果上傳 Cloud，Claude LLM 生成溯源分析與處理建議
5. 管理人員透過 Angular 儀表板即時查看污染地圖與 AI 報告

**核心優勢**：
- 光譜可區分透明液體（視覺無法做到）
- Edge 推論不依賴網路，斷線仍可即時辨識
- FL 架構，多地點部署，資料不離開現場

---

## 3. 系統架構

```
┌──────────────────────────────────────────────────────────┐
│  DEVICE LAYER（現場採集）                                  │
│  樹莓派 3B+  +  MSR-001 近紅外線光譜儀                    │
│  • 採集 1280 維光譜（300-1100 nm）                         │
│  • 同步產生 BMP 影像（光譜可視化，400×1280）               │
│  • 透過 Tailscale 傳送至 Edge Jetson                      │
└──────────────────────┬───────────────────────────────────┘
                       │ 光譜(.txt) + 影像(.bmp)
                       ▼
┌──────────────────────────────────────────────────────────┐
│  EDGE LAYER（現場推論）                                    │
│  Jetson Orin Nano                                        │
│  • 載入 MLP-Transformer 模型（best_mlp_transformer_v2.pth）│
│  • 前處理：光譜 /255.0，影像 Resize(400,1280) + Normalize  │
│  • 推論：TransformerFusionModel → 5 類分類 + softmax 信心度 │
│  • XAI：LIME 影像解釋 + IG 波段重要性                      │
│  • FL Client：本地資料本地訓練，只上傳 model weights        │
│  • POST /api/report → Cloud 3060                         │
└──────────────────────┬───────────────────────────────────┘
                       │ JSON {prediction, confidence, spectrum_img}
                       ▼
┌──────────────────────────────────────────────────────────┐
│  CLOUD LAYER                                              │
│                                                           │
│  3060（RTX 3060，100.108.78.9）                           │
│  • Flask API port 5000                                    │
│    - /api/report  POST  接收 Edge 辨識結果                 │
│    - /api/results GET   最新 50 筆紀錄                     │
│    - /api/analyze POST  污染完整分析（LLM+RAG）             │
│    - /api/chat    POST  LLM 自由問答                       │
│    - /api/health  GET   健康確認                           │
│  • RAG：ChromaDB（188 向量，7 份污染文獻）                  │
│  • Claude API（claude-haiku-4-5-20251001）                │
│  • FL Server：FedAvg 聚合多點 Edge weights（port 8080）    │
│                                                           │
│  4060（Windows，100.125.219.94）                          │
│  • Angular 儀表板（port 4200）                             │
│  • 監測紀錄頁面 / 污染分析頁面 / LLM 問答頁面               │
└──────────────────────────────────────────────────────────┘
```

---

## 4. 詳細流程

### 4.1 資料採集流程（Device → Edge）

```
MSR-001 光譜儀
  │
  ├─ 輸出：{timestamp}_sg.txt（1280 個浮點數，300-1100nm）
  └─ 輸出：{timestamp}.bmp（光譜可視化影像）
         │
         └─ 樹莓派打包 → Tailscale → Jetson /tmp/incoming/
```

**光譜正規化**（Edge 推論前）：
```python
txt = np.loadtxt("xxx_sg.txt")  # shape (1280,)
txt = txt / 255.0               # 固定縮放，保留振幅差異（不用 MinMax）
```

**影像前處理**（Edge 推論前）：
```python
img = Image.open("xxx.bmp").convert("RGB")
img = Resize((400, 1280)) → ToTensor() → Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
```

---

### 4.2 CGAN 訓練流程（離線，跑在 3060）

**目的**：用少量真實資料（30 張/類）生成大量訓練資料（1000 張/類）

```
真實量測（data_paper/real_source/cgan_source/train/{class}/，30 張）
  │
  ├─ CGAN_curve.py（光譜曲線 CGAN）
  │    Generator：Linear → Reshape → Conv1D → Sigmoid → 1280 維光譜
  │    Discriminator：Conv1D → 真/假判斷
  │    損失 = slope_loss + 1.0×linear_loss + 10.0×recon_loss + 12.0×high_band_loss
  │    high_band_loss：real > 0.6 波段額外懲罰（防止高強度塌縮）
  │    epochs：1000，每類獨立訓練
  │    輸出：data/dataset/train/{class}/txt/（1000 筆 _sg.txt）
  │
  └─ CGAN_spectrum.py（BMP 影像 CGAN）
       Generator：noise(100) + condition(1280) → ConvTranspose2d → BMP
       輸出：data/dataset/train/{class}/bmp/（1000 張 .bmp）
```

**CGAN 品質驗證（2026-06-12）**：
| 類別 | 平均差 | 修正紀錄 |
|------|:------:|---------|
| motor_oil | 0.86 | — |
| olive_oil | 0.95 | — |
| palm_oil | 1.72 | band 477-494 修正（195→174，真實 176.7）|
| lard | 2.81 | band 237-241 修正（31→203，真實 203.6）|
| water | 4.91 | — |

---

### 4.3 MLP-Transformer 訓練流程

**資料集**：
```
訓練集：data/dataset/train/{class}/（CGAN 生成，1000/類，共 5000）
測試集：data_paper/real_source/real_test/{class}/（真實量測，250/類，共 1250）
```

**模型架構（TransformerFusionModel）**：
```python
# MLP 分支（處理光譜）
Linear(1280→512) → ReLU → Dropout(0.3) → Linear(512→128)  # → spec_feat(128)

# Transformer 分支（處理影像）
Conv2d(3→512, kernel=40, stride=40)  # 400×1280 → 320 個 patch
→ TransformerEncoder(d_model=512, nhead=8, num_layers=3)
→ mean pooling → Linear(512→128)    # → img_feat(128)

# 融合
concat(spec_feat, img_feat)  # (256,)
→ Linear(256→5)              # → logits(5類)
```

**訓練設定**：
```
Loss：CrossEntropyLoss(label_smoothing=0.1)
Optimizer：Adam(lr=5e-5, weight_decay=1e-4)
Scheduler：CosineAnnealingLR(T_max=100)
Epochs：100
Batch size：32
```

**資料增強**：
```python
txt += torch.randn_like(txt) * 0.02        # 光譜加高斯雜訊
if random() < 0.05:                        # 5% corruption
    txt = random_spectrum; label = random_class
```

**訓練輸出**：
```
每輪：列印混淆矩陣 + 各類 acc/prec/recall/F1
最佳模型：models/best_mlp_transformer_v2.pth（test accuracy 更高才存）
最終模型：models/mlp_transformer_v2.pth（epoch 100）
訓練圖表：outputs/loss_curve / metrics_curve / class_metrics_curve（100 輪結束後生成）
```

---

### 4.4 Edge 推論流程（Jetson 部署後）

```python
# 1. 載入模型（開機時載入一次）
model = TransformerFusionModel()
model.load_state_dict(torch.load("models/mlp_transformer_v2.pth"))
model.eval()

# 2. 收到新樣本
txt = np.loadtxt("xxx_sg.txt") / 255.0
img = transform(Image.open("xxx.bmp"))

# 3. 推論
with torch.no_grad():
    logits = model(txt.unsqueeze(0), img.unsqueeze(0))
    probs  = torch.softmax(logits, dim=1)
    pred   = probs.argmax().item()          # 類別索引
    conf   = probs.max().item() * 100       # 信心度 %

# 4. 上傳結果
requests.post("http://100.108.78.9:5000/api/report",
    json={"prediction": CLASSES[pred], "confidence": conf})
```

---

### 4.5 Cloud Flask API 流程

**接收辨識結果（/api/report）**：
```
Edge POST → Flask 儲存至 results_store（deque, maxlen=50）
         → 非同步觸發 Claude API 分析（污染等級 ≥ 2 才分析）
```

**污染分析（/api/analyze）**：
```
1. RAG 查詢：ChromaDB similarity search（top-3 文獻片段）
2. 組合 Prompt：辨識結果 + 文獻片段 → 4 段分析要求
3. Claude API 呼叫（claude-haiku-4-5-20251001）
4. 回傳：污染分析 / 溯源分析 / 問題分析 / 處理建議
```

**RAG 知識庫**：
```
src/fog_expert_db/（ChromaDB）
├── 188 個向量
└── 7 份文件（FOG 污染、水質標準、應急處理等）
Embedding model：all-MiniLM-L6-v2（HuggingFace）
```

---

### 4.6 XAI 分析流程

**LIME（影像解釋）**：
```
輸入：真實樣本 BMP 影像
方法：LIME 對影像分割成語義區塊，隨機遮蔽，觀察預測變化
輸出：哪些影像區域對分類貢獻最大（綠色=支持，紅色=反對）
腳本：src/LIME.py
```

**IG 分類器（波段重要性）**：
```
輸入：真實樣本光譜（1280 維）
方法：Integrated Gradients，從 baseline(0) 積分到真實輸入
目標：對真實類別的 logit 求梯度
輸出：哪些波長（nm）最影響分類決策（5 類各一張 + 合併圖）
腳本：src/ig_classifier.py
Delta ≈ 0 表示收斂良好
```

**IG CGAN（生成器解釋）**：
```
輸入：條件光譜（1280 維）
方法：IG 對 CGAN Generator，目標為生成圖的總強度
輸出：哪些輸入波段最影響 CGAN 的生成品質
腳本：src/integrated_gradients.py
```

---

### 4.7 Federated Learning 流程

**架構**：Flower（flwr）框架，FedAvg 策略

```
初始化：3060 Server 載入 global model（mlp_transformer_v2.pth）
         │
         ▼
每輪 FL：
  Server 廣播 global weights → 各 Edge Client
         │
  Client 用本地資料訓練 N epochs（資料不離開現場）
         │
  Client 上傳 Δweights → Server
         │
  Server FedAvg 聚合：w_global = Σ(n_i/n_total × w_i)
         │
  更新 global model → 儲存至 models/fl_rounds/round_{N}.pth
```

**啟動方式**：
```bash
# Server（3060）
python src/federated/fl_server.py

# Client（Jetson 或模擬，另開終端機）
python src/federated/fl_client.py
```

**實際訓練結果（2026-06-19）**：
- 10 rounds，1 client（261 筆，208 訓練 / 53 驗證），每輪 5 epochs
- Round 1 Accuracy=100.00%，Round 3 短暫降至 98.11%，Round 4 起穩定 100.00%
- Loss：0.413（Round 1）→ 0.393（Round 9，最低）→ 0.399（Round 10）
- 總耗時：435.89 秒
- 數據存於：`outputs/fl_training_log.txt`
- 收斂曲線圖：`paper_figures/fl_convergence_curve.pdf`（雙軸，TNR 13pt，300dpi）

**FL 收斂圖生成**：
```bash
python src/make_fl_figure.py
```

---

## 5. 實驗結果（最終）

**最佳模型（Epoch 70）**：`best_mlp_transformer_v2.pth`，訓練期間評估 **91.76%**，make_eval_figures.py 重評 **94.40%**
**最終模型（Epoch 100）**：`mlp_transformer_v2.pth`，**83.68%**

**論文採用 Epoch 100 圖**（`confusion_matrix_mlptransformer`，83.68%）
**paper_figures/ 共 92 個檔案**（46 張圖 × PNG+PDF + 10 BMP）
  - 包含：CGAN對比、訓練曲線、XAI、FL收斂曲線、best model備用圖
  - 說明文件：`paper_figures/FIGURES_GUIDE.txt`

| 類別 | 論文名稱 | Epoch 100 準確率 | 說明 |
|------|---------|:------:|------|
| motor_oil | Motor Oil | 242/250 (96.8%) | ✓ |
| olive_oil | Olive Oil | 243/251 (96.8%) | ✓ |
| palm_oil | Palm Oil | **82/256 (32.0%)** | palm↔water 混淆，寫進 Discussion |
| lard | Lard | 240/253 (94.9%) | ✓ |
| water | Water | 239/242 (98.8%) | ✓ |

**論文圖說明**：
- `confusion_matrix_mlptransformer.pdf` → 論文主混淆矩陣（epoch 100）
- `confusion_matrix_best.pdf` → 備用（best model，94.40%）
- `paper_figures/` → 所有論文用圖（41 張，PNG + PDF + 10 BMP）

合成測試（純參考）：92.80%（CGAN train + CGAN test）

---

## 6. 消融實驗

| 改進 | 合成測試 | 原理 |
|------|:-------:|------|
| 起點（有 EarlyConv）| 52% | 過擬合生成影像顆粒，真實影像對不上 |
| 移除 EarlyConv | 61% | 泛化提升 |
| CGAN 加重建損失 | 75.86% | 補償對抗訊號弱的問題 |
| 正規化改 /255 | 87.74% | 保留類別間振幅差異 |
| 真實測試集評估 | **77.76%** | 最終泛化結果 |

---

## 7. XAI 完成狀態

| 腳本 | 解釋對象 | 狀態 | 主要輸出 |
|------|---------|:----:|---------|
| `src/LIME.py` | 分類器影像區域 | ✓ | `lime_palm_oil.png` |
| `src/ig_classifier.py` | 分類器波段重要性 | ✓ | `ig_cls_all_comparison.png` |
| `src/integrated_gradients.py` | CGAN 生成器 | ✓ | `ig_palm_oil_band_importance.png` |

palm_oil IG 顯示模型用 water 相關波段判斷 → Discussion 佐證 palm↔water 混淆原因。

---

## 8. 程式碼索引

| 檔案 | 用途 |
|------|------|
| `src/CGAN_curve.py` | 光譜曲線 CGAN |
| `src/CGAN_spectrum.py` | BMP 影像 CGAN |
| `src/MLP+Tran.py` | 主模型訓練 |
| `src/make_comparison_plots.py` | CGAN 對比圖 / BMP 比較 / 五類光譜疊圖 |
| `src/make_confusion_figures.py` | 混淆矩陣（真實測試集 1250 筆，250/類，best 模型 epoch 96）|
| `src/LIME.py` | XAI — LIME |
| `src/ig_classifier.py` | XAI — IG 分類器 |
| `src/integrated_gradients.py` | XAI — IG CGAN |
| `src/app.py` | Flask Cloud API |
| `src/federated/fl_server.py` | FL Server（10 rounds，FedAvg）|
| `src/federated/fl_client.py` | FL Client（本地 261 筆，5 epochs/輪）|
| `src/federated/fl_model.py` | FL 模型定義 |
| `src/make_fl_figure.py` | FL 收斂曲線圖生成（雙軸 Accuracy+Loss）|

---

## 9. 資料路徑

```
data_paper/real_source/
├── cgan_source/train/{class}/   ← CGAN 來源（30 張/類）
└── real_test/{class}/           ← 測試集（250 張/類，含 _cp）

data/data/{class}/               ← 5/31 量測（102/100/102/100/52 張）
data/data/{class}_50/            ← 6/10 量測（~100-112 張）
data/dataset/train/{class}/txt/  ← CGAN 生成訓練集（1000/類）
data/dataset/train/{class}/bmp/  ← 對應 BMP
models/                          ← .pth 模型檔
outputs/                         ← 圖表輸出（詳見 outputs/OUTPUTS_GUIDE.md）
runs/                            ← 實驗備份
```

---

## 10. 硬體與網路

| 設備 | 角色 | Tailscale IP |
|------|------|-------------|
| 樹莓派 3B+ + MSR-001 | Device 採集 | — |
| Jetson Orin Nano | Edge 推論 + FL Client | — |
| 3060（RTX 3060 12GB）| Cloud Flask + FL Server | 100.108.78.9 |
| 4060（Windows）| Cloud Angular UI | 100.125.219.94 |

---

## 11. 與 Howard 互動要點

- 中文回覆，直接、有憑據
- 有疑問先跑指令查證，不要猜測
- 不要自動跑訓練（給指令讓他自己跑）
- 他常打錯字/注音（稿=去做，散個=幾個，心=看），用上下文理解意圖
- 做大動作前先說清楚影響（刪檔、覆蓋模型等）
