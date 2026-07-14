# RUN_GUIDE.md — 操作指令快速參考

> 先跑 `conda activate spectral` 再做任何事。

---

## A. CGAN 生成訓練集

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/CGAN_curve.py
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/CGAN_spectrum.py
```

來源：`data_paper/real_source/cgan_source/train/{class}/`（30 張/類）
輸出：`data/dataset/train/{class}/txt/` + `/bmp/`（1000/類）

---

## B. 訓練主模型

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/MLP+Tran.py
```

- 訓練：CGAN 生成 1000/類 → 測試：真實量測 250/類
- 最佳模型：`models/best_mlp_transformer_v2.pth`（epoch 70，**91.76%**）
- 最終模型：`models/mlp_transformer_v2.pth`（epoch 100，**83.68%**）
- 論文使用：epoch 100（`confusion_matrix_mlptransformer`）

---

## C. 論文圖生成

### 完整流程（含重新訓練）
```bash
bash REGEN_FIGURES.sh
```

### 只重產圖（不重新訓練）
```bash
bash run_split_figures.sh
```

個別腳本：
```bash
# 光譜對比 / 殘差圖 / BMP（個別 + 合併）
python src/make_comparison_plots.py

# best model → ROC / PR / confusion / class_metrics（_best 後綴）
# + 各指標獨立圖（class_acc/prec/rec/f1_curve_mlptransformer）
python src/make_eval_figures.py

# 混淆矩陣（best model，fig_confusion_panel_best）
python src/make_confusion_figures.py

# LIME 影像解釋
python src/LIME.py

# IG 分類器波段重要性（5 類 + 合併圖）
python src/ig_classifier.py

# IG CGAN 生成器波段重要性
python src/integrated_gradients.py
```

### 論文圖清單（paper_figures/）

| 檔案 | 說明 |
|------|------|
| `confusion_matrix_mlptransformer` | 混淆矩陣，epoch 100，83.68% |
| `roc_curve_mlptransformer` | ROC 曲線 |
| `pr_curve_mlptransformer` | PR 曲線 |
| `loss_curve_mlptransformer` | 訓練 Loss 曲線 |
| `mae_curve_mlptransformer` | MAE 曲線 |
| `metrics_curve_mlptransformer` | 整體 Accuracy 曲線 |
| `class_acc/prec/rec/f1_curve_mlptransformer` | 各類指標曲線（4 張）|
| `food_oils_compare` | 五類光譜平均疊圖 |
| `{class}_real_curve` | 各類真實光譜（5 張）|
| `{class}_generated_curve` | 各類 CGAN 生成光譜（5 張）|
| `{class}_curve_difference` | 各類殘差圖，垂直 3×1（5 張）|
| `{class}_real.bmp` / `{class}_generated.bmp` | 原始 BMP（10 個）|
| `lime_palm_oil` | LIME 影像解釋 |
| `ig_cls_all_comparison` | IG 分類器合併圖 |
| `ig_cls_{class}` | IG 分類器個別圖（5 張）|
| `ig_palm_oil_band_importance` | IG CGAN 生成器 |
| `ig_palm_oil_spectrogram` / `ig_palm_oil_spectrum` | IG CGAN 輔助圖 |
| `fl_convergence_curve` | FL 收斂曲線（Accuracy + Loss，10 rounds）|
| `confusion_matrix_best` / `roc_curve_best` / `pr_curve_best` | 最佳模型備用圖 |

---

## D. 啟動 Flask API（Claude API）

**首次設定：**
```bash
pip install anthropic
echo 'export ANTHROPIC_API_KEY="sk-ant-你的key"' >> ~/.bashrc
source ~/.bashrc
```

**啟動：**
```bash
nohup python src/app.py > logs/app.log 2>&1 &
tail -f logs/app.log
```

**測試：**
```bash
curl http://localhost:5000/api/health
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"水中發現機油污染應如何處理？"}'
```

**停止：**
```bash
pkill -f "python src/app.py"
```

---

## E. 備份模型 + 輸出圖

```bash
TIMESTAMP=$(date +%Y%m%d)
mkdir -p runs/${TIMESTAMP}_real_test
cp models/best_mlp_transformer_v2.pth runs/${TIMESTAMP}_real_test/
cp models/mlp_transformer_v2.pth      runs/${TIMESTAMP}_real_test/
cp outputs/*.png runs/${TIMESTAMP}_real_test/
cp outputs/*.txt runs/${TIMESTAMP}_real_test/ 2>/dev/null || true
```

---

## F. 清理 outputs 舊圖

```bash
cd /home/t1204-3060/Howard/spectral_monitor/outputs
rm -f CGAN_curve_*_B_test.png CGAN_curve_*_A_train.png
rm -f CGAN_spectrum_*_B_test.png CGAN_spectrum_*_A_train.png
rm -f lime_palm_oil_0001.png
rm -f ig_palm_oil.png
rm -f data_audit.png font_test.png font_check_now.png
rm -f fig_confusion_panel_preview.png fig_confusion_panel.png
rm -f curve_comparison.png cgan_curve_comparison*.png
cd ..
```

---

## G. Federated Learning

```bash
# Server（3060，port 8080）
python src/federated/fl_server.py

# Client（Jetson 或模擬，另開終端機）
python src/federated/fl_client.py
```

FL 跑完後，產生收斂曲線圖：
```bash
python src/make_fl_figure.py
cp outputs/fl_convergence_curve.pdf outputs/fl_convergence_curve.png paper_figures/
```

**實際跑出結果（2026-06-19）**：10 rounds，第 1 輪 100.00%，第 3 輪短暫降至 98.11%，
第 4 輪起穩定 100.00%，Loss 從 0.413 降至 0.393，總耗時約 436 秒。
詳細數據見 `outputs/fl_training_log.txt`。

---

## 常用資料夾速查

| 資料夾 | 內容 |
|--------|------|
| `data/data/{class}/` | 5/31 真實量測（102/100/102/100/52 張）|
| `data/data/{class}_50/` | 6/10 真實量測（102-112 張）|
| `data/dataset/train/{class}/` | CGAN 生成訓練集（1000/類）|
| `data_paper/real_source/cgan_source/train/{class}/` | CGAN 來源（30 張/類）|
| `data_paper/real_source/real_test/{class}/` | 測試集（250 張/類）|
| `models/` | 模型 .pth |
| `outputs/` | 所有輸出圖表 |
| `paper_figures/` | 論文用圖（PNG + PDF + BMP）|
| `runs/` | 實驗備份 |
