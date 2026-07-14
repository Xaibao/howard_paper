# outputs/ 圖片說明

> 論文用圖以 ★ 標示；可刪除的舊圖以 ✗ 標示。

---

## 一、MLP-Transformer 主模型訓練結果

| 檔案 | 說明 | 論文用 |
|------|------|:------:|
| `loss_curve_mlptransformer.png` | 100 epochs Train Loss vs Test Loss 曲線。Test Loss 在前 20 epoch 震盪後逐漸收斂，Train Loss 全程平穩下降。 | ★ |
| `metrics_curve_mlptransformer.png` | 100 epochs Overall Accuracy / Precision / Recall / F1 曲線（百分比）。最終 Accuracy ≈ 77.76%（Epoch 96 最佳）。 | ★ |
| `class_metrics_curve_mlptransformer.png` | 四格子圖：各類別（motor_oil / olive_oil / palm_oil / lard / water）的 Accuracy、Precision、Recall、F1 逐 epoch 變化。palm_oil 波動最大，顯示辨識困難。 | 參考 |
| `mae_curve_mlptransformer.png` | Mean Absolute Error 曲線（訓練過程）。 | 參考 |
| `confusion_matrix_mlptransformer.png` | 單張混淆矩陣（最終 epoch，合成測試集）。 | 參考 |
| `fig_confusion_panel.png` | ★ **論文主圖**：雙面板混淆矩陣。(a) 合成測試集 Acc=95.7%；(b) 真實量測 Acc=87.7%。⚠️ 注意：(b) 每類只用 50 張原始樣本，與訓練評估 77.76% 有出入，需確認統一。 | ★ |
| `roc_curve_mlptransformer.png` | 5 類 ROC 曲線 + AUC 值。motor_oil AUC=1.00、lard AUC=1.00、palm_oil AUC=0.92（最低）。 | ★ |
| `pr_curve_mlptransformer.png` | 5 類 Precision-Recall 曲線 + AUC 值。palm_oil PR-AUC=0.74，其餘 ≥ 0.96。 | ★ |
| `{class}_metrics_mlptransformer.txt` | 各類別 Precision / Recall / F1 / Support 數值紀錄（5 份）。 | 參考 |

---

## 二、CGAN 光譜曲線生成品質

| 檔案 | 說明 | 論文用 |
|------|------|:------:|
| `cgan_curve_comparison.png` | 5 類 × 2 欄對比圖：左欄真實光譜（多條），右欄 CGAN 生成光譜（多條）。顯示波形整體相似度。 | ★ |
| `cgan_curve_comparison_meanstd.png` | 5 類疊圖：真實 mean±std（實線+陰影）vs CGAN 生成 mean±std（虛線+陰影）。更清楚呈現分佈一致性。 | ★ |
| `curve_comparison.png` | 原始版波形對比（同 cgan_curve_comparison 功能，較舊）。 | 參考 |
| `curve_difference.png` | 每類三格子：真實 mean±std / 生成 mean±std / 殘差（真實 − 生成）。殘差接近 0 代表 CGAN 品質佳。 | ★ |
| `food_oils_compare.png` | 5 類平均光譜疊圖（Motor Oil / Olive Oil / Palm Oil / Pork Fat / Water）。顯示不同物質的光譜特徵差異，論文 Introduction 用。 | ★ |
| `CGAN_curve_{class}_train.png` | 各類 CGAN_curve 訓練損失（1000 epochs，Discriminator vs Generator）。共 5 張。 | 參考 |
| `CGAN_curve_{class}_A_train.png` | 舊版 CGAN 訓練圖（被 _train.png 取代）。 | ✗ |
| `CGAN_curve_{class}_B_test.png` | 舊版 CGAN 測試圖。 | ✗ |

---

## 三、CGAN BMP 影像生成品質

| 檔案 | 說明 | 論文用 |
|------|------|:------:|
| `bmp_comparison.png` | 5 類 × 4 欄：左 2 欄真實 BMP 光譜影像，右 2 欄 CGAN 生成 BMP。生成影像雜訊較多，反映 Domain Gap。 | ★ |
| `CGAN_spectrum_{class}_A_train.png` | 各類 CGAN_spectrum BMP 生成損失（訓練集）。共 5 張。 | 參考 |
| `CGAN_spectrum_{class}_B_test.png` | 各類 CGAN_spectrum BMP 生成損失（測試集）。共 5 張。 | 參考 |
| `{class}_generated.bmp` | CGAN 生成 BMP 樣本，每類各 1 張。共 5 張。 | 參考 |
| `{class}_real.bmp` | 真實量測 BMP 樣本，每類各 1 張。共 5 張。 | 參考 |

---

## 四、XAI 可解釋 AI

| 檔案 | 說明 | 論文用 |
|------|------|:------:|
| `ig_cls_all_comparison.png` | ★ **論文主圖**：5 類合併 IG 波段重要性長條圖。紅色 = Top-5 重要波段（400-500 nm 範圍最關鍵）。 | ★ |
| `ig_cls_{class}.png` | 各類 IG 雙格子：上方原始光譜，下方波段重要性。共 5 張（motor_oil / olive_oil / palm_oil / lard / water）。 | ★ |
| `ig_palm_oil_band_importance.png` | CGAN Generator 的 IG 分析：palm_oil 輸入波段對生成品質的貢獻。論文 Discussion 說明 palm_oil 混淆原因。 | ★ |
| `lime_palm_oil.png` | LIME 影像解釋：上方原始 BMP，下方黃色輪廓 = 對分類最重要的影像區塊（palm_oil 被誤判為 water）。 | ★ |
| `ig_palm_oil.png` | 舊版 IG 大圖（被 ig_palm_oil_band_importance.png 取代）。 | ✗ |
| `ig_palm_oil_spectrogram.png` | 舊版 IG spectrogram 圖。 | ✗ |
| `ig_palm_oil_spectrum.png` | 舊版 IG spectrum 圖。 | ✗ |
| `lime_palm_oil_0001.png` | LIME 舊版（與 lime_palm_oil.png 重複）。 | ✗ |

---

## 五、可刪除的測試/雜項

| 檔案 | 說明 |
|------|------|
| `font_test.png` | 字型測試圖，無用。 |
| `font_check_now.png` | Times New Roman 驗證圖，無用。 |
| `data_audit.png` | 資料審計測試圖，無用。 |
| `validate_real_confusion.png` | 舊版 validate_real.py 輸出，不再使用。 |
| `validate_real_report.txt` | 同上。 |
| `fl_training_log.txt` | FL 訓練日誌（空檔，只有 24 bytes）。 |
| `sample_bmp/` | BMP 樣本子資料夾（與根目錄 _generated/_real.bmp 重複）。 |

---

## 六、論文圖片清單（快速總覽）

| # | 檔案 | 對應論文段落 |
|---|------|------------|
| 1 | `food_oils_compare.png` | Introduction — 光譜特徵差異 |
| 2 | `cgan_curve_comparison_meanstd.png` | Methodology — CGAN 生成品質 |
| 3 | `curve_difference.png` | Methodology — 殘差分析 |
| 4 | `bmp_comparison.png` | Methodology — BMP Domain Gap |
| 5 | `loss_curve_mlptransformer.png` | Results — 訓練收斂 |
| 6 | `metrics_curve_mlptransformer.png` | Results — 訓練準確率 |
| 7 | `fig_confusion_panel.png` | Results — 混淆矩陣雙面板 ⚠️ |
| 8 | `roc_curve_mlptransformer.png` | Results — ROC |
| 9 | `pr_curve_mlptransformer.png` | Results — PR Curve |
| 10 | `ig_cls_all_comparison.png` | XAI — 波段重要性 |
| 11 | `ig_palm_oil_band_importance.png` | XAI / Discussion — palm_oil 混淆分析 |
| 12 | `lime_palm_oil.png` | XAI — 影像解釋 |

> ⚠️ `fig_confusion_panel.png` 的 (b) Acc=87.7% 與訓練報告 77.76% 不一致，需確認後再引用。
