#!/bin/bash
# ============================================================
# run_all.sh ── 一鍵跑完整 pipeline，所有圖輸出到 outputs/
# 用法： conda activate spectral && bash run_all.sh
# ============================================================
set -e  # 遇錯即停
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo " Spectral Monitor ── 完整 Pipeline"
echo "============================================================"

# ── 階段 1：CGAN 資料增強（慢，約 30-50 分）────────────────────
echo ""
echo ">>> [1/6] CGAN 生成波形 (CGAN_curve.py)"
python src/CGAN_curve.py

echo ""
echo ">>> [2/6] CGAN 生成影像 (CGAN_spectrum.py)"
python src/CGAN_spectrum.py

# ── 階段 2：訓練主模型（約 30-60 分）──────────────────────────
echo ""
echo ">>> [3/6] 訓練 MLP-Transformer (MLP+Tran.py)"
python src/MLP+Tran.py

# ── 階段 3：驗證 + 出圖（快，幾分鐘）──────────────────────────
echo ""
echo ">>> [4/6] 第二層真實驗證 (validate_real.py)"
python src/validate_real.py

echo ""
echo ">>> [5/6] 產生對比圖 + 統一混淆矩陣"
python src/make_comparison_plots.py
python src/make_confusion_figures.py

# ── 階段 4：XAI（快）──────────────────────────────────────────
echo ""
echo ">>> [6/6] XAI 可解釋性 (LIME + Integrated Gradients)"
python src/LIME.py
python src/integrated_gradients.py

echo ""
echo "============================================================"
echo " 全部完成！所有圖在 outputs/"
echo "============================================================"
ls -1 outputs/*.png
