#!/bin/bash
# ============================================================
# run_figures.sh ── 只重跑「出圖/驗證」，不重訓不重生成（快，幾分鐘）
# 用現有的 models/ 和 data/ 產生所有論文圖到 outputs/
# 用法： conda activate spectral && bash run_figures.sh
# ============================================================
set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ">>> [1/4] 第二層真實驗證"
python src/validate_real.py

echo ">>> [2/4] 對比圖 + 統一混淆矩陣"
python src/make_comparison_plots.py
python src/make_confusion_figures.py

echo ">>> [3/4] XAI（LIME + Integrated Gradients）"
python src/LIME.py
python src/integrated_gradients.py

echo ">>> [4/4] FL 收斂曲線圖"
python src/make_fl_figure.py

echo "完成！outputs/ 的圖："
ls -1 outputs/*.png

echo ""
echo "✅ 論文用圖都更新好了，去 outputs/ 看"
