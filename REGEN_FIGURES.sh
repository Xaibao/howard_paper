#!/bin/bash
# 完整流程：重新訓練 + 產出所有論文圖 + 更新 paper_figures/
# 使用方式：bash REGEN_FIGURES.sh

set -e

# ── conda activate ────────────────────────────────────────────────────────────
source ~/miniforge3/etc/profile.d/conda.sh
conda activate spectral
echo "✓ conda env: $(conda info --envs | grep '*' | awk '{print $1}')"

cd /home/t1204-3060/Howard/spectral_monitor

# ── Step 1: 重新訓練 MLP-Transformer ─────────────────────────────────────────
# 產出：loss/mae/metrics/class_metrics/roc/pr/confusion（TNR 12pt, 300dpi）
# 儲存：mlp_transformer_v2.pth（epoch 100）+ best_mlp_transformer_v2.pth（最佳）
echo ""
echo "=== Step 1: 重新訓練 MLP+Tran.py ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/MLP+Tran.py
echo "✓ 訓練完成，models/ 與 outputs/ 訓練圖已更新"

# ── Step 2: best model → ROC / PR / confusion / class_metrics（_best 後綴）──
echo ""
echo "=== Step 2: make_eval_figures.py (best model) ==="
python src/make_eval_figures.py
echo "✓ confusion_matrix_best / roc_curve_best / pr_curve_best / class_metrics_curve_best 更新完畢"

# ── Step 3: 光譜對比圖 / 殘差圖（個別 + 合併）────────────────────────────────
echo ""
echo "=== Step 3: make_comparison_plots.py ==="
python src/make_comparison_plots.py
echo "✓ 各類光譜圖 / curve_difference / food_oils_compare 更新完畢"

# ── Step 4: 混淆矩陣（真實 1250 筆，best model）→ fig_confusion_panel_best ────
echo ""
echo "=== Step 4: make_confusion_figures.py ==="
python src/make_confusion_figures.py
echo "✓ fig_confusion_panel_best 更新完畢"

# ── Step 5: LIME 影像解釋 ─────────────────────────────────────────────────────
echo ""
echo "=== Step 5: LIME.py ==="
python src/LIME.py
echo "✓ lime_palm_oil 更新完畢"

# ── Step 6: IG 分類器波段重要性 ───────────────────────────────────────────────
echo ""
echo "=== Step 6: ig_classifier.py ==="
python src/ig_classifier.py
echo "✓ ig_cls_all_comparison + 各類個別 IG 更新完畢"

# ── Step 7: IG CGAN 生成器波段重要性 ─────────────────────────────────────────
echo ""
echo "=== Step 7: integrated_gradients.py ==="
python src/integrated_gradients.py
echo "✓ ig_palm_oil_band_importance 更新完畢"

# ── Step 8: FL 收斂曲線圖 ────────────────────────────────────────────────
echo ""
echo "=== Step 8: make_fl_figure.py (FL 收斂曲線) ==="
python src/make_fl_figure.py
echo "✓ fl_convergence_curve 更新完畢"

# ── Step 9: 同步到 paper_figures/ ────────────────────────────────────────────
echo ""
echo "=== Step 8: 同步 paper_figures/ ==="
cp \
  outputs/confusion_matrix_mlptransformer.pdf  outputs/confusion_matrix_mlptransformer.png \
  outputs/roc_curve_mlptransformer.pdf         outputs/roc_curve_mlptransformer.png \
  outputs/pr_curve_mlptransformer.pdf          outputs/pr_curve_mlptransformer.png \
  outputs/loss_curve_mlptransformer.pdf        outputs/loss_curve_mlptransformer.png \
  outputs/mae_curve_mlptransformer.pdf         outputs/mae_curve_mlptransformer.png \
  outputs/metrics_curve_mlptransformer.pdf     outputs/metrics_curve_mlptransformer.png \
  outputs/class_acc_curve_mlptransformer.pdf   outputs/class_acc_curve_mlptransformer.png \
  outputs/class_prec_curve_mlptransformer.pdf  outputs/class_prec_curve_mlptransformer.png \
  outputs/class_rec_curve_mlptransformer.pdf   outputs/class_rec_curve_mlptransformer.png \
  outputs/class_f1_curve_mlptransformer.pdf    outputs/class_f1_curve_mlptransformer.png \
  outputs/food_oils_compare.pdf                outputs/food_oils_compare.png \
  outputs/motor_oil_real_curve.pdf   outputs/motor_oil_real_curve.png \
  outputs/motor_oil_generated_curve.pdf outputs/motor_oil_generated_curve.png \
  outputs/olive_oil_real_curve.pdf   outputs/olive_oil_real_curve.png \
  outputs/olive_oil_generated_curve.pdf outputs/olive_oil_generated_curve.png \
  outputs/palm_oil_real_curve.pdf    outputs/palm_oil_real_curve.png \
  outputs/palm_oil_generated_curve.pdf  outputs/palm_oil_generated_curve.png \
  outputs/lard_real_curve.pdf        outputs/lard_real_curve.png \
  outputs/lard_generated_curve.pdf   outputs/lard_generated_curve.png \
  outputs/water_real_curve.pdf       outputs/water_real_curve.png \
  outputs/water_generated_curve.pdf  outputs/water_generated_curve.png \
  outputs/motor_oil_curve_difference.pdf outputs/motor_oil_curve_difference.png \
  outputs/olive_oil_curve_difference.pdf outputs/olive_oil_curve_difference.png \
  outputs/palm_oil_curve_difference.pdf  outputs/palm_oil_curve_difference.png \
  outputs/lard_curve_difference.pdf      outputs/lard_curve_difference.png \
  outputs/water_curve_difference.pdf     outputs/water_curve_difference.png \
  outputs/motor_oil_real.bmp outputs/motor_oil_generated.bmp \
  outputs/olive_oil_real.bmp outputs/olive_oil_generated.bmp \
  outputs/palm_oil_real.bmp  outputs/palm_oil_generated.bmp \
  outputs/lard_real.bmp      outputs/lard_generated.bmp \
  outputs/water_real.bmp     outputs/water_generated.bmp \
  outputs/lime_palm_oil.pdf              outputs/lime_palm_oil.png \
  outputs/ig_cls_all_comparison.pdf      outputs/ig_cls_all_comparison.png \
  outputs/ig_cls_motor_oil.pdf  outputs/ig_cls_motor_oil.png \
  outputs/ig_cls_olive_oil.pdf  outputs/ig_cls_olive_oil.png \
  outputs/ig_cls_palm_oil.pdf   outputs/ig_cls_palm_oil.png \
  outputs/ig_cls_lard.pdf       outputs/ig_cls_lard.png \
  outputs/ig_cls_water.pdf      outputs/ig_cls_water.png \
  outputs/ig_palm_oil_band_importance.pdf outputs/ig_palm_oil_band_importance.png \
  outputs/ig_palm_oil_spectrogram.pdf     outputs/ig_palm_oil_spectrogram.png \
  outputs/ig_palm_oil_spectrum.pdf        outputs/ig_palm_oil_spectrum.png \
  outputs/fl_convergence_curve.pdf        outputs/fl_convergence_curve.png \
  outputs/confusion_matrix_best.pdf       outputs/confusion_matrix_best.png \
  outputs/roc_curve_best.pdf              outputs/roc_curve_best.png \
  outputs/pr_curve_best.pdf               outputs/pr_curve_best.png \
  paper_figures/
echo "✓ paper_figures/ 同步完畢（92 個檔案）"

# ── 完成 ──────────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  全部完成！"
echo "  outputs/     → 所有輸出圖"
echo "  paper_figures/ → 論文用圖（PNG+PDF+BMP）"
echo "  TNR 12-15pt，300-600dpi"
echo "========================================"
