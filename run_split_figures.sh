#!/bin/bash
# 產生拆分版圖（不需重新訓練）
set -e

source ~/miniforge3/etc/profile.d/conda.sh
conda activate spectral
cd /home/t1204-3060/Howard/spectral_monitor

python src/make_comparison_plots.py
python src/make_eval_figures.py

# 複製到 paper_figures/
cp \
  outputs/motor_oil_curve_difference.pdf  outputs/motor_oil_curve_difference.png \
  outputs/olive_oil_curve_difference.pdf  outputs/olive_oil_curve_difference.png \
  outputs/palm_oil_curve_difference.pdf   outputs/palm_oil_curve_difference.png \
  outputs/lard_curve_difference.pdf       outputs/lard_curve_difference.png \
  outputs/water_curve_difference.pdf      outputs/water_curve_difference.png \
  outputs/class_acc_curve_mlptransformer.pdf  outputs/class_acc_curve_mlptransformer.png \
  outputs/class_prec_curve_mlptransformer.pdf outputs/class_prec_curve_mlptransformer.png \
  outputs/class_rec_curve_mlptransformer.pdf  outputs/class_rec_curve_mlptransformer.png \
  outputs/class_f1_curve_mlptransformer.pdf   outputs/class_f1_curve_mlptransformer.png \
  outputs/motor_oil_real.bmp outputs/motor_oil_generated.bmp \
  outputs/olive_oil_real.bmp outputs/olive_oil_generated.bmp \
  outputs/palm_oil_real.bmp  outputs/palm_oil_generated.bmp \
  outputs/lard_real.bmp      outputs/lard_generated.bmp \
  outputs/water_real.bmp     outputs/water_generated.bmp \
  paper_figures/

# FL 收斂曲線圖
python src/make_fl_figure.py
cp outputs/fl_convergence_curve.pdf outputs/fl_convergence_curve.png paper_figures/

echo "完成！paper_figures/ 已更新（92 個檔案）"
