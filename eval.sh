#!/bin/bash

# ================= 1. 环境与路径配置 =================

# Python 解释器路径
PYTHON_EXEC="/opt/dlami/nvme/miniconda3/envs/rpf/bin/python"

# 环境变量设置
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=3  # 指定 GPU

# ================= 2. 批量参数列表 =================
echo "Wait... 正在评估: shadowhand"
$PYTHON_EXEC test.py \
  --config config/test_palm_unconditioned_shadow.yaml

echo "Wait... 正在评估: allegro"
$PYTHON_EXEC test.py \
  --config config/test_palm_unconditioned_allegro.yaml

echo "Wait... 正在评估: barrett"
$PYTHON_EXEC test.py \
  --config config/test_palm_unconditioned_barrett.yaml

echo "Wait... 正在评估: leaphand"
$PYTHON_EXEC test.py \
  --config config/test_palm_unconditioned_leap.yaml

echo "Wait... 正在评估: robotiq_3finger"
$PYTHON_EXEC test.py \
  --config config/test_palm_unconditioned_robotiq_3finger.yaml

echo "Wait... 正在评估: ezgripper"
$PYTHON_EXEC test.py \
  --config config/test_palm_unconditioned_ezgripper.yaml  

echo "🎉 所有任务执行完毕。"