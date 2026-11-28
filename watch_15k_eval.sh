#!/bin/bash
# 守望器 1：等待 ckpt_step_015000.pt 出现 → 自动跑 200-seq 评估（per-channel 量化器）
# 输出：eval_v7_15k/ ；完成标记 eval_v7_15k/DONE
cd /home/titans_v0.5/titans-memory
CKPT=checkpoints_v7/ckpt_step_015000.pt
OUT=eval_v7_15k

while [ ! -f "$CKPT" ]; do
  # 训练若已整体完成（50000 final），也用 final 跑
  if [ -f "checkpoints_v7/ckpt_step_0050000_final.pt" ]; then
    CKPT=checkpoints_v7/ckpt_step_0050000_final.pt
    OUT=eval_v7_15k
    break
  fi
  sleep 600
done

echo "[$(date)] checkpoint 就绪: $CKPT — 启动 200-seq 评估"
# 动态选当前最空的 4 张卡
GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
       | sort -t, -k2 -rn | head -4 | cut -d, -f1 | paste -sd,)
echo "[$(date)] eval GPUs=$GPUS"

V7_CKPT=$CKPT EVAL_OUT_DIR=$OUT CUDA_VISIBLE_DEVICES=$GPUS \
EVAL_SEQS=200 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/opt/conda/envs/titans/bin/python evaluate_v7_final.py > eval_v7_15k.log 2>&1
EXIT=$?
echo "[$(date)] eval EXIT=$EXIT" | tee -a eval_v7_15k.log
mkdir -p $OUT && touch $OUT/DONE
