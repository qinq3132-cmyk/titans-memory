#!/bin/bash
# V7 50k 长训启动器：动态选卡 + 双重检查 + 阶梯降级（4→2→1 卡）+ 自动重试 15 次
# 日志/断点独立于冒烟：train_v7_50k.log, train_v7_50k_attempts.log
# 断点：checkpoints_v7/best.pt（冒烟已适应 ~2.2k 步）→ 续训至 50000 步
cd /home/titans_v0.5/titans-memory
PY=/opt/conda/envs/titans/bin/torchrun
MIN_FREE_MB=${MIN_FREE_MB:-15000}
LOG=train_v7_50k.log
ATT=train_v7_50k_attempts.log

pick_gpus() {
  local n=$1
  local first=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
                | awk -F', ' -v m=$MIN_FREE_MB '$2 >= m {print $1}')
  sleep 20
  local second=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
                 | awk -F', ' -v m=$MIN_FREE_MB '$2 >= m {print $1}')
  comm -12 <(echo "$first" | sort) <(echo "$second" | sort) | paste -sd,
}

run_attempt() {
  local n=$1
  local gpus=$(pick_gpus $n)
  if [ -z "$gpus" ]; then echo "  (no stable GPUs >=${MIN_FREE_MB}MB free)"; return 1; fi
  local accum=$(( 8 / n ))
  echo "  trying ${n} GPU(s): $gpus  GRAD_ACCUM=$accum  $(date)"
  CUDA_VISIBLE_DEVICES=$gpus \
  NUM_BATCHES=50000 FQ_WARMUP_STEPS=5000 GRAD_ACCUM=$accum \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY --nproc_per_node=$n train_mac_v7.py >> $LOG 2>&1
  return $?
}

for ATTEMPT in $(seq 1 15); do
  echo "[attempt $ATTEMPT] $(date)" >> $ATT
  if run_attempt 4; then EXIT=0
  elif run_attempt 2; then EXIT=0
  elif run_attempt 1; then EXIT=0
  else EXIT=1
  fi
  echo "[attempt $ATTEMPT] EXIT=$EXIT  $(date)" >> $ATT
  if [ $EXIT -eq 0 ]; then
    echo "[attempt $ATTEMPT] SUCCESS" >> $ATT
    exit 0
  fi
  sleep 120
done
echo "[all attempts failed]" >> $ATT
exit 1
