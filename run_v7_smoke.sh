#!/bin/bash
# V7 冒烟训练自动重试包装器 v2：
#   - 双重检查：间隔 20s 两次采样显存都 ≥ MIN_FREE_MB 才入选（过滤波动卡）
#   - 阶梯降级：4 卡 → 2 卡 → 1 卡（GRAD_ACCUM 相应调整为 2/4/8，保持等效 batch=32）
#   - 最多 12 次尝试，间隔 120s
cd /home/titans_v0.5/titans-memory
PY=/opt/conda/envs/titans/bin/torchrun
MIN_FREE_MB=${MIN_FREE_MB:-20000}

pick_gpus() {  # $1 = 需要的卡数；输出双重检查通过的卡列表
  local n=$1
  local first=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
                | awk -F', ' -v m=$MIN_FREE_MB '$2 >= m {print $1}')
  sleep 20
  local second=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
                 | awk -F', ' -v m=$MIN_FREE_MB '$2 >= m {print $1}')
  # 取两次都空闲的卡
  local stable=$(comm -12 <(echo "$first" | sort) <(echo "$second" | sort) | paste -sd,)
  echo "$stable" | awk -F',' -v n=$n 'NF {print} NR==n {exit}'
}

run_attempt() {  # $1 = 卡数
  local n=$1
  local gpus=$(pick_gpus $n)
  if [ -z "$gpus" ]; then echo "  (no stable GPUs with >=${MIN_FREE_MB}MB free)"; return 1; fi
  local accum=$(( 8 / n ))
  echo "  trying ${n} GPU(s): $gpus  GRAD_ACCUM=$accum"
  CUDA_VISIBLE_DEVICES=$gpus \
  NUM_BATCHES=2000 FQ_WARMUP_STEPS=${FQ_WARMUP_STEPS:-1000} GRAD_ACCUM=$accum \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY --nproc_per_node=$n train_mac_v7.py >> train_v7_smoke.log 2>&1
  return $?
}

for ATTEMPT in 1 2 3 4 5 6 7 8 9 10 11 12; do
  echo "[attempt $ATTEMPT] $(date)" >> train_v7_smoke_attempts.log
  # 阶梯：4 卡 → 2 卡 → 1 卡
  if run_attempt 4; then EXIT=0
  elif run_attempt 2; then EXIT=0
  elif run_attempt 1; then EXIT=0
  else EXIT=1
  fi
  echo "[attempt $ATTEMPT] EXIT=$EXIT  $(date)" >> train_v7_smoke_attempts.log
  if [ $EXIT -eq 0 ]; then
    echo "[attempt $ATTEMPT] SUCCESS" >> train_v7_smoke_attempts.log
    exit 0
  fi
  sleep 120
done
echo "[all attempts failed]" >> train_v7_smoke_attempts.log
exit 1
