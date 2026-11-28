#!/bin/bash
# 守望器 2：探测 4 卡同时空闲窗口（≥15GB × 4，间隔 15s 双重确认）→
# 杀掉当前 2 卡档训练 → 从最新 checkpoint 重启到 4 卡档（断点续训已验证）
# 训练完成（50000 final 存在）或 15k 评估运行中时自动退出/暂缓
cd /home/titans_v0.5/titans-memory

check_window() {
  local first=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
                | awk -F', ' -v m=15000 '$2 >= m {print $1}')
  sleep 15
  local second=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
                 | awk -F', ' -v m=15000 '$2 >= m {print $1}')
  local count=$(comm -12 <(echo "$first" | sort) <(echo "$second" | sort) | wc -l)
  [ "$count" -ge 4 ] && return 0 || return 1
}

while true; do
  # 训练整体完成 → 退出守望
  [ -f "checkpoints_v7/ckpt_step_0050000_final.pt" ] && { echo "[$(date)] training complete, watcher exit"; exit 0; }
  # 15k 评估运行中 → 暂缓重启（避免争抢）
  pgrep -f "evaluate_v7_[f]inal" > /dev/null && { echo "[$(date)] eval running, postpone"; sleep 300; continue; }

  if check_window; then
    echo "[$(date)] 4-GPU window detected — restarting training on 4 GPUs"
    # 杀掉旧启动器与训练（括号技巧防自匹配）
    pkill -f "run_v7_50[k].sh" 2>/dev/null
    pkill -f "train_mac_[v]7.py" 2>/dev/null
    sleep 20
    # 确认清干净
    pgrep -f "train_mac_[v]7.py" > /dev/null && { echo "[$(date)] old procs still alive, retry later"; sleep 120; continue; }
    # 从最新 checkpoint 重启（run_v7_50k.sh 会重新选卡——刚确认过窗口，大概率选到 4 卡）
    nohup ./run_v7_50k.sh > train_v7_50k_wrapper.log 2>&1 & disown
    echo "[$(date)] relaunched on 4 GPUs, watcher exit"
    exit 0
  fi
  sleep 300
done
