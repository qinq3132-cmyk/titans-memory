# 基于单器件 Fast–Slow Dynamics 的 Titans Long-Term Memory 方案

## 核心思想

不再分别用一套易失性阵列保存 historical surprise \(S\)、另一套非易失性阵列保存 long-term weight \(W\)。  
而是利用**同一个忆阻器内部的快、慢两类动态状态**：

```text
G_ij = Slow persistent state + Fast transient state
             ↑                      ↑
       long-term weight W      historical surprise
```

Fast state 保存近期更新痕迹并自然 relaxation；Slow state 保存真正长期稳定的记忆。

---

## 更新规则

每次得到 \(\Delta W_{ij}\) 后，先做 ternary quantization：

```text
ΔW_ij > +θ   →   +1 pulse
ΔW_ij < -θ   →   -1 pulse
otherwise    →    no operation
```

不要求器件精确实现高精度 analog \(\Delta W\)，只保留：

- 更新方向；
- 是否值得更新。

---

## Fast state：累积与衰减

一次孤立脉冲：

```text
+1 pulse
   ↓
Fast state 增大
   ↓
没有后续刺激
   ↓
自然 relaxation
   ↓
该变化逐渐消失
```

因此，小幅、偶发更新不会立即永久改变 long-term memory。

---

## Repeated update：从 Fast 到 Slow 的巩固

如果在 relaxation 时间窗口内持续出现同方向更新：

```text
+1   +1   +1   +1
 ↓    ↓    ↓    ↓
Fast state 持续累积
          ↓
达到 consolidation condition
          ↓
形成 persistent conductance change
          ↓
Slow state 改变
          ↓
Long-term W 真正更新
```

于是：

- 偶发更新 → 衰减掉；
- 持续重复更新 → 被巩固为长期记忆。

整个过程发生在**同一个器件**里，不需要：

```text
S array → read → threshold → transfer → W array
```

---

## 与 Titans 的对应关系

原始 Titans：

```text
current ΔW
    ↓
historical surprise S
    ↓
decay / momentum accumulation
    ↓
update long-term memory W
```

本方案：

```text
ternary ΔW pulse
       ↓
single dynamic memristor
       ↓
fast-state accumulation
       ↓
natural relaxation / repeated reinforcement
       ↓
slow persistent change
       ↓
long-term W
```

对应关系：

| Titans | 单器件物理实现 |
|---|---|
| Current update | +1 / -1 / 0 pulse |
| Historical surprise | Fast transient state |
| Surprise decay | Natural relaxation |
| Repeated surprise | Repeated same-direction pulses |
| Consolidation | Fast-to-slow transition |
| Long-term memory | Persistent conductance state |

---

## 为什么不能简单说“利用 RRAM drift”

普通 RRAM drift 往往意味着：

```text
已经保存好的 W
      ↓
随时间漂移
      ↓
长期记忆被破坏
```

这通常是非理想性。

我们真正希望利用的是：

```text
新产生的小变化
      ↓
暂时存在并 relaxation

持续重复的变化
      ↓
逐渐变稳定
      ↓
形成长期记忆
```

因此更准确的表述应是：

> **Fast-relaxation / Slow-consolidation dynamics**

或：

> **Activity-dependent retention**

而不是简单的 conductance drift。

---

## 为提高可扩展性而采用的两个原则

### 1. 不依赖精确 decay 数值

不要求所有器件有完全相同的衰减曲线，只要求：

```text
pulse interval < τ
→ 更新能够累积

pulse interval >> τ
→ transient state 基本消失
```

算法利用的是“时间窗口”，而不是精确的器件 decay 数值。

### 2. 使用 ternary update

不要求：

```text
ΔW = 0.037 → 器件精确变化 0.037
```

而只要求：

```text
significant positive update → +1 pulse
significant negative update → -1 pulse
small update → no pulse
```

算法主要依赖：

> **方向 + 更新频率 + 时间相关性**

---

## 核心优势

相比原始 Titans 显式维护 \(S\)，该方案可以：

- 不需要额外的 surprise-state memory；
- 不需要反复读取/写回 \(S\)；
- 不需要数字实现 momentum / decay；
- 不需要两套阵列之间搬运历史更新；
- 小幅、瞬时更新不会立刻变成长期记忆；
- 只有重复的重要更新才逐渐 consolidate；
- 更充分利用忆阻器本身的时间动力学。

真正的创新点不是：

> 用忆阻器加速 Titans

而是：

> **利用单器件 fast–slow internal dynamics，重新实现 history-dependent long-term memory update。**

---

## 当前最大风险

方案能否扩展到大规模阵列，主要取决于：

- relaxation time \(	au\) 的器件间分布；
- fast-to-slow transition threshold 的一致性；
- potentiation / depression 的非对称性；
- 温度和状态对 dynamics 的影响；
- 是否能通过 tile-level calibration 或 hardware-aware training 吸收 variation。

第一阶段最重要的验证应当是：

> **在真实或拟真的器件动力学 variation 下，ternary pulse + relaxation + consolidation 是否还能稳定实现 history-dependent memory update。**

---

## 一句话总结

> **将 Titans 中显式的“ΔW history → surprise state S → decay → W”重构为“ternary ΔW pulses → single-device fast-state accumulation / relaxation → slow-state consolidation”，用一个动态忆阻器单元同时承担短期更新痕迹和长期记忆。**
