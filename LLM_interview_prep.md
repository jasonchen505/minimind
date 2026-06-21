# LLM算法实习面试准备手册

> 基于MiniMind项目代码深度整理，覆盖LLM & Agent应用及后训练核心考察点
> 适用：2028级MS在读，找LLM算法实习

---

## 目录

1. [项目介绍模板](#一项目介绍模板)
2. [模型架构深挖](#二模型架构深挖)
3. [训练技术深挖](#三训练技术深挖)
4. [RLHF/后训练深挖](#四rlhf后训练深挖)
5. [LoRA参数高效微调](#五lora参数高效微调)
6. [数据处理深挖](#六数据处理深挖)
7. [Agent & Tool Use](#七agent--tool-use)
8. [推理与采样策略](#八推理与采样策略)
9. [高频追问与开放题](#九高频追问与开放题)

---

## 一、项目介绍模板（1-2分钟）

### 开场白

> 我独立复现了一个64M参数的LLM全流程训练项目，覆盖预训练、SFT、LoRA微调、DPO强化学习和知识蒸馏。项目使用PyTorch原生实现所有核心算法，不依赖peft/trl等第三方高层抽象。通过这个项目，我深入理解了Transformer架构、RoPE位置编码、GQA注意力机制、SwiGLU激活函数等核心技术，并掌握了混合精度训练、梯度累积、DDP分布式训练等工程实践。

### 技术亮点（可选追问点）

1. **从零实现**：手写LoRA、DPO等算法，不依赖peft/trl库
2. **完整pipeline**：预训练 -> SFT -> RLHF -> 蒸馏全覆盖
3. **工程实践**：混合精度、梯度累积、断点续训、DDP分布式
4. **对齐Qwen3生态**：模型结构对齐主流开源模型

---

## 二、模型架构深挖

### 2.1 Transformer核心组件

#### Q1: 请介绍MiniMind的模型架构

**考察点**：对Transformer Decoder-Only架构的理解

**参考回答**：

MiniMind采用Transformer Decoder-Only结构，对齐Qwen3生态：

```
核心配置：
- hidden_size = 768
- num_hidden_layers = 8
- num_attention_heads = 8
- num_key_value_heads = 4 (GQA)
- vocab_size = 6400
- max_position_embeddings = 32768
- rope_theta = 1e6

关键组件：
1. RMSNorm (替代LayerNorm)
2. RoPE旋转位置编码 (支持YaRN外推)
3. GQA分组查询注意力 (8Q/4KV heads)
4. SwiGLU激活函数
5. MoE混合专家 (可选)
```

**追问**：
- 为什么用RMSNorm而不是LayerNorm？
- GQA相比MHA和MQA有什么优势？
- SwiGLU相比ReLU有什么改进？

---

### 2.2 RMSNorm vs LayerNorm

#### Q2: 为什么用RMSNorm？它和LayerNorm有什么区别？

**考察点**：归一化技术的理解

**代码位置**：`model/model_minimind.py:50-60`

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)
```

**回答要点**：

1. **LayerNorm**：`y = (x - mean) / sqrt(var + eps) * gamma + beta`
   - 需要计算均值和方差
   - 有centering操作（减均值）

2. **RMSNorm**：`y = x / sqrt(mean(x^2) + eps) * gamma`
   - 只计算RMS（Root Mean Square）
   - 去掉了均值中心化操作
   - 没有beta偏置

3. **优势**：
   - 计算更快（省去均值计算）
   - 训练更稳定
   - 参数更少（没有beta）

**深挖**：
- 为什么去掉centering操作仍然有效？（答：研究表明centering对最终效果影响不大，但计算开销明显）
- 在什么场景下LayerNorm可能优于RMSNorm？（答：RNN/Transformer Encoder中可能LayerNorm更稳定）

---

### 2.3 RoPE旋转位置编码

#### Q3: 请解释RoPE的原理，以及YaRN是如何实现长文本外推的？

**考察点**：位置编码的深入理解

**代码位置**：`model/model_minimind.py:62-84`

```python
def precompute_freqs_cis(dim, end, rope_base, rope_scaling):
    # 基础频率计算
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
    
    # YaRN外推
    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        
        if end / orig_max > 1.0:
            # 频率缩放
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin
```

**回答要点**：

1. **RoPE核心思想**：
   - 将位置信息编码到复数空间
   - 通过旋转实现相对位置编码
   - `q_m * k_n = Re[q_m * conj(k_n) * e^{i(m-n)theta}]`
   - 天然支持相对位置，无需学习

2. **RoPE公式**：
   ```
   f(x, m) = x * cos(m*theta) + rotate_half(x) * sin(m*theta)
   ```
   其中theta是频率，m是位置

3. **YaRN外推原理**：
   - 问题：超过训练长度时，高频分量失真
   - 解决：对不同频率分量采用不同缩放策略
   - 高频（beta_fast）：保持不变
   - 低频（beta_slow）：缩放factor倍
   - 中频：线性插值（ramp函数）

4. **rope_theta的作用**：
   - theta越大，支持的上下文越长
   - 1e6支持约32K上下文
   - Llama3使用5e5，Qwen2使用1e6

**深挖**：
- RoPE相比Sinusoidal位置编码有什么优势？（答：RoPE是相对位置编码，Sinusoidal是绝对位置编码；RoPE天然支持相对位置，泛化性更好）
- 为什么RoPE天然支持相对位置？（答：因为注意力分数只依赖于(m-n)*theta，即相对位置）
- YaRN的ramp函数有什么作用？（答：实现高频到低频的平滑过渡，避免突变）

---

### 2.4 GQA分组查询注意力

#### Q4: 请解释GQA的原理，以及它相比MHA和MQA的优势？

**考察点**：注意力机制的深入理解

**代码位置**：`model/model_minimind.py:91-134`

```python
class Attention(nn.Module):
    def __init__(self, config):
        self.num_key_value_heads = 4  # KV头数
        self.n_local_heads = 8        # Q头数
        self.n_rep = 2                # 重复系数 = 8/4
        
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
```

**回答要点**：

1. **三种注意力机制对比**：

| 机制 | KV头数 | 参数量 | 推理速度 | 效果 |
|------|--------|--------|----------|------|
| MHA | = Q头数 | 最大 | 最慢 | 最好 |
| GQA | < Q头数 | 中等 | 中等 | 接近MHA |
| MQA | 1 | 最小 | 最快 | 损失较大 |

2. **GQA原理**：
   - 每n个Q头共享1个KV头
   - MiniMind: 8Q/4KV，每2个Q共享1个KV
   - 通过`repeat_kv`函数复制KV

3. **优势**：
   - 减少KV缓存50%
   - 推理速度提升
   - 效果接近MHA
   - 比MQA更稳定

4. **代码实现**：
```python
def repeat_kv(x, n_rep):
    if n_rep == 1: return x
    return x[:, :, :, None, :].expand(..., n_rep, ...).reshape(...)
```

**深挖**：
- 为什么MQA效果损失较大？（答：所有Q头共享同一个KV，信息瓶颈太大）
- GQA的分组策略如何选择？（答：通常2:1或4:1，如Llama2-70B用8Q/4KV）
- 推理时KV缓存如何优化？（答：GQA减少KV缓存，还可以用PagedAttention、量化等）

---

### 2.5 SwiGLU激活函数

#### Q5: 请解释SwiGLU的原理，以及它相比ReLU的优势？

**考察点**：激活函数和FFN的理解

**代码位置**：`model/model_minimind.py:136-146`

```python
class FeedForward(nn.Module):
    def __init__(self, config):
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN['silu']

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

**回答要点**：

1. **SwiGLU公式**：
   ```
   SwiGLU(x) = (xW_gate * SiLU(xW_up)) W_down
   SiLU(x) = x * sigmoid(x)  # 也叫Swish
   ```

2. **相比ReLU的优势**：
   - 门控机制：gate_proj控制信息流
   - 平滑激活：SiLU比ReLU更平滑
   - 非单调：SiLU在负值区域有小的负值
   - 效果更好：实验表明提升1-2%

3. **FFN结构变化**：
   - 标准FFN：`W_down * ReLU(W_up * x)`
   - SwiGLU：`W_down * (SiLU(W_gate * x) * W_up * x)`
   - 参数量增加50%（多一个gate_proj）

4. **intermediate_size计算**：
```python
intermediate_size = math.ceil(hidden_size * math.pi / 64) * 64
# 768 -> 2048
```

**深挖**：
- 为什么SiLU比ReLU效果好？（答：平滑、非单调、门控机制）
- 门控机制的作用是什么？（答：选择性地传递信息，类似LSTM的门）
- GLU家族还有哪些变体？（答：ReGLU、GeGLU、SwiGLU等）

---

### 2.6 MoE混合专家

#### Q6: 请解释MoE的原理，以及负载均衡问题如何解决？

**考察点**：MoE架构的理解

**代码位置**：`model/model_minimind.py:148-176`

```python
class MOEFeedForward(nn.Module):
    def __init__(self, config):
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(config.num_experts)])

    def forward(self, x):
        # Gate路由
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok)
        
        # 负载均衡损失
        if self.training:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
```

**回答要点**：

1. **MoE核心思想**：
   - 稀疏激活：每个token只激活部分专家
   - 门控路由：Gate网络决定token分配给哪个专家
   - Top-K选择：选择K个专家

2. **MiniMind MoE配置**：
   - num_experts = 4
   - num_experts_per_tok = 1 (Top-1)
   - 比dense模型慢约50%

3. **负载均衡问题**：
   - 问题：专家负载不均，部分专家被过度使用
   - 解决：辅助损失（aux_loss）
   - 公式：`aux_loss = load * scores.mean() * num_experts * coef`

4. **训练时的trick**：
```python
# 防止未被选中的专家梯度消失
elif self.training:
    y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
```

**深挖**：
- MoE为什么推理比dense慢？（答：需要动态路由、无法批量计算、kernel启停开销）
- 如何设计更好的路由策略？（答：Expert Choice、Hash Routing、Soft MoE等）
- shared expert有什么作用？（答：提供通用知识，减少专家间的冗余）

---

## 三、训练技术深挖

### 3.1 预训练 vs SFT

#### Q7: 预训练和SFT有什么区别？数据格式和Loss计算有什么不同？

**考察点**：训练流程的理解

**代码位置**：
- 预训练：`dataset/lm_dataset.py:37-55`
- SFT：`dataset/lm_dataset.py:58-119`

**回答要点**：

1. **目标不同**：
   - 预训练：学习语言基础，词语接龙
   - SFT：学习对话格式，指令跟随

2. **数据格式不同**：
```python
# 预训练数据
{"text": "如何才能摆脱拖延症？"}

# SFT数据
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"}
    ]
}
```

3. **Loss计算不同**：
```python
# 预训练：所有token都计算loss
labels = input_ids.clone()
labels[input_ids == pad_token_id] = -100

# SFT：只在assistant回答位置计算loss
def generate_labels(input_ids):
    labels = [-100] * len(input_ids)
    # 找到assistant位置，只在这些位置计算loss
    ...
```

4. **学习率不同**：
   - 预训练：5e-4
   - SFT：1e-5（小50倍）

**深挖**：
- 为什么SFT只在assistant位置计算loss？（答：只训练模型生成回答，不训练"复读"用户输入）
- 预训练数据需要什么质量？（答：高质量、多样化、去重、清洗）
- SFT数据量多少合适？（答：质量比数量重要，通常几万到几十万条）

---

### 3.2 混合精度训练

#### Q8: 请解释混合精度训练的原理，以及GradScaler的作用？

**考察点**：训练工程实践

**代码位置**：`trainer/train_pretrain.py:119-137`

```python
# 设置混合精度
dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)

# GradScaler
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

# 训练循环
with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
scaler.step(optimizer)
scaler.update()
```

**回答要点**：

1. **混合精度原理**：
   - 前向传播：FP16/BF16计算
   - 反向传播：FP16/BF16梯度
   - 参数更新：FP32主权重

2. **为什么需要GradScaler**：
   - FP16表示范围小，梯度容易下溢
   - 先放大梯度（scale），更新后再缩小
   - 动态调整scale因子

3. **BF16 vs FP16**：
   - BF16：指数位更多，范围更大，不易溢出
   - FP16：精度更高，但容易溢出
   - 推荐使用BF16

4. **显存优化**：
   - 减少50%显存占用
   - 训练速度提升
   - Tensor Core加速

**深挖**：
- 什么时候用FP16，什么时候用BF16？（答：BF16优先，因为更稳定；FP16需要GradScaler）
- GradScaler的动态调整策略是什么？（答：如果连续N步没有inf/nan，增大scale；如果有inf/nan，减小scale）
- 混合精度会影响模型效果吗？（答：通常不会，甚至可能因为正则化效果略有提升）

---

### 3.3 梯度累积

#### Q9: 梯度累积的原理是什么？为什么需要梯度累积？

**考察点**：训练工程实践

**代码位置**：`trainer/train_pretrain.py:40-49`

```python
loss = loss / args.accumulation_steps
scaler.scale(loss).backward()

if step % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

**回答要点**：

1. **原理**：
   - 将大batch分成多个小batch
   - 累积梯度，等效于大batch训练
   - 每accumulation_steps步更新一次

2. **为什么需要**：
   - 显存限制：无法一次性放太大batch
   - 等效扩大batch size
   - 提高训练稳定性

3. **注意事项**：
   - loss需要除以accumulation_steps
   - 学习率可能需要相应调整
   - 每个micro-batch的梯度是平均的

**深挖**：
- 梯度累积和直接大batch训练有区别吗？（答：数学上等价，但BatchNorm等统计量可能有差异）
- accumulation_steps如何选择？（答：根据显存和目标batch size，通常2-8）
- 对BatchNorm有影响吗？（答：有，每个micro-batch的统计量不同；LLM通常用RMSNorm，无此问题）

---

### 3.4 学习率调度

#### Q10: 请解释学习率调度策略，Warmup和Cosine Decay的作用？

**考察点**：训练优化

**代码位置**：`trainer/trainer_utils.py:40-41`

```python
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```

**回答要点**：

1. **Warmup**：
   - 前10%步数线性增加学习率
   - 避免初期大学习率导致不稳定
   - 让模型先探索

2. **Cosine Decay**：
   - 学习率按余弦函数衰减
   - 初期衰减慢，后期衰减快
   - 比线性衰减效果更好

3. **公式**：
   ```
   lr = base_lr * (0.1 + 0.45 * (1 + cos(pi * step / total_steps)))
   ```

4. **为什么这样设计**：
   - 初期：大学习率快速收敛
   - 中期：适中学习率稳定学习
   - 后期：小学习率精细调整

**深挖**：
- Warmup的步数如何选择？（答：通常总步数的5%-10%）
- Cosine Decay vs 线性衰减？（答：Cosine通常更好，初期衰减慢有利于充分学习）
- 有没有更好的调度策略？（答：WSD、Inverse Sqrt等；可以提一下）

---

## 四、RLHF/后训练深挖

### 4.1 DPO vs PPO

#### Q11: 请比较DPO和PPO的原理和优缺点？

**考察点**：强化学习的理解

**代码位置**：`trainer/train_dpo.py:25-50`

```python
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # 计算log ratio
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```

**回答要点**：

1. **PPO流程**：
   - Actor生成回答
   - Reward Model打分
   - 计算优势函数（GAE）
   - 更新Actor（clipped objective）
   - KL散度约束

2. **DPO流程**：
   - 直接优化偏好
   - 无需训练Reward Model
   - 输入：(chosen, rejected)对
   - 最大化chosen概率，最小化rejected概率

3. **DPO损失函数**：
   ```
   L_DPO = -log sigmoid(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x)))
   ```
   其中y_w是chosen，y_l是rejected

4. **对比**：

| 特性 | PPO | DPO |
|------|-----|-----|
| 奖励模型 | 需要 | 不需要 |
| 训练稳定性 | 较差 | 较好 |
| 实现复杂度 | 复杂 | 简单 |
| 效果 | 接近 | 接近 |
| 在线采样 | 需要 | 不需要 |

**深挖**：
- DPO的beta参数有什么作用？（答：控制与参考模型的偏离程度，越大越保守）
- DPO有什么局限性？（答：离线数据、无法探索、对数据质量敏感）
- 什么时候用PPO，什么时候用DPO？（答：DPO适合快速迭代，PPO适合追求极致效果）

---

### 4.2 GRPO原理

#### Q12: 请解释GRPO的原理，以及它相比PPO的改进？

**考察点**：强化学习的深入理解

**代码位置**：`trainer/train_grpo.py:121-143`

```python
# 组内相对排序
grouped_rewards = rewards.view(-1, args.num_generations)
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
advantages = (rewards - mean_r) / (std_r + 1e-4)

# GRPO/CISPO损失
ratio = torch.exp(per_token_logps - old_per_token_logps)
if args.loss_type == "cispo":
    clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
    per_token_loss = -(clamped_ratio * advantages * per_token_logps - beta * per_token_kl)
else:
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    per_token_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages) - beta * per_token_kl)
```

**回答要点**：

1. **GRPO核心思想**：
   - 组内相对排序，无需绝对奖励值
   - 每个prompt生成多个回答（num_generations=6）
   - 组内计算优势函数

2. **优势函数计算**：
   ```
   advantage = (reward - mean(rewards)) / std(rewards)
   ```

3. **相比PPO的改进**：
   - 无需Critic模型（省去一个模型）
   - 训练更稳定（组内相对排序）
   - 实现更简单

4. **CISPO变体**：
   - 限制ratio上界（epsilon_high=5.0）
   - 更稳定的训练
   - 防止policy大幅偏离

**深挖**：
- 为什么组内相对排序更好？（答：减少奖励尺度的影响，更稳定）
- num_generations如何选择？（答：通常4-8，越大越稳定但越慢）
- KL惩罚的作用是什么？（答：防止policy偏离ref太远，保持通用能力）

---

### 4.3 Rollout引擎

#### Q13: 请解释Rollout引擎的作用，以及如何实现高效的采样？

**考察点**：推理优化的理解

**代码位置**：`trainer/rollout_engine.py`

**回答要点**：

1. **Rollout的作用**：
   - 生成回答样本
   - 计算log概率
   - 支持多种后端

2. **两种后端**：
   - Torch：PyTorch原生推理
   - SGLang：高性能推理引擎（支持RadixAttention、连续批处理）

3. **优化技巧**：
   - KV Cache：避免重复计算
   - 批量生成：提高吞吐
   - 异步采样：与训练重叠

4. **代码设计**：
```python
class RolloutEngine:
    def rollout(self, prompt_ids, num_generations, max_new_tokens, temperature):
        # 生成多个回答
        # 计算log概率
        # 返回结果
```

**深挖**：
- SGLang相比PyTorch推理有什么优势？（答：RadixAttention、连续批处理、CUDA Graph）
- 如何实现高效的在线采样？（答：异步采样、VLLM/SGLang后端）
- Rollout和训练如何并行？（答：异步rollout，采样和训练重叠）

---

## 五、LoRA参数高效微调

### 5.1 LoRA原理

#### Q14: 请解释LoRA的原理，以及为什么有效？

**考察点**：参数高效微调的理解

**代码位置**：`model/model_lora.py:6-18`

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))
```

**回答要点**：

1. **核心思想**：
   - 低秩分解：W' = W + delta_W = W + BA
   - 只训练A和B，冻结原始权重
   - r << d，参数量大幅减少

2. **为什么有效**：
   - 预训练权重已经很好
   - 微调只需要小幅调整
   - 低秩假设：更新矩阵是低秩的（Aghajanyan et al., 2020）

3. **初始化策略**：
   - A：高斯初始化（打破对称性）
   - B：全零初始化（初始时delta_W=0，不破坏预训练权重）

4. **参数效率**：
```python
# MiniMind LoRA统计
LLM 总参数量: 64.305 M
LoRA 参数量: 0.393 M
LoRA 参数占比: 0.61%
```

**深挖**：
- rank如何选择？（答：通常8-64，根据任务复杂度和数据量）
- 应该在哪些层加LoRA？（答：通常Q/K/V/O，也可以加FFN）
- LoRA有什么局限性？（答：表达能力有限，复杂任务可能不够）

---

### 5.2 LoRA实现细节

#### Q15: LoRA的实现有哪些关键点？如何合并权重？

**考察点**：工程实现能力

**代码位置**：`model/model_lora.py:21-64`

```python
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        # 只在方阵的Linear层加LoRA
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank)
            setattr(module, "lora", lora)
            
            # 修改forward
            original_forward = module.forward
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)
            module.forward = forward_with_lora

def merge_lora(model, lora_path, save_path):
    # 合并权重：W' = W + BA
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            state_dict[f'{name}.weight'] += (module.lora.B.weight @ module.lora.A.weight)
```

**回答要点**：

1. **应用LoRA**：
   - 遍历所有Linear层
   - 只在方阵上加LoRA（Q/K/V/O）
   - 修改forward函数（hook方式）

2. **合并权重**：
   - W' = W + BA
   - 推理时无需额外计算
   - 无额外延迟

3. **缩放因子**：
   - 通常有alpha/r的缩放
   - 控制LoRA的影响程度

**深挖**：
- 为什么只在方阵上加LoRA？（答：Q/K/V/O投影通常是方阵，FFN不是；但也可以加FFN）
- LoRA的rank和alpha如何调优？（答：alpha通常=2*rank，通过验证集调优）
- 推理时LoRA如何部署？（答：合并权重后与原模型一样部署；也可以动态切换LoRA）

---

## 六、数据处理深挖

### 6.1 Loss Mask

#### Q16: SFT训练中Loss Mask的作用是什么？如何实现？

**考察点**：数据处理的理解

**代码位置**：`dataset/lm_dataset.py:88-104`

```python
def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)
    i = 0
    while i < len(input_ids):
        # 找到assistant的开始位置
        if input_ids[i:i + len(self.bos_id)] == self.bos_id:
            start = i + len(self.bos_id)
            end = start
            # 找到assistant的结束位置
            while end < len(input_ids):
                if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                    break
                end += 1
            # 只在assistant位置计算loss
            for j in range(start, min(end + len(self.eos_id), self.max_length)):
                labels[j] = input_ids[j]
            i = end + len(self.eos_id)
        else:
            i += 1
    return labels
```

**回答要点**：

1. **为什么需要Loss Mask**：
   - 只训练模型生成回答
   - 不训练模型学习"复读"用户输入
   - 避免user部分干扰梯度

2. **实现方式**：
   - -100表示不计算loss
   - cross_entropy会忽略-100位置
   - 只在assistant位置计算

3. **边界处理**：
   - 包含assistant开始标记
   - 包含assistant结束标记
   - 处理截断情况

**深挖**：
- 为什么用-100而不是0？（答：PyTorch cross_entropy的ignore_index参数默认-100）
- 预训练需要Loss Mask吗？（答：不需要，所有token都计算loss）
- 多轮对话如何处理？（答：每轮assistant都计算loss，user/system/tool不计算）

---

### 6.2 数据增强

#### Q17: SFT数据有哪些增强技巧？

**考察点**：数据工程

**代码位置**：`dataset/lm_dataset.py:9-35`

```python
def pre_processing_chat(conversations, add_system_ratio=0.2):
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        ...
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content
```

**回答要点**：

1. **System Prompt增强**：
   - 20%概率添加system prompt
   - 多种候选system prompt
   - 增加模型对system指令的适应性

2. **思考标签处理**：
   - 80%概率移除空思考标签
   - 避免模型学习无意义的思考模式
   - 支持自适应思考

3. **其他技巧**：
   - 多轮对话拼接
   - 长度截断
   - 数据清洗和去重
   - 蒸馏数据合成

---

## 七、Agent & Tool Use

### 7.1 Tool Call实现

#### Q18: 请解释Tool Call的实现原理？

**考察点**：Agent能力的理解

**数据格式**：
```json
{
    "conversations": [
        {"role": "system", "content": "# Tools ...", "tools": "[...]"},
        {"role": "user", "content": "帮我算一下 256 乘以 37"},
        {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"calculate_math\",\"arguments\":{\"expression\":\"256 * 37\"}}]"},
        {"role": "tool", "content": "{\"result\":\"9472\"}"},
        {"role": "assistant", "content": "256 乘以 37 等于 9472。"}
    ]
}
```

**回答要点**：

1. **Chat Template**：
   - `<tool_call>{name, arguments}</tool_call>`
   - `<tool_response>{result}</think>`
   - 通过tokenizer的apply_chat_template实现

2. **训练数据**：
   - 混入SFT数据中
   - 约10万条tool call数据
   - 覆盖约10个模拟工具

3. **推理流程**：
   - 模型输出`<tool_call>`标签
   - 解析工具名和参数
   - 调用工具获取结果
   - 将结果放入`<tool_response>`标签
   - 继续生成最终回答

4. **关键设计**：
   - tools挂在system消息上
   - tool_calls挂在assistant消息上
   - 训练时自动展开为标签格式

**深挖**：
- 如何处理工具调用失败？（答：返回错误信息，让模型重试或解释）
- 多轮工具调用如何实现？（答：循环调用直到模型不再输出tool_call）
- 如何保证工具调用的准确性？（答：工具描述清晰、few-shot示例、约束解码）

---

### 7.2 自适应思考

#### Q19: 请解释自适应思考（Adaptive Thinking）的实现？

**考察点**：Reasoning能力的理解

**回答要点**：

1. **open_thinking开关**：
   - `open_thinking=0`：注入空的`<think>\n\n</think>`，直接回答
   - `open_thinking=1`：注入`<think>`起始标签，模型继续输出思考过程

2. **训练策略**：
   - 混合空think和显式reasoning
   - thinking_ratio控制开启概率
   - 让模型学会"该想时想、该直答时直答"

3. **推理时切换**：
```python
# CLI推理
python eval_llm.py --open_thinking 1

# API调用
response = client.chat.completions.create(
    model="minimind",
    messages=[{"role": "user", "content": "你是谁？"}],
    extra_body={"chat_template_kwargs": {"open_thinking": True}}
)
```

**深挖**：
- 为什么不用单独的reason模型？（答：统一模型更灵活，通过模板控制）
- thinking_ratio如何选择？（答：通常0.5-0.9，根据任务需求）
- 思考标签对模型效果有什么影响？（答：可能提升推理能力，但增加输出长度）

---

## 八、推理与采样策略

### 8.1 生成策略

#### Q20: 请解释LLM推理时的采样策略？

**考察点**：推理优化

**代码位置**：`model/model_minimind.py:257-288`

```python
@torch.inference_mode()
def generate(self, inputs, max_new_tokens, temperature, top_p, top_k, ...):
    for _ in range(max_new_tokens):
        logits = outputs.logits[:, -1, :] / temperature
        
        # repetition_penalty
        if repetition_penalty != 1.0:
            seen = torch.unique(input_ids[i])
            logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
        
        # top_k filtering
        if top_k > 0:
            logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
        
        # top_p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
            logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
        
        # sampling
        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
```

**回答要点**：

1. **Temperature**：
   - 控制随机性
   - 越大越随机，越小越确定
   - 通常0.7-1.0

2. **Top-K采样**：
   - 只保留概率最高的K个token
   - K=50是常见选择

3. **Top-P (Nucleus) 采样**：
   - 只保留累积概率超过P的token
   - P=0.95是常见选择
   - 比Top-K更自适应

4. **Repetition Penalty**：
   - 惩罚已出现的token
   - 减少重复生成
   - 通常1.0-1.2

5. **KV Cache**：
   - 缓存已计算的K/V
   - 避免重复计算
   - 推理速度提升

**深挖**：
- Temperature和Top-P如何配合？（答：先Temperature缩放，再Top-P过滤）
- 什么时候用greedy，什么时候用sampling？（答：greedy用于确定性任务，sampling用于创造性任务）
- 如何减少重复生成？（答：repetition_penalty、presence_penalty、frequency_penalty）

---

## 九、高频追问与开放题

### 9.1 Scaling Law相关

#### Q21: 小模型和大模型有什么区别？Scaling Law在小模型上还适用吗？

**回答要点**：

1. **小模型特点**：
   - 参数少，容量有限
   - 训练快，迭代快
   - 适合验证想法

2. **Scaling Law**：
   - 经典Scaling Law：Loss = aN^(-b) + cD^(-d) + E
   - 小模型可能不完全遵循
   - MobileLLM发现：深度比宽度更重要

3. **MiniMind的选择**：
   - dim=768, n_layers=8（矮胖子）
   - 更浅训练更快，同时dim不至于过小

**深挖**：
- 如何选择模型大小？（答：根据算力预算和任务需求）
- 深度和宽度如何权衡？（答：MobileLLM：深度更重要；但dim<512时宽度优先）

---

### 9.2 训练稳定性

#### Q22: 训练中遇到loss震荡或NaN怎么办？

**回答要点**：

1. **Loss震荡**：
   - 降低学习率
   - 增加batch size
   - 检查数据质量
   - 增加warmup步数

2. **NaN/Inf**：
   - 检查梯度范数
   - 降低学习率
   - 使用BF16代替FP16
   - 检查数据是否有异常值
   - 使用梯度裁剪

3. **过拟合**：
   - 增加数据量
   - 使用正则化（dropout、weight decay）
   - 早停

**深挖**：
- 如何监控训练稳定性？（答：loss、grad_norm、learning_rate、token/s）
- 梯度裁剪的作用是什么？（答：防止梯度爆炸，通常clip=1.0）

---

### 9.3 模型评估

#### Q23: 如何评估LLM的效果？

**回答要点**：

1. **自动评估**：
   - PPL (Perplexity)：语言模型困惑度
   - BPB (Bits Per Byte)：跨tokenizer可比
   - 下游任务准确率

2. **人工评估**：
   - 对话质量
   - 事实准确性
   - 安全性

3. **Benchmark**：
   - C-Eval：中文综合评测
   - MMLU：多任务语言理解
   - HumanEval：代码生成

4. **MiniMind评估**：
   - 自动测试：预设问题
   - 手动输入：交互式测试
   - 生成速度：tokens/s

**深挖**：
- PPL和效果有什么关系？（答：PPL越低通常效果越好，但不是绝对）
- 如何评估对话模型？（答：多维度评估：有用性、安全性、流畅性）

---

### 9.4 工程实践

#### Q24: 如何优化LLM训练速度？

**回答要点**：

1. **数据加载**：
   - 多进程DataLoader
   - 预加载到内存
   - 使用内存映射

2. **计算优化**：
   - 混合精度训练
   - 梯度累积
   - torch.compile
   - Flash Attention

3. **分布式训练**：
   - DDP数据并行
   - FSDP/DeepSpeed模型并行
   - Pipeline并行

4. **通信优化**：
   - 梯度压缩
   - 异步通信
   - 重叠计算和通信

**深挖**：
- torch.compile有什么作用？（答：JIT编译，减少Python开销，提升10-30%速度）
- Flash Attention的原理是什么？（答：分块计算，减少HBM访问，IO-aware）

---

### 9.5 Agent相关

#### Q25: LLM Agent的核心组件是什么？

**回答要点**：

1. **规划（Planning）**：
   - 任务分解
   - 反思与改进
   - 思维链（CoT）

2. **记忆（Memory）**：
   - 短期记忆：上下文窗口
   - 长期记忆：向量数据库
   - 工作记忆：中间推理

3. **工具使用（Tool Use）**：
   - 代码执行
   - API调用
   - 搜索引擎

4. **行动（Action）**：
   - 生成回答
   - 调用工具
   - 与环境交互

**深挖**：
- 如何让LLM学会使用工具？（答：SFT数据中混入tool call样本）
- Agent RL如何训练？（答：多轮交互、奖励信号、GRPO/PPO）
- ReAct和CoT有什么区别？（答：ReAct有行动，CoT只有思考）

---

## 附录：常见面试问题清单

### 模型架构类
1. 请介绍Transformer的架构
2. 为什么用RMSNorm而不是LayerNorm？
3. RoPE的原理是什么？
4. GQA相比MHA有什么优势？
5. SwiGLU相比ReLU有什么改进？
6. MoE如何实现负载均衡？

### 训练技术类
7. 预训练和SFT有什么区别？
8. 混合精度训练的原理是什么？
9. 梯度累积的作用是什么？
10. 学习率调度策略有哪些？

### RLHF/后训练类
11. DPO和PPO有什么区别？
12. GRPO的原理是什么？
13. 如何设计奖励模型？
14. KL惩罚的作用是什么？

### LoRA/微调类
15. LoRA的原理是什么？
16. LoRA的rank如何选择？
17. 如何合并LoRA权重？

### 数据处理类
18. SFT为什么需要Loss Mask？
19. 数据增强有哪些技巧？
20. 如何保证数据质量？

### 推理优化类
21. KV Cache的原理是什么？
22. Top-P和Top-K采样有什么区别？
23. 如何减少重复生成？

### Agent类
24. Tool Call如何实现？
25. Agent RL如何训练？
26. ReAct和CoT有什么区别？

---

## 附录：代码位置速查表

| 问题 | 代码位置 |
|------|----------|
| RMSNorm | `model/model_minimind.py:50-60` |
| RoPE | `model/model_minimind.py:62-84` |
| GQA Attention | `model/model_minimind.py:91-134` |
| SwiGLU FFN | `model/model_minimind.py:136-146` |
| MoE | `model/model_minimind.py:148-176` |
| LoRA | `model/model_lora.py:6-18` |
| 预训练数据 | `dataset/lm_dataset.py:37-55` |
| SFT数据 | `dataset/lm_dataset.py:58-119` |
| DPO数据 | `dataset/lm_dataset.py:122-192` |
| DPO Loss | `trainer/train_dpo.py:25-50` |
| GRPO训练 | `trainer/train_grpo.py:71-204` |
| 学习率调度 | `trainer/trainer_utils.py:40-41` |
| 生成函数 | `model/model_minimind.py:257-288` |

---

*文档生成时间：2026年5月*
*基于MiniMind项目：https://github.com/jingyaogong/minimind*
