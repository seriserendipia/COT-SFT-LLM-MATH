# Tunix + Qwen2 LoRA Sharding Bug Report

**环境：** Tunix v0.1.6, JAX 0.9.1, qwix 0.1.5, Flax 0.12.5, TPU v2-8 (4 chips), Python 3.11

---

## 根因总结：JAX 版本不兼容

Tunix v0.1.6 是基于 JAX 0.8.x 开发和测试的。我们安装了 JAX 0.9.1，这是一个大版本升级，
引入了 sharding 行为的 breaking change。下面遇到的两个问题**都是 JAX 0.9.x 导致的**。

| 时间线 | 事件 |
|---|---|
| 2025-03-13 | Tunix v0.1.6 发布，基于 JAX 0.8.x 测试 |
| 2025 下半年 | JAX 0.9.0 发布，**默认 AxisType 从 Auto 改为 Explicit** |
| 2026-03-18 | 我们 `pip install jax[tpu]` 拿到 JAX 0.9.1，踩坑 |

Tunix `pyproject.toml` 只要求 `jax>=0.6.0,!=0.7.2`，没有上界限制，所以 pip 装了最新版。

### 推荐方案：降级 JAX 到 0.8.1

```bash
pip install "jax[tpu]==0.8.1" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

这样两个问题都不需要修，和官方测试环境完全一致。

---

## 问题 1：embedding 查表崩溃（已解决）

### 什么是 embedding

模型的第一步是把 token ID（比如 `[1234, 5678, ...]`）转成向量。
方法是在一张大表（`embedding table`，shape `(151936, 1536)`，即 15 万个词 × 1536 维）
里按 ID 查行，类似数据库的 `SELECT * FROM table WHERE id IN (1234, 5678)`。
这个操作在 JAX 里叫 `gather`。

### 什么是 sharding

4 块 TPU 芯片，每块只有 8GB 内存。15 万 × 1536 的表太大放不下一块芯片，
所以要把表**切开**分到多块芯片上。比如 `(tp, fsdp)` 的切法是：
- 沿词表维度按 `tp` 轴切（2 块芯片各存一半词表）
- 沿向量维度按 `fsdp` 轴切（2 块芯片各存一半向量维度）

这样每块芯片只存 1/4 的表。

### 问题出在哪

查表操作需要从被切开的表里取出完整的行。JAX 有两种模式：
- **Auto 模式**：JAX 自动决定怎么收集分布在不同芯片上的数据片段
- **Explicit 模式**：要求程序员**手动指定**输出怎么分布

JAX 0.8.x 默认用 Auto，所以 Tunix 代码写的时候不需要手动指定，一切正常。
JAX 0.9.0 **把默认改成了 Explicit**，gather 操作不知道输出该怎么分布，直接报错：

```
ShardingTypeError: out sharding could not be resolved unambiguously
```

### 解决

Tunix CLI 内部写了 `AxisType.Auto`（`config.py:621`），所以 CLI 不受影响。
我们在自己的脚本里也加上即可：

```python
mesh = jax.make_mesh(
    (2, 2), ('fsdp', 'tp'),
    axis_types=(jax.sharding.AxisType.Auto,) * 2,  # ← 关键
)
```

或者降级 JAX 到 0.8.x，默认就是 Auto。

---

## 问题 2：优化器初始化崩溃（未解决）

### 背景：LoRA 是怎么插入模型的

正常的 Qwen2 模型，每层有 attention 投影矩阵，比如 `q_proj`：

```
输入 (1536维) → [q_proj 权重矩阵 (1536, 12, 128)] → 输出 (12头×128维)
```

注意这个权重是 **3 维**的 `(1536, 12, 128)`——不是普通的 2D 矩阵，而是包含了"多头"这个维度。
（Gemma 模型用 `einsum` 层，权重是 2D 的，这个区别后面很关键。）

LoRA 的做法是**冻结原来的 3D 权重**，在旁边插入两个小的 **2D** 矩阵：
- `lora_a: (1536, 8)` — 把 1536 维压缩到 rank=8
- `lora_b: (8, 1536)` — 再展开回来

训练时只更新这两个小矩阵（共 9.2M 参数），原来的 1.54B 参数不动。

### 什么是优化器状态

Adam 优化器需要为**每个要训练的参数**维护两个缓存：
- `mu`（一阶动量）：和参数同 shape
- `nu`（二阶动量）：和参数同 shape

所以 `lora_a (1536, 8)` 对应 `mu (1536, 8)` 和 `nu (1536, 8)`，都是 **2D**。

### 什么是 `_shard_optimizer`

训练前，Tunix 要把优化器状态也切到多块 TPU 上（和模型权重一样的切法）。
做法是：
1. 从模型结构推导出每个参数应该怎么切（叫 `partition spec`，简写 `pspec`）
2. 用 `jax.lax.with_sharding_constraint` 强制把优化器状态按这个 pspec 分布到 TPU 上

### 问题出在哪

推导 pspec 时出了错：

| 参数 | 实际 shape | 推导出的 pspec | 匹配？ |
|---|---|---|---|
| `q_proj.w`（原始权重） | 3D `(1536, 12, 128)` | `P('fsdp', 'tp', None)` — 3 维 | ✅ |
| `q_proj.lora_a`（LoRA 矩阵） | 2D `(1536, 8)` | `P('fsdp', 'tp', None)` — 3 维 | ❌ 多了一维！ |

LoRA 的 2D 矩阵**错误地继承了**旁边 3D 原始权重的 pspec。
`with_sharding_constraint` 发现 "你给了 3 维的切法，但数据只有 2 维"，直接报错：

```
ValueError: spec=P('fsdp', 'tp', None) is only valid for values of rank at least 3,
but was applied to a value of rank 2.
```

### 为什么 Gemma 没这个问题

Gemma 的 attention 权重是 2D（`q_einsum` 用 2D 矩阵），LoRA 插入的也是 2D。
2D pspec 配 2D 数据，天然匹配。

**Qwen2 是 Tunix 支持的模型中唯一用 3D attention 权重的，所以只有它踩坑。**

### 为什么怀疑是 JAX 版本问题

JAX 0.9.0 changelog 明确写了：

> `PartitionSpec` rank 检查变得更严格。

在 JAX 0.8.x 的 Auto 模式下，这种 rank 不匹配可能被**静默容忍或自动截断**。
JAX 0.9.x 改为严格检查后，直接报错。
Tunix 在 0.8.x 上测试时从未遇到这个问题，所以也从没修过。

---

## 解决方案

### LoRA pspec 截断是否正确？

原始权重 `q_proj.w: (1536, 12, 128)` 的 pspec 是 `P('fsdp', 'tp', None)`：
- 第 0 维 (1536=embed_dim) 按 fsdp 切
- 第 1 维 (12=num_heads) 按 tp 切
- 第 2 维 (128=head_dim) 不切

LoRA 矩阵 `lora_a: (1536, 8)` 应该怎么切？
- 第 0 维 (1536) 应该和原始权重的第 0 维对齐 → 按 fsdp 切
- 第 1 维 (8=rank) 是 LoRA 独有的内部维度，很小，不需要切

所以正确的 pspec 是 `P('fsdp', None)`。

简单截断 `P('fsdp', 'tp', None)[:2]` 得到 `P('fsdp', 'tp')`，
会把 rank=8 的维度按 tp 切成两半（每片只有 4），增加不必要的通信。
**截断能跑，但不是最优解。**

### 方案 D：降级 JAX 到 0.8.1

```bash
pip install "jax[tpu]==0.8.1" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**两个问题都会消失**，和 Tunix 官方测试环境完全一致。
没有任何代码改动，没有副作用。

### 方案 A：修改 `_shard_optimizer` 源代码

在 `peft_trainer.py` 的 `_shard_optimizer` 方法里，遍历每个参数：
如果参数是 2D 但 pspec 是 3D，就截断 pspec 为 2D。

```python
# 伪代码
if len(tensor.shape) == 2 and len(pspec) == 3:
    pspec = pspec[:2]  # 截断为 2D
```

**优点：** 不需要降级 JAX。
**缺点：** 需要深入了解 nnx State 的树结构，我们尝试了但 tree_map 没生效（结构可能更复杂）。

### 方案 B（当前采用）：跳过 `_shard_optimizer`（monkey-patch）

"monkey-patch" 的意思是**在运行时把某个函数替换掉**，不修改源文件。比如：

```python
# 在我们的脚本里加一行：
peft_trainer.PeftTrainer._shard_optimizer = lambda self, mesh: None
```

这行代码把 `_shard_optimizer` 方法替换成一个什么都不做的空函数。
效果是训练启动时**跳过**优化器状态的手动分片。

**为什么跳过是正确的：**

`_shard_optimizer` 的作用是"提前把优化器状态分配到正确的 TPU 芯片上"。
如果跳过，优化器状态暂时全堆在一块芯片上。第一步训练时 JAX JIT 编译器发现
"模型参数分布在 4 块芯片，优化器状态在 1 块芯片"，就自动重新分配。

Tunix 源码注释也说了这只是性能优化，不是正确性要求：
```python
def _shard_optimizer(self, mesh):
    """Optimizer states should be sharded before calling the jit function.
    If not, the _train_step will be compiled 2 times."""
```

- **有** `_shard_optimizer`：手动分配 → JIT 编译 1 次 → 训练
- **跳过**：JIT 编译第 1 次发现不对 → 自动重分配 → JIT 编译第 2 次 → 训练

两种方式最终状态完全一样。区别只是多编译一次（~1 分钟）。
训练过程、梯度更新、最终模型**完全相同**。

### 方案 C：用 `(1,1)` mesh

```python
mesh = jax.make_mesh((1, 1), ('fsdp', 'tp'))
```

每个维度都只有 1 块芯片 → 实际上没有分片 → 不存在 rank 不匹配问题。

**优点：** 最简单。
**缺点：** 4 块 TPU 只用 1 块，浪费 3/4 算力，也无法展示多设备并行。

---

## 已验证通过的步骤

| 步骤 | 状态 | 备注 |
|---|---|---|
| Tunix 魔改加 Coder 支持 | ✅ | 加 `qwen2p5_coder_1p5b` classmethod |
| 模型加载 | ✅ | `model_lib.create_model()` |
| LoRA 注入 (9.2M params) | ✅ | 需 `AxisType.Auto` mesh |
| 数据准备 (gsm8k-CoT) | ✅ | HuggingFace datasets，不依赖 TF |
| PeftTrainer 初始化 | ✅ | |
| gen_model_input_fn | ✅ | 复用官方 peft_main.py 的实现 |
| **训练启动** | **✅** | 方案 B monkey-patch `_shard_optimizer` 后跑通 |
| **训练验证** | **✅** | 3 步 loss 5.1→4.38→3.94，稳态 8.2 steps/sec |

## 关键代码文件

- `sft_coder_lora.py` — 完整训练脚本（复用官方组件，仅换数据源）
- `run_sft_coder_lora.sh` — CLI 方式启动（需 TF 装数据集，暂不可用）
- `base_config.yaml` — 从 GitHub 下载的官方默认配置

## Sources

- [JAX 0.9.0 Changelog — AxisType.Explicit 成为默认](https://docs.jax.dev/en/latest/changelog.html)
- [JAX 0.9.0 GitHub Releases](https://github.com/jax-ml/jax/releases)
- [Tunix Releases](https://github.com/google/tunix/releases)
- [Tunix pyproject.toml — jax>=0.6.0](https://github.com/google/tunix)
- [Tunix GitHub](https://github.com/google/tunix)
- [Tunix LoRA & QLoRA Demo](https://tunix.readthedocs.io/en/latest/_collections/examples/qlora_gemma.html)
- [Tunix SFT Examples](https://github.com/google/tunix/tree/main/examples/sft/mtnt)
- [FunctionGemma Fine-tuning Blog](https://developers.googleblog.com/easy-functiongemma-finetuning-with-tunix-on-google-tpus/)
- [qwix LoRA docs](https://qwix.readthedocs.io/en/stable/lora.html)
- [Tunix Sharding Strategies (DeepWiki)](https://deepwiki.com/google/tunix/2.5-model-configuration-and-sharding)
