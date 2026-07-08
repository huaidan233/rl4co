# LUOP 模型改造与稳定性报告

## 背景与目标

本轮工作围绕土地利用优化问题（LUOP）进行模型改造和稳定性调试。原始需求有四个核心点：将动作从“选择地块”改成“选择地块 + 选择类型”的双动作形式；保证动作满足土地利用约束；支持多目标优化与 Pareto 前沿评估；在可用 RTX 4090 上完成训练稳定性验证，同时避免显存爆掉。

最终实现没有继续把 LUOP 特化逻辑塞进通用 `zoo/am`，而是在 `rl4co/models/zoo/luop_am/` 下建立了独立模型入口。这样可以保留原 RL4CO Attention Model 的公共能力，同时把 LUOP 的双动作解码、约束检查、多目标指标和稳定性策略隔离开，避免污染通用路由/调度模型。

## 代码结构

主要改动集中在以下位置：

- `rl4co/envs/urbanplan/cityplan/env.py`：LUOP 环境、联合动作编码/解码、动作 mask、约束检查、目标权重与 reward components。
- `rl4co/models/zoo/luop_am/`：LUOP 专用 Attention Model、policy、decoder 和 constructive decoding 逻辑。
- `rl4co/models/nn/env_embeddings/`：LUOP 初始、上下文和动态 embedding，加入目标权重、约束压力、剩余缺口和 pending type 信息。
- `rl4co/utils/multi_objective.py`：Pareto 前沿、hypervolume、候选权重网格评估与动作 artifact 校验。
- `scripts/luop_4090_stability.py`：4090 显存保护、稳定性扫描、Hydra 配置检查和 adaptive batch runner。
- `tests/test_luop_joint_multiobjective.py` 与 `tests/test_luop_4090_runner.py`：双动作、多目标、mask、replay、Pareto、baseline 和 4090 runner 回归测试。

## 双动作建模

LUOP 现在支持两种等价动作表示：

- flat action: `action = type * num_loc + parcel`
- dual action: `type_action` 与 `parcel_action`

环境提供 `encode_action()` 和 `decode_action()`，用于在 flat 与 dual 之间转换。`step()` 既接受 flat `action`，也接受显式的 `type_action + parcel_action`。显式 dual action 会经过同一套 action mask 与约束检查，因此不会绕开可行性约束。

Policy 输出中也会保留动作 artifact：

- `actions`
- `type_actions`
- `parcel_actions`

对于已经完成的行，padding 使用 `-1`，并且 replay、likelihood、entropy 与 Pareto 评估都显式处理该 padding。

## 约束与 Mask

环境维护三层 mask：

- `parcel_action_mask`：哪些地块仍可选择。
- `type_action_mask`：当前状态下哪些类型至少有一个可行地块。
- `type_parcel_action_mask` / `action_mask`：类型-地块联合动作是否可行。

约束逻辑在 `_build_action_masks()` 中生成，并覆盖固定地块、比例上下界、组约束、剩余面积、剩余大地块数量等条件。Type-first 解码时，第一步选择 type；随后 decoder 将该 type 写入 `pending_type_action`，第二步 parcel mask 会读取 pending type，从而保证 parcel 选择能“看到”刚选的类型。

Replay 路径也做了防御性校验：动作长度、batch shape、整数类型、范围、done-row padding、flat 与 dual 编码一致性、当前 mask 可行性都会检查。发现不合法动作时，会在 policy 或 Pareto 工具层提前抛出可读错误，而不是等环境内部断言崩掉。

## LUOP 专用 Attention Model

新模型入口是：

```yaml
_target_: rl4co.models.LUOPAttentionModel
```

对应文件为：

- `rl4co/models/zoo/luop_am/model.py`
- `rl4co/models/zoo/luop_am/policy.py`
- `rl4co/models/zoo/luop_am/decoder.py`
- `rl4co/models/zoo/luop_am/constructive.py`

`LUOPAttentionModelDecoder` 在原 AM pointer 逻辑上增加了 type logits、type embedding 和 type-parcel joint query。最终 logits 仍展开成 `[batch, num_types * num_loc]`，因此可以兼容原 constructive decoding 框架；同时 policy 层可以将其拆成 type 与 parcel 两阶段解码。

默认配置位于 `configs/experiment/cityplan/am.yaml`。当前默认使用：

```yaml
model:
  batch_size: 2048
  val_batch_size: 2048
  test_batch_size: 2048
  optimizer_kwargs:
    lr: 5e-5
  policy_kwargs:
    train_decode_type: sampling
    val_decode_type: greedy
    test_decode_type: greedy
    decode_type_first: false
```

LUOP200 是显存边界场景，实际运行时通过命令行覆盖为 `batch_size=256`。

## 多目标与 Pareto 评估

奖励现在保留两个 component：

- `compatibility_reward`
- `accessibility_reward`

标量 reward 由 `objective_weights` 对 `reward_components` 加权得到。权重会归一化，并拒绝负数、全零、维度错误或 batch shape 不匹配的输入。

评估阶段使用固定权重 `[0.5, 0.5]`，避免验证曲线因为随机目标权重而抖动。Pareto 评估使用权重网格：

```yaml
pareto_eval_weights:
  - [1.0, 0.0]
  - [0.75, 0.25]
  - [0.5, 0.5]
  - [0.25, 0.75]
  - [0.0, 1.0]
pareto_reference: [0.0, 0.0]
```

验证和测试会记录：

- `reward`
- `compatibility_reward`
- `accessibility_reward`
- `pareto_hypervolume`
- `pareto_front_size`
- `checkpoint_score`

`checkpoint_score` 当前定义为 scalar reward、accessibility reward、Pareto hypervolume 的均值，用于 checkpoint monitor：

```yaml
callbacks:
  model_checkpoint:
    monitor: "val/checkpoint_score"
```

这样避免单看 `val/reward` 或单看 Pareto hypervolume 时忽略 component collapse。

## Embedding 调整

LUOP embedding 不再只看静态地块特征。当前 LUOP init/context/dynamic embedding 会纳入：

- 地块坐标与面积比例。
- 当前 plan 与未分配标记。
- `objective_weights`，让策略知道当前目标偏好。
- `constraint_pressure` 和 remaining deficit，用于表达约束压力。
- `pending_type_action`，让 type-first 的 parcel 阶段能感知刚选择的类型。
- previous selected parcel 标记，用于动态状态。

相关测试覆盖了 objective weights、constraint pressure、remaining deficit、pending type 和 shaped batch fallback。

## Baseline 稳定性修复

训练不稳定的一个真实来源在 rollout baseline：`RolloutBaseline.rollout()` 在 epoch-end baseline challenge 时会把 live candidate policy 切到 eval mode，但之前没有恢复原训练状态。这可能导致后续 epoch 在 eval-mode encoder 行为下训练。

现在 rollout 路径在 `finally` 中恢复 live policy 的原始 training mode；baseline policy 副本仍保持 eval mode。对应回归测试为：

```bash
python -m pytest tests/test_luop_joint_multiobjective.py::test_luop_rollout_baseline_rollout_restores_policy_training_mode -q
```

另外，baseline 更新现在复用固定 challenge dataset，避免 t-test 与新 baseline 均值在不同 LUOP 实例上比较；warmup 在 `alpha == 1` 前保持 exponential/mixed baseline 路径，避免部分 warmup 被提前绕过。

## 4090 Batch 与显存策略

常规 LUOP 验证默认使用 RTX 4090，并采用已验证的大 batch：

| 问题规模 | 默认 batch | 说明 |
| --- | ---: | --- |
| LUOP50 | 2048 | 常规 4090 大 batch |
| LUOP100 | 1024 | 常规 4090 大 batch |
| LUOP200 | 256 | 显存边界场景，保持保守 |

不要在 LUOP200 上直接使用 `batch_size=512`。真实 probe 曾达到约 `24073 / 24564 MiB` device memory；考虑共享显存压力后，该路径可能耗尽机器内存。

稳定性 runner 内置保护：

- LUOP200 `batch_size > 256` 会在启动前拒绝。
- 默认 runtime cap 为 `22000 MiB`。
- 如果触发 CUDA OOM 或显存 cap，adaptive runner 只在纯 OOM / cap 场景下尝试更小 batch。
- 命令使用 `sys.executable`，兼容远端 `.venv4090/bin/python` 环境。

常用命令：

```bash
python scripts/luop_4090_stability.py --suite --log-dir logs/luop_4090_stability_resume
```

单独检查 LUOP200：

```bash
python scripts/luop_4090_stability.py \
  --num-loc 200 \
  --target-batch 256 \
  --min-batch 128 \
  --log-dir logs/luop_4090_memory_probe
```

完整稳定性验证：

```bash
python scripts/luop_4090_stability.py \
  --num-loc 200 \
  --target-batch 256 \
  --min-batch 128 \
  --max-epochs 20 \
  --train-data-size 8192 \
  --eval-data-size 2048 \
  --full \
  --min-validation-points 16 \
  --log-dir logs/luop200_4090_bs256_20ep_full_20260703
```

## 稳定性扫描规则

Runner 会扫描日志、CSV metrics 和 Hydra config。主要拒绝条件包括：

- CUDA OOM、NaN、Inf、非法概率分布。
- all-false mask、无合法联合动作、不可行动作。
- flat / dual action 越界、非整数、padding 错误、replay 与当前 mask 不一致。
- Pareto action artifact 缺失、shape 错误、非 suffix padding、dual 与 flat 不一致。
- validation/test 未使用固定 eval weights。
- checkpoint monitor 不是 `val/checkpoint_score`。
- `val/reward`、component reward、Pareto hypervolume 或 checkpoint score 出现超过阈值的 post-best drop 或振荡。

Scanner 允许一个很窄的“补偿型目标重平衡”情况：如果某个 component 小幅下降，但 scalar reward、Pareto hypervolume、checkpoint score 均稳定或提升，且两个 component 的差距缩小，则不视为 collapse。它仍会拒绝无补偿的 component collapse。

## 验证结果

### 本地测试

完整测试命令：

```bash
python -m pytest tests -q
```

最新结果：

```text
267 passed, 27 warnings
```

Focused LUOP 测试也通过：

```bash
python -m pytest tests/test_luop_joint_multiobjective.py tests/test_luop_4090_runner.py -q
```

结果：

```text
264 passed, 27 warnings
```

### 4090 Smoke Suite

4090 smoke suite 扫描均为 `[]`：

| Run | Batch | Peak memory | Scan |
| --- | ---: | ---: | --- |
| LUOP50 linear | 2048 | 8169/8277 MiB | `[]` |
| LUOP100 linear | 1024 | 12521 MiB | `[]` |
| LUOP200 linear | 256 | 11873 MiB | `[]` |
| LUOP50 Chebyshev | 2048 | 8169 MiB | `[]` |

### 4090 八轮诊断

```text
logs/luop50_4090_bs2048_8ep_20260703
val/reward: 0.4876 -> 0.5880
val/checkpoint_score: 0.3322 -> 0.4439

logs/luop100_4090_bs1024_8ep_20260703
val/reward: 0.5178 -> 0.6146
val/checkpoint_score: 0.3372 -> 0.4642

logs/luop200_4090_bs256_8ep_20260703
val/reward: 0.5405 -> 0.6447
val/checkpoint_score: 0.3567 -> 0.4806
```

### LUOP200 二十轮完整验证

运行目录：

```text
logs/luop200_4090_bs256_20ep_full_20260703
```

关键配置：

```text
batch_size=256
train_data_size=8192
eval_data_size=2048
fixed eval weights=[0.5, 0.5]
checkpoint monitor=val/checkpoint_score
```

结果：

```text
status=0
scanner failures=[]
gpu_peak_memory_mib=11795
best checkpoint=epoch_019.ckpt
```

验证曲线：

```text
val/reward:
0.5543 -> 0.6172 -> 0.6504 -> 0.6443 -> 0.6614 -> 0.6752 -> 0.6761 -> 0.6937 -> 0.7000 -> 0.7131 -> 0.7138 -> 0.7162 -> 0.7312 -> 0.7317 -> 0.7254 -> 0.7350 -> 0.7199 -> 0.7257 -> 0.7355 -> 0.7418

val/compatibility_reward:
0.8278 -> 0.8586 -> 0.8682 -> 0.8659 -> 0.8717 -> 0.8620 -> 0.8727 -> 0.8717 -> 0.8787 -> 0.8813 -> 0.8789 -> 0.8827 -> 0.8734 -> 0.8778 -> 0.8794 -> 0.8814 -> 0.8817 -> 0.8872 -> 0.8847 -> 0.8857

val/accessibility_reward:
0.2809 -> 0.3758 -> 0.4326 -> 0.4226 -> 0.4511 -> 0.4884 -> 0.4794 -> 0.5156 -> 0.5213 -> 0.5448 -> 0.5488 -> 0.5496 -> 0.5891 -> 0.5856 -> 0.5714 -> 0.5886 -> 0.5581 -> 0.5642 -> 0.5864 -> 0.5979

val/pareto_hypervolume:
0.2607 -> 0.3299 -> 0.3779 -> 0.3685 -> 0.3958 -> 0.4242 -> 0.4203 -> 0.4506 -> 0.4591 -> 0.4804 -> 0.4828 -> 0.4871 -> 0.5156 -> 0.5142 -> 0.5025 -> 0.5191 -> 0.4930 -> 0.5010 -> 0.5191 -> 0.5300

val/checkpoint_score:
0.3653 -> 0.4410 -> 0.4870 -> 0.4784 -> 0.5028 -> 0.5293 -> 0.5253 -> 0.5533 -> 0.5601 -> 0.5794 -> 0.5818 -> 0.5843 -> 0.6120 -> 0.6105 -> 0.5998 -> 0.6142 -> 0.5903 -> 0.5970 -> 0.6137 -> 0.6233
```

最终测试指标：

```text
test/reward: 0.7406
test/compatibility_reward: 0.8853
test/accessibility_reward: 0.5960
test/pareto_hypervolume: 0.5281
test/checkpoint_score: 0.6216
```

## 使用建议

常规训练使用：

```bash
python run.py experiment=cityplan/am
```

指定问题规模：

```bash
python run.py experiment=cityplan/am env.generator_params.num_loc=100
```

LUOP200 在 4090 上建议显式覆盖 batch：

```bash
python run.py experiment=cityplan/am \
  env.generator_params.num_loc=200 \
  model.batch_size=256 \
  model.val_batch_size=256 \
  model.test_batch_size=256
```

调试时优先用 runner，而不是手写长命令：

```bash
python scripts/luop_4090_stability.py --suite --log-dir logs/luop_4090_stability
```

如果要测试 noisy evaluation，才设置：

```bash
env.sample_eval_objective_weights=true
```

否则验证和测试保持固定 `[0.5, 0.5]`，这样曲线更可解释。

## 当前结论

当前代码已满足本轮目标：LUOP 使用独立模型入口，支持双动作和 type-first 路径，动作受联合 mask 和约束保护，reward components 与 Pareto 指标完整记录，embedding 能感知目标权重和约束压力。完整本地测试通过，4090 上 LUOP50/100/200 smoke 与 LUOP200 20-epoch full validation 均通过稳定性扫描。LUOP200 保持 `batch_size=256` 是当前经过验证的安全策略；除这种显存边界情况外，常规验证优先使用 4090 大 batch。

## 注意事项

不要提交 SSH 密码、API key、私有数据或机器特定路径。`logs/`、`lightning_logs/`、`wandb/` 和大型 checkpoint 默认视为本地实验产物，除非明确需要归档。新增实验结果如果要作为稳定性证据，必须同时保留命令、Hydra config、metrics.csv、显存峰值和 scanner 输出。
