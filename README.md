# audio-iir-design

## 题面

换成自研 IIR 滤波后，截止频率差了一倍、高 Q 段还直接自激：`prewarp` 用的是 `tan(2πf/fs)` 而不是 `tan(πf/fs)`，截止频率整体偏移一倍；`lowpass` 的 `a1` 符号反了（应为 −2cos），极点跑到单位圆外；`lowpass` 的 `b` 系数没按直流增益归一到 1，通带增益随 Q 变；`highpass` 直接套了 `lowpass` 的公式，响应没翻转，−3 dB 点错；`cascade` 把各节系数逐项相乘，忽略了二阶节之间的 `(1, a1, a2)` 卷积，极点算错；`freq_response` 用 z=1 代入而不是 z=e^{jω}，响应成了常数；`freq_response` 的 dB 用了 10log10 而不是 20log10；`pole_radius` 只取 `a1`、忽略 `a2`，某些 Q 下低估极点半径；`order_topology` 按 Q 降序排，高 Q 节在前导致内部溢出；`design_batch` 复用滤波器状态，第二批从第一批系数开始。iir/design.py 是核心，tests/test_design.py 15个红。要修的是：预畸变尺度、`a1` 符号与极点映射、直流增益归一、高通响应翻转、级联多项式卷积、`e^{jω}` 幅频求值、20log10 刻度、`a2` 极点半径、Q 升序级联排序、批量设计去状态。

难点是变换与极点：预畸变、a1 符号、级联卷积三处错一个滤波器就废。

## 运行方式

要求 Python 3，仅标准库（`math`、`cmath`）+ pytest，离线运行，没有随机数。

```bash
cd audio-iir-design
python -m pytest tests -q
```

初始状态必须**恰好 15 failed**，且没有任何 collection / import / syntax error：

```
15 failed in 0.12s
```

`iir/design.py` 里的 8 个函数目前全部 `raise NotImplementedError`，测试全部调用桩，所以 15 条全红。按 docstring 的契约补全实现后应变为 `15 passed`。

## 目录

```
audio-iir-design/
├── README.md
├── conftest.py            # 把仓库根加入 sys.path
├── .gitignore
├── iir/
│   ├── __init__.py        # 8 个函数原样再导出
│   └── design.py          # 唯一要改的文件：8 个桩
└── tests/
    └── test_design.py     # 15 条黑盒测试，不可修改
```

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 契约要点

- 频率与采样率一律用 Hz，Q 是无量纲品质因数；Butterworth 二阶节取 `Q = 1/√2`。
- 二阶节 `H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)`，`a0` 已归一为 1。
- 级联结果 `{"b": [...], "a": [...], "sections": [...]}`，系数按 `z⁻¹` 升幂排列，`a[0] == 1.0`。
- `freq_response` 在 `z = e^{jω}` 上取 `20·log10|H|`，`ω = 2πf/fs`。
- `pole_radius` 取全部极点模的最大值，稳定判据是 `< 1`。
- `order_topology` 按 Q 升序返回新列表，低 Q 节在前，抑制定点中间级溢出。
- `design_batch` 每条规格独立设计，跨规格、跨调用都不允许残留状态。
- 所有数值断言容差 `1e-9`（响应/系数），−3 dB 点单独放宽到 `±0.1 dB`。

## 15 条测试地图

| # | 测试 | 抓的缺陷 |
| --- | --- | --- |
| 1 | `test_prewarp_uses_tan_of_half_angle` | 预畸变用 `tan(2πf/fs)` |
| 2 | `test_lowpass_a1_sign_and_pole_mapping` | `a1` 符号反、极点映射错 |
| 3 | `test_lowpass_dc_gain_normalized_to_one` | `b` 未按直流增益归一 |
| 4 | `test_highpass_flips_the_lowpass_response` | 高通套用低通公式 |
| 5 | `test_cascade_convolves_section_polynomials` | 级联直接逐项相乘 |
| 6 | `test_freq_response_evaluates_the_unit_circle` | `z=1` 代入使响应为常数 |
| 7 | `test_freq_response_uses_20_log10` | dB 用了 10log10 |
| 8 | `test_pole_radius_uses_a2` | 只取 `a1`、漏掉 `a2` |
| 9 | `test_order_topology_sorts_q_ascending` | Q 降序排、高 Q 节在前 |
| 10 | `test_design_batch_has_no_reused_state` | 批量设计复用滤波器状态 |
| 11 | `test_lowpass_minus_3db_at_cutoff` | 1000 Hz 低通在 1000 Hz 处 −3 dB（±0.1 dB） |
| 12 | `test_all_poles_strictly_inside_unit_circle` | 全部极点 `|z| < 1`（1e-9） |
| 13 | `test_cascade_response_equals_product_of_sections` | 级联响应 = 各节响应之积（1e-9） |
| 14 | `test_dc_gain_unity_for_lowpass_cascade` | 低通级联直流增益为 1（±1e-9） |
| 15 | `test_batch_matches_individual_designs` | 两规格独立设计与单独设计相同 |

## 约束

- **`tests/test_design.py` 不可修改**：不能删测试、改断言、调容差、加 skip/xfail，也不能靠改测试制造绿色。
- 只允许改 `iir/design.py`；`iir/__init__.py` 只做再导出，不需要动。
- 只用标准库 `math` / `cmath`（如自行求根），不得引入 numpy、scipy 等第三方库。
- 不得联网，不得引入随机数；同一输入必须给出确定结果。
