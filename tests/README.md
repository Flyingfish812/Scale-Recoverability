# 单元测试 — 公式 ↔ 实现映射 (P0-6)

覆盖计划文档 §10 的核心数学性质。运行:

```bash
conda run -n sana python -m pytest tests/unit -q
```

| 测试文件 | 论文公式/性质 | luna 实现 | 说明 |
|---|---|---|---|
| `test_dwt_orthogonality.py` | §10.1 ‖u−û‖² = Σ_b‖W_b(u−û)‖² | `luna/wavelet/transform.decompose_field_2d` / `recompose_field_2d` | Parseval + 重组合; float32 分量容差 1e-4 |
| `test_ger_band_identity.py` | §10.2 GER² = Σ_b ω_b E_direct(b)² | `rel_l2` + `band_error` + `decompose_field_2d` | 由正交性导出; 平滑场 A4 主导能量 |
| `test_sfull.py` | §10.3 S_full 连续可恢复 | `metrics.contiguous_recoverable_index` / `compute_S_full` | 全通过/A4失败/中间失败/重通过/边界/NaN/inf/零分母 |
| `test_scoh.py` | §10.4 P_b²=P_b, γ_b∈[0,1], 不假设 S_coh≥S_full | `metrics.compute_S_coh` | 正交投影幂等; 捕获率; S_coh<S_full 构造 |
| `test_ridge_closed_form.py` | §10.5 闭式解对齐 sklearn, bias 不正则, 确定性 | 正规方程 (与论文 s10/s05 一致) | 标准化空间对比; 大 λ 下 bias 语义 |
| `test_gappy_rank.py` | §10.6 强制 r≤M, validation 选 rank, 可追踪 | 截断+选择逻辑 (论文 S2.3 修正) | 回归: M=20 rank32→20 |
| `test_three_layer.py` | §10.7 分母定义一致, 三角不等式 | `metrics.compute_three_layer_errors` | E_total/trunc/pred 分母核对; ‖u−û‖≤‖u−u_o‖+‖u_o−û‖ |

未覆盖 (实现不在 luna/, 属日期脚本):
- §10.8 模态 NRMSE (`scripts/20260721/s21_per_mode_nrmse.py`) — 见 P0-7 回归
- VCNN 标记 external reference discrepancy — 文档层约定
