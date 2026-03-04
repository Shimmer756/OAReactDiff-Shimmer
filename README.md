# Physics-Informed Diffusion Models for MECI Search

本项目致力于利用基于 X-MACE 势能面的生成式扩散模型，实现光化学反应中最低能量锥形交叉（MECI）点的精确搜索。

## 🧪 核心物理机制

项目采用“训练内化 + 采样引导”的双重物理约束策略：

1. **训练阶段 (Training)**: 引入物理正则化项 $L_{phys} = |E_{S_1} - E_{S_0}|$，赋予模型关于势能面简并性的先验知识。
2. **采样阶段 (Inference)**: 利用增强能量描述符（Augmented Energy, $AE$）产生显式引导力 $\vec{F}_{phys} = -\nabla_{\vec{R}} AE$。
   
   $$AE(\vec{R}) = \alpha \cdot (E_{S_1} - E_{S_0})^2 + \beta \cdot E_{S_1}$$
   
   该公式驱动构型在简并缝隙（Crossing Seam）上向局部极小值收敛。

## 📊 实验基准测试 (Benchmark Results)

我们在包含 63 个分子的验证集上对比了不同物理权重 ($w_{phys}$) 的表现。结果表明，提升物理权重能显著优化简并精度，且保持了极高的几何保真度。

| 权重标签 ($w_{phys}$) | 平均能隙 (Mean Gap) | 能隙中位数 (Median Gap) | 平均几何偏差 (Mean RMSD) |
| :--- | :--- | :--- | :--- |
| **w=0.01** | 0.1891 eV | 0.1162 eV | 0.1697 Å |
| **w=0.10** | **0.1133 eV** | **0.0437 eV** | 0.1746 Å |

**关键结论**：$w=0.1$ 配置使能隙中位数下降了 **62%**，同时 RMSD 波动小于 3%，实现了物理一致性与几何精度的最优平衡。

## 🚀 快速评估

使用自动化评估脚本进行批量采样与物理性质计算：

```bash
python eval_valid_all.py
