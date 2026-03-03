没问题！这就为你奉上这份 README 的纯正中文版本。你可以直接把它复制并保存为 `README.md`，提交到你的 GitHub 仓库中。

---

# OAReactDiff-Shimmer: 基于扩散模型的 MECI 生成

本仓库包含一个经过微调的扩散模型（基于 LeftNet/DDPM）实现，专用于从反应物（R）态生成最低能量圆锥交叉点（MECI）的三维几何构型。本项目集成了 **X-MACE** 物理引擎，在生成过程中及生成后提供严格的结构与物理约束（如 S1-S0 能隙）。

## 🚀 近期修改与突破

在近期的更新中，我们解决了原框架中的几个关键 Bug，建立了一条从训练到物理评估的完整且畅通的流水线：

1. **修复图聚合中的 `NaN` / FP32 溢出问题：**
* 确诊原 `LeftNet` 模型底层使用了 `sum`（求和）池化。当处理大分子（例如 > 25 个原子，甚至多达 46 个原子的体系）时，消息传递层会遭遇“聚合爆炸”，导致 FP32 精度下瞬间出现 `NaN` 报错。
* **解决方案：** 构建了稳健的数据清洗脚本，严格过滤掉大于 25 个原子的分子以确保稳定训练。测试结果证明，只要体系尺度合适，该扩散架构就能完美运行。


2. **修复数据划分 Bug (`use_by_ind`)：**
* 修正了 `DataLoader` 读取过时的硬编码索引列表，导致丢弃大部分数据集的严重 Bug。通过将配置设为 `use_by_ind=False`，成功实现了全量清洗数据集的正确加载。


3. **纯 ASE + X-MACE 物理评估流水线：**
* 彻底绕过了原代码中脆弱且高度硬编码的字典传参方式（即 `XMACEPotential` 强行寻找 `'mask'` 和 `'pos'` 键的问题）。
* 编写了强大的独立评估脚本 (`eval_meci_32.py`)，直接使用纯 `ASE (Atoms)` 对象和原生的 `MACECalculator(n_energies=2)` 进行双态能量计算。
* **当前巅峰性能：** 模型目前在测试集上达到了极其出色的几何精度（**RMSD < 0.2 Å**）和物理精度（**S1-S0 能隙 ~ 0.2 - 0.4 eV**）。



---

## 📂 数据集位置

所有经过预处理和对齐的数据集均位于 `reactot/data_meci/` 目录下：

* **`train_rpsb_filtered_25.pkl`**: 稳定的训练数据集，仅包含原子数 <= 25 的分子（**当前稳定训练的推荐数据集**）。
* **`train_rpsb_final_format.pkl`**: 完全对齐的完整数据集（包含大于 25 个原子的巨型分子）。
* **`train_rpsb_all_centered.pkl`**: 坐标已去中心化的数据，防止在计算 RBF 径向基距离时出现空间坐标偏移问题。
* **`full_data_backup.pkl`**: 原始提取数据的纯净备份。

*注：原作者的官方数据集保留在 `reactot/data/train_rpsb_all.pkl` 中。*

---

## ⚠️ 已知问题与未来工作

* **大分子尺度限制（“46原子”屏障）：**
由于底层的图神经网络（`LeftNet`）采用了激进的 `sum` 聚合方式，强行在大于 25~30 个原子的分子上进行训练会导致网络内的特征数值呈指数级膨胀，最终引发 `NaN` 崩溃。
* **建议修复方案：** 若要在包含 46 原子分子的全量数据集上进行训练，必须修改 `LeftNet` 的底层源码（`reactot/model/leftnet.py`），改用 `mean`（平均）池化，或引入归一化聚合（例如将特征除以 $\sqrt{N}$）。需要特别注意的是，这样做会导致现有预训练权重（`reactot-pretrained.ckpt`）完全失效，必须**从零开始重新进行预训练（Train from scratch）**。



---

## 🏃 如何运行

### 1. 训练（微调阶段）

运行以下命令启动基于过滤后数据集的全量微调：

```bash
python -m reactot.trainer.train_rp_finetune

```

*训练产生的权重文件（Checkpoints）将自动保存在 `checkpoint/R2P_Finetune/` 目录下。*

### 2. 模型评估（MECI 采样与 MACE 能隙计算）

运行以下脚本，从测试集中抽取反应物，使用扩散模型生成 MECI 构型，并通过 X-MACE 物理引擎计算其与真实构型的 RMSD 和 S0-S1 能量间隙：

```bash
python eval_meci_32.py

```

---

## 🌳 核心项目结构

```text
.
├── checkpoint/
│   └── R2P_Finetune/           # 自动保存的模型权重 (例如: last-v1.ckpt)
├── reactot/
│   ├── data_meci/              # 预处理好的训练/验证数据集 (.pkl)
│   ├── dynamics/               # MACE 封装器及 ODE/SDE 动力学 (xmace_potential.py)
│   ├── model/                  # 核心扩散模型架构 (leftnet.py, egnn.py)
│   └── trainer/                # PyTorch Lightning 训练主循环 (train_rp_finetune.py)
├── eval_meci_32.py             # 终极物理/几何评估脚本 (计算 RMSD + ASE 能量 Gap)
├── reactot-pretrained.ckpt     # 原作者提供的预训练权重
└── wandb/                      # 训练日志与可视化数据

`
