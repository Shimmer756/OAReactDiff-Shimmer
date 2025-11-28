import torch
import numpy as np
from pathlib import Path
from typing import List
from torch.utils.data import DataLoader

from pymatgen.core import Molecule
from oa_reactdiff.analyze.rmsd import pymatgen_rmsd

# 导入项目模块
from oa_reactdiff.trainer.pl_trainer import DDPMModule
from oa_reactdiff.dataset.transition1x import ProcessedTS1x
from oa_reactdiff.diffusion._normalizer import FEATURE_MAPPING


# [🔥 核心修复] 添加这行导入
from oa_reactdiff.diffusion._schedule import PredefinedNoiseSchedule, DiffSchedule


# --- 1. 配置 ---
MODEL_CHECKPOINT_PATH = "checkpoint/OAReactDiff/leftnet-0-20f22da4eb62/ddpm-epoch=1978-val-totloss=300.81.ckpt"
# 注意：这里使用的是原始的 pickle 数据集
CUSTOM_DATA_PATH = Path("oa_reactdiff/data/transition1x/valid_addprop.pkl") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 必要的模型配置
leftnet_config = dict(
    pos_require_grad=False, cutoff=10.0, num_layers=6, hidden_channels=196,
    num_radial=96, 
    # 注意：如果您用的是官方预训练模型(5种原子)，这里应该是 8
    # 如果是您自己训练的(11种原子)，这里是 14。请根据实际 checkpoint 调整。
    in_hidden_channels=14, 
    reflect_equiv=True, legacy=True,
    update=True, pos_grad=False, single_layer_output=True, object_aware=True,
)

# 推理配置
inference_config = dict(
    datadir=str(CUSTOM_DATA_PATH), remove_h=False,
    bz=1,                 
    num_workers=0, 
    swapping_react_prod=False, 
    single_frag_only=False,
    clip_grad=False, 
    gradient_clip_val=None, 
    ema=False, 
    ema_decay=0.999,
    # [🔥 核心修复] 必须与训练脚本 (train_ts1x.py) 中的设置完全一致！
    noise_schedule="cosine",
    timesteps=5000,
)

# --- 2. 核心生成函数 ---
def generate_single_molecule(ddpm_module, data_loader):
    model = ddpm_module.ddpm.to(device)
    
    # ==========================================
    # 🔥 核心修复：替换为验证时的采样时间表
    # ==========================================
    print("正在切换到推理时间表 (Polynomial_2, 150 steps)...")
    
    # 1. 创建新的 Gamma 模块 (Poly2, 150步)
    sampling_gamma = PredefinedNoiseSchedule(
        noise_schedule="polynomial_2",
        timesteps=150,
        precision=1e-5,
    ).to(device)
    
    # 2. 创建新的 Schedule 并覆盖模型
    sampling_schedule = DiffSchedule(
        gamma_module=sampling_gamma,
        norm_values=model.normalizer.norm_values
    )
    
    # 3. 强制替换模型的 schedule 和 T
    model.schedule = sampling_schedule
    model.T = 150
    # ==========================================

    model.eval()

    print(f"🚀 开始生成单个分子...")

    for batch_idx, batch in enumerate(data_loader):
        representations, conditions = batch

        # 移动数据到 GPU
        representations = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in r.items()} for r in representations]
        conditions = conditions.to(device)

        # ==========================================
        # 🔥 新增：打印真实数据 (Ground Truth)
        # ==========================================
        print("\n====== 🟢 原始数据集真实坐标 (Ground Truth) ======")
        
        # R (Reactant) - 索引 0
        gt_R_pos = representations[0]['pos'][:5].cpu().numpy()
        print(f"\n[真实 Reactant] 前 5 行坐标:")
        print(gt_R_pos)

        # TS (Transition State) - 索引 1
        gt_TS_pos = representations[1]['pos'][:5].cpu().numpy()
        print(f"\n[真实 Transition State] 前 5 行坐标:")
        print(gt_TS_pos)

        # P (Product) - 索引 2
        gt_P_pos = representations[2]['pos'][:5].cpu().numpy()
        print(f"\n[真实 Product] 前 5 行坐标:")
        print(gt_P_pos)
        
        print("\n==================================================\n")

        # 准备固定输入
        xh_fixed = [
            torch.cat([repre[feature_type] for feature_type in FEATURE_MAPPING], dim=1)
            for repre in representations
        ]

        n_samples = representations[0]["size"].size(0)
        fragments_nodes = [repre["size"] for repre in representations]

        # 核心：固定 Reactant(0) 和 Product(2)，生成 TS(1)
        frag_fixed = [0, 2]

        with torch.no_grad():
            out_samples, _ = model.inpaint(
                n_samples=n_samples,
                fragments_nodes=fragments_nodes,
                conditions=conditions,
                return_frames=1,
                resamplings=3,
                jump_length=1,
                xh_fixed=xh_fixed,
                frag_fixed=frag_fixed,
            )

        # 提取生成的 TS (Index 1)
        generated_fragments = out_samples[0]
        gen_TS_tensor = generated_fragments[1]

        gen_TS_pos = gen_TS_tensor[:, :3].cpu().numpy()

        print("====== 🔵 模型生成结果 (Generated) ======")
        print(f"\n[生成 Transition State] 前 5 行坐标:")
        print(gen_TS_pos[:5])
        
        # 计算一个简单的误差看看 (仅针对前5个原子)
        # 注意：这里没有进行 Kabsch 对齐，直接减可能误差很大，但能看个大概
        diff = np.abs(gen_TS_pos[:5] - gt_TS_pos[:5])
        print(f"\n[坐标差异 (Abs Diff)] 前 5 行:")
        print(diff)
        

        # =========================================================
        # 👇👇👇 在这里插入您的新代码 (替换掉原来的简单 diff 计算) 👇👇👇
        # =========================================================
        
        # 定义辅助函数 (也可以放在文件外面)
        def tensor_to_mol(pos, charges):
            return Molecule(
                species=charges.flatten().cpu().long().numpy(),
                coords=pos.cpu().numpy()
            )

        # 1. 构造分子对象
        # gen_TS_tensor 的最后一列 (-1) 是电荷，前3列 (:3) 是坐标
        mol_gen = tensor_to_mol(
            gen_TS_tensor[:, :3], 
            gen_TS_tensor[:, -1]
        )

        # 真实数据在 representations[1] 中
        mol_true = tensor_to_mol(
            representations[1]['pos'], 
            representations[1]['charge']
        )

        # 2. 计算对齐后的 RMSD
        try:
            print("\n====== 📐 正在进行 Kabsch 对齐与计算... ======")
            # pymatgen_rmsd 会自动处理旋转和平移对齐
            real_rmsd = pymatgen_rmsd(mol_gen, mol_true, ignore_chirality=True)
            
            print(f"✅ 对齐后的真实 RMSD: {real_rmsd:.4f} Å")
            
            if real_rmsd < 0.5:
                print(">> 结果判定: 🌟 非常准确 (Excellent)")
            elif real_rmsd < 1.0:
                print(">> 结果判定: 👍 结构合理 (Good)")
            else:
                print(">> 结果判定: ⚠️ 偏差较大 (Poor)")
                
        except Exception as e:
            print(f"❌ RMSD 计算出错: {e}")
            
        # =========================================================
        # 👆👆👆 插入结束 👆👆👆
        # =========================================================


        # 退出循环
        break

# --- 3. 执行 ---
if __name__ == "__main__":
    print("正在加载模型...")
    # 如果加载官方模型报错维度不对，请将上面的 in_hidden_channels 改回 8
    ddpm_module = DDPMModule.load_from_checkpoint(
        checkpoint_path=MODEL_CHECKPOINT_PATH,
        map_location=device,
        model_config=leftnet_config,
        training_config=inference_config,

        # [🔥 核心修复] 显式覆盖 __init__ 的默认值
        noise_schedule=inference_config["noise_schedule"],
        timesteps=inference_config["timesteps"],
    )

    print("正在加载数据...")
    # ProcessedTS1x 会自动处理 pkl 文件
    dataset = ProcessedTS1x(npz_path=str(CUSTOM_DATA_PATH), **inference_config)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

    generate_single_molecule(ddpm_module, loader)
