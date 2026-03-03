import torch
import numpy as np
import os
from scipy.spatial.transform import Rotation
import random

# ==========================================
# 0. 绝对的确定性环境锁定
# ==========================================
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

seed_all(42)

from ase.io import read
from ase import Atoms
from mace.calculators import MACECalculator
from reactot.trainer.train_rp_finetune import PhysicsInformedSBModule


def save_comparison_xyz(filename, charges, pos_r, pos_p):
    """
    将反应物和生成物保存为多帧 XYZ 文件，方便可视化对比。
    """
    # 常用原子序数映射
    symbol_map = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl'}
    
    with open(filename, 'w') as f:
        # 帧 0：反应物 (Reactant)
        f.write(f"{len(charges)}\n")
        f.write("Frame 0: Reactant (Initial Stage)\n")
        for z, pos in zip(charges, pos_r):
            symbol = symbol_map.get(int(z), 'X')
            f.write(f"{symbol:2s} {pos[0]:12.8f} {pos[1]:12.8f} {pos[2]:12.8f}\n")
            
        # 帧 1：生成的 MECI 候选构型
        f.write(f"{len(charges)}\n")
        f.write("Frame 1: Generated MECI (Model Prediction)\n")
        for z, pos in zip(charges, pos_p):
            symbol = symbol_map.get(int(z), 'X')
            f.write(f"{symbol:2s} {pos[0]:12.8f} {pos[1]:12.8f} {pos[2]:12.8f}\n")
    
    print(f"\n📂 轨迹对比文件已保存: {filename}")

class OPT:
    def __init__(self):
        self.solver = "ddpm"
        self.method = "midpoint"
        self.atol = 1e-2
        self.rtol = 1e-2

def calculate_rmsd(pos1, pos2):
    p1_centered = pos1 - np.mean(pos1, axis=0)
    p2_centered = pos2 - np.mean(pos2, axis=0)
    rotation, _ = Rotation.align_vectors(p2_centered, p1_centered)
    p1_aligned = rotation.apply(p1_centered)
    rmsd = np.sqrt(np.mean(np.sum((p1_aligned - p2_centered) ** 2, axis=1)))
    return rmsd

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xyz_path = "molecule.xyz"
    ckpt_path = "checkpoint/R2P_Finetune/last-v1.ckpt"
    mace_model_path = "/root/X-MACE_2/meci_energies_forces.model"

    print(f"🚀 使用设备: {device}")

    # 1. 初始化 MACE 物理引擎 (FP64 精度)
    mace_calc = MACECalculator(model_paths=mace_model_path, device=str(device), n_energies=2, default_dtype='float64')

    # 2. 加载微调模型并提升至双精度 (消灭 NaN)
    model = PhysicsInformedSBModule.load_from_checkpoint(
        checkpoint_path=ckpt_path, mace_model_path=mace_model_path, strict=False
    ).to(device).double()
    model.eval()
    model.ddpm.opt = OPT() 

    # 3. 读取数据
    frames = read(xyz_path, index=':')
    reactant_atoms = frames[0]
    target_atoms = frames[1]
    #print(reactant_atoms)
    #print(target_atoms)
    
    r_pos_np = reactant_atoms.get_positions()
    p_pos_true = target_atoms.get_positions()
    #print(r_pos_np)
    #print(p_pos_true)
    
    charges_np = reactant_atoms.get_atomic_numbers()
    print(charges_np)
    
    N = len(charges_np)
    print(N)
    
    # ==========================================
    # 🌟 新增：计算反应物 (Initial Reactant) 的基准能隙
    # ==========================================
    print("\n📊 [基准评估] 正在计算原始反应物的能隙...")
    try:
        # 创建反应物 Atoms 对象
        r_atoms_eval = Atoms(numbers=charges_np, positions=r_pos_np)
        r_atoms_eval.calc = mace_calc
        
        # 计算能量
        r_energies = r_atoms_eval.get_potential_energy().flatten()
        r_gap = abs(r_energies[1] - r_energies[0])
        
        print("-" * 30)
        print(f"🚩 反应物 S0 能量: {r_energies[0]:.6f} eV")
        print(f"🚩 反应物 S1 能量: {r_energies[1]:.6f} eV")
        print(f"🚩 反应物 初始 Gap: {r_gap:.6f} eV")
        print("-" * 30)
    except Exception as e:
        print(f"❌ 反应物基准评估失败: {e}")
        r_gap = None

    # 4. 构造符合 EGNNDynamics 严苛要求的特征字典
    atom_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
    z_indices = torch.tensor([atom_map.get(int(c), 0) for c in charges_np], device=device)
    z_onehot = torch.nn.functional.one_hot(z_indices, num_classes=5).double()
    
    # 🌟 核心：必须是 float64 且形状为 [N, 1] 用于物理引擎
    z_charge = torch.tensor(charges_np, dtype=torch.float64, device=device).unsqueeze(-1) 
    
    # 🌟 核心：必须是 Long 类型用于动态广播
    batch_idx = torch.zeros(N, dtype=torch.long, device=device)
    #batch_idx = torch.ones(N, dtype=torch.long, device=device)
    print(batch_idx)
    repre_template = {
        "one_hot": z_onehot,
        "charge": z_charge,
        "size": torch.tensor([N], dtype=torch.long, device=device),
        "mask": batch_idx, # EGNNDynamics 内部的 combined_mask 来源
        "batch": batch_idx
    }

    repre_R = repre_template.copy()
    repre_R["pos"] = torch.tensor(r_pos_np, dtype=torch.float64, device=device)
    repre_P = repre_template.copy()
    repre_P["pos"] = torch.tensor(p_pos_true, dtype=torch.float64, device=device)

    representations = [repre_R, repre_P]
    # 🌟 核心：必须是 [Batch_size, condition_nf] 形状的 Tensor
    conditions = torch.zeros(1, 1, dtype=torch.float64, device=device)
    print("\n⏳ 正在通过确定性轨迹(ODE)推演全分子 MECI 构型...")
    with torch.no_grad():
        x1_full = torch.tensor(r_pos_np, dtype=torch.float64, device=device).unsqueeze(0).reshape(1, -1)
        conditions = torch.zeros(1, 1, dtype=torch.float64, device=device)

        sample_out = model.ddpm.sample(
            x1_full,
            representations,
            conditions,
            ot_ode=True,
            nfe=500,
            log_count=50
        )
        # 将轨迹数据先拉到 CPU 内存中，脱离计算图
        trajectories = sample_out[1].detach().cpu()

    # 2. 评估过程：必须开启梯度（MACE 引擎计算能量需要）
    print(f"\n🧪 采样完成，共得到 {trajectories.shape[1]} 帧轨迹。开始寻找物理最优解...")
    
    best_gap = float('inf')
    best_pos = None
    best_frame_idx = -1
    best_energies = None

    # 🌟 关键：不要在 with torch.no_grad() 里面跑下面这个循环
    for f in range(trajectories.shape[1]):
        current_pos_np = trajectories[:, f, :].numpy()
        
        try:
            temp_atoms = Atoms(numbers=charges_np, positions=current_pos_np)
            temp_atoms.calc = mace_calc
            # 🌟 核心：对于 MACE，直接调用即可，它内部会自己管理梯度环境
            energies = temp_atoms.get_potential_energy().flatten()
            current_gap = abs(energies[1] - energies[0])
            
            if current_gap < best_gap:
                best_gap = current_gap
                best_pos = current_pos_np
                best_frame_idx = f
                best_energies = energies
            
            if f % 10 == 0 or f == trajectories.shape[1] - 1:
                print(f"  [Frame {f:2d}] Current Gap: {current_gap:.6f} eV")
        except Exception as e:
            print(f"  [Frame {f:2d}] 评估跳过: {e}")
        # 3. 🌟 将“最强一帧”作为最终结果
        p_pos_gen_np = best_pos
        print(f"\n🏆 轨迹搜索完毕！最优帧为第 {best_frame_idx} 帧")

    # ==========================================
    # 🌟 最终评估（针对最优帧）
    # ==========================================
    print("-" * 50)
    print(f"📏 几何评估 (Best Frame) -> RMSD: {calculate_rmsd(p_pos_gen_np, p_pos_true):.4f} Å")
    print(f"🧪 [MACE 最优结果]")
    print(f"基态 S0 能量: {best_energies[0]:.6f} eV")
    print(f"激发态 S1 能量: {best_energies[1]:.6f} eV")
    print(f"🔋 物理评估 -> 最小能隙: {best_gap:.6f} eV")

    print(f"✅ 全分子采样完成！当前原子数: {p_pos_gen_np.shape[0]}")
    # 验证最终原子数
    generated_N = p_pos_gen_np.shape[0]
    print(f"✅ 采样完成！模型输出总原子数: {generated_N}")
    
    if generated_N != N:
        print(f"⚠️ 警告: 模型生成的原子数({generated_N})与输入({N})不符！")
    # 6. 几何与物理评估
    # 🌟 修复：定义对比的目标值（即数据集里的真实 MECI 坐标）
    p_pos_target_np = p_pos_true
    rmsd_val = calculate_rmsd(p_pos_gen_np, p_pos_target_np)
    
    # 保存可视化对比文件
    save_comparison_xyz("comparison_result.xyz", charges_np, r_pos_np, p_pos_gen_np)

    print("-" * 50)
    print(f"📏 几何评估 -> RMSD: {rmsd_val:.4f} Å")

    # ==========================================
    # 🌟 回归最简单的“你的”逻辑
    # ==========================================
    print("\n111111")
    try:
        # 1. 创建 Atoms 并绑定计算器
        gen_atoms = Atoms(numbers=charges_np, positions=p_pos_gen_np)
        gen_atoms.calc = mace_calc
        
        # 2. 直接用你之前的写法：flatten 拿到所有能级
        energies = gen_atoms.get_potential_energy().flatten()

        print("-" * 50)
        print("🧪 [MACE 预测结果]")
        print(f"基态 S0 能量: {energies[0]:.6f} eV")
        print(f"激发态 S1 能量: {energies[1]:.6f} eV")
        
        gap_val = abs(energies[1] - energies[0])
        print(f"🔋 物理评估 -> S1-S0 Gap: {gap_val:.6f} eV")

        if gap_val < 0.1:
            print("✨ 结论: 生成结构非常接近圆锥交叉点 (MECI)！")
        else:
            print(f"  结论: 生成结构能隙为 {gap_val:.4f} eV。")

    except Exception as e:
        print(f"❌ 物理评估失败: {e}")

if __name__ == "__main__":
    main()
