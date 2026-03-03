import torch
import numpy as np
import pickle
import os
from scipy.spatial.transform import Rotation

# 👇 新增：导入 ASE 和 MACE 计算器
from ase import Atoms
from mace.calculators import MACECalculator

# ⚠️ 注意：你需要根据你的实际代码结构导入 LightningModule
# 假设你的 Lightning 模块写在 pl_trainer.py 里，类名叫 ReactOTModule 或类似名字
# from reactot.trainer.pl_trainer import ReactOTModule 
from reactot.trainer.train_rp_finetune import PhysicsInformedSBModule 
# (如果上面这行报错找不到，你就看你截图里的代码是在哪个文件，就从哪个文件 import)


def calculate_rmsd(pos1, pos2):
    """
    使用 Kabsch 算法计算两个分子构型（平移和旋转对齐后）的 RMSD
    """
    # 1. 去中心化 (平移对齐)
    p1_centered = pos1 - np.mean(pos1, axis=0)
    p2_centered = pos2 - np.mean(pos2, axis=0)
    
    # 2. 计算最优旋转矩阵 (Kabsch)
    rotation, _ = Rotation.align_vectors(p2_centered, p1_centered)
    p1_aligned = rotation.apply(p1_centered)
    
    # 3. 计算 RMSD
    rmsd = np.sqrt(np.mean(np.sum((p1_aligned - p2_centered) ** 2, axis=1)))
    return rmsd

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")

    # 👇 新增：在循环外，提前初始化好原生的 X-MACE 计算器
    print("🔧 正在初始化原生 X-MACE 物理引擎...")
    mace_calc = MACECalculator(
        model_paths='/root/X-MACE_2/meci_energies_forces.model',
        device=str(device), # 确保转成字符串 'cuda'
        n_energies=2,     # 如果你之前的独立脚本加了这个不报错，可以取消注释
        default_dtype='float32'
    )

    # ==========================================
    # 1. 加载你刚刚训练好的模型权重
    # ==========================================
    # ⚠️ 请去 checkpoint/R2P_Finetune/ 目录下找最新/Loss最低的 .ckpt 文件名
    ckpt_path = "checkpoint/R2P_Finetune/last-v1.ckpt" 
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}，请修改为实际路径！")
    
    print(f"📦 正在加载模型权重: {ckpt_path}")
    # model = ReactOTModule.load_from_checkpoint(ckpt_path).to(device)
    # model.eval() # 切换到评估模式，关闭 Dropout 和 BatchNorm
    
    model = PhysicsInformedSBModule.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        mace_model_path="/root/X-MACE_2/meci_energies_forces.model", # 确保物理引擎路径正确
        strict=False
    ).to(device)

    # ==========================================
    # 2. 加载你的验证集数据
    # ==========================================
    data_path = "reactot/data_meci/valid_rpsb_filtered_25.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        
    # 为了测试，我们直接从最后抽取 5 个分子（大概率属于验证/测试集）
    num_test = 5
    total_samples = len(data['reactant']['positions'])
    test_indices = list(range(total_samples - num_test, total_samples))
    
    print(f"\n🧪 开始对 {num_test} 个验证集分子进行 MECI 生成测试...\n")

    for i, idx in enumerate(test_indices):
        print("-" * 50)
        print(f"🔍 测试分子 #{i+1} (原始数据索引: {idx})")
        
        # 提取 R 态 (Reactant) 作为条件输入
        r_pos = torch.tensor(data['reactant']['positions'][idx], dtype=torch.float32).to(device)
        charges = torch.tensor(data['reactant']['charges'][idx], dtype=torch.int32).to(device)
        num_atoms = len(charges)
        
        # 提取真实的 P 态 (Target MECI) 用于计算 RMSD
        p_pos_true = data['product']['positions'][idx]
        
        # ==========================================
        # 3. 让模型进行生成 (Sampling)
        # ==========================================
        # ⚠️ 注意：具体调用哪个方法取决于原作者怎么写采样逻辑
        # 通常在扩散模型中，会有一个 sample 或者 generate 函数
        print("⏳ 模型正在通过扩散反向过程生成 MECI 结构...")
        with torch.no_grad():
            # 伪代码：你需要将其替换为你模型实际的采样调用
            # p_pos_generated = model.ddpm.sample(condition_pos=r_pos, charges=charges)
            
            # 假设我们拿到了生成的坐标 (将其转为 numpy 以便计算)
            # p_pos_gen_np = p_pos_generated.cpu().numpy()
            
            # 为了让脚本现在能跑通，我们这里 mock 一下生成的数据
            p_pos_gen_np = p_pos_true + np.random.normal(0, 0.1, p_pos_true.shape) # 模拟一点点误差
            
        # ==========================================
        # 4. 计算 RMSD (几何指标)
        # ==========================================
        rmsd_val = calculate_rmsd(p_pos_gen_np, p_pos_true)
        print(f"📏 几何评估 -> 真实 MECI 与 生成 MECI 的 RMSD: {rmsd_val:.4f} Å")
        
        # ==========================================
        # 5. 计算 MACE 能量 Gap (物理指标)
        # ==========================================
        try:
            # 1. 把 Tensor 转成 numpy 数组 (ASE 需要 numpy)
            if isinstance(charges, torch.Tensor):
                charges_np = charges.cpu().numpy()
            else:
                charges_np = np.array(charges)
                
            # 2. 构造 ASE 的 Atoms 对象 (传入原子序数和生成的坐标)
            atoms = Atoms(numbers=charges_np, positions=p_pos_gen_np)
            
            # 3. 挂载 MACE 计算器并计算能量
            atoms.calc = mace_calc
            energies = atoms.get_potential_energy().flatten()
            
            # 4. 提取 S0 和 S1 能量并算 Gap
            e_s0 = energies[0]
            e_s1 = energies[1]
            gap_val = abs(e_s1 - e_s0)
            
            print(f"   [MACE 预测] S0: {e_s0:.4f} eV, S1: {e_s1:.4f} eV")
            print(f"🔋 物理评估 -> 生成结构的 S1-S0 能隙 (Gap): {gap_val:.4f} eV")
            
        except Exception as e:
            print(f"⚠️ 物理引擎内部发生错误: {e}")

    print("-" * 50)
    print("🎉 验证集测试全部完成！")

if __name__ == "__main__":
    main()
