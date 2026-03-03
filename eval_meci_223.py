import torch
from ase.io import read
from mace.calculators import MACECalculator

def evaluate_generated_meci(xyz_path, mace_model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- 🔍 正在使用 {device} 评估生成的 MECI 真实质量 ---")

    # 初始化原汁原味的 MACE 物理引擎 (复用你之前的成功经验)
    calc = MACECalculator(
        model_paths=mace_model_path,
        device=device,
        n_energies=6,  # 确保与你模型一致
        default_dtype='float32'
    )

    # 读取生成的分子 (支持读取包含多个分子的轨迹 xyz 文件)
    try:
        atoms_list = read(xyz_path, index=':')
        print(f"✅ 成功读取 {len(atoms_list)} 个生成的分子构型。\n")
    except Exception as e:
        print(f"❌ 读取分子失败: {e}")
        return

    gaps = []
    print("-" * 50)
    for i, atoms in enumerate(atoms_list):
        atoms.calc = calc
        # 获取 S0 和 S1 的真实预测能量
        energies = atoms.get_potential_energy().flatten()
        e_s0 = energies[0]
        e_s1 = energies[1]
        
        # 计算纯粹的能量差
        gap = abs(e_s1 - e_s0)
        gaps.append(gap)
        
        print(f"分子 {i+1:03d} | S0: {e_s0:.2f} eV | S1: {e_s1:.2f} eV | 真实 Gap: {gap:.4f} eV")

    print("-" * 50)
    mean_gap = sum(gaps) / len(gaps)
    print(f"🎉 评估完成！这批生成分子的【平均真实能量差】为: {mean_gap:.4f} eV")

if __name__ == "__main__":
    # ==========================================
    # ⚠️ 你只需要修改下面这两个路径 ⚠️
    # 1. 你用 sampling 脚本生成的 xyz 文件路径
    GENERATED_XYZ = "generated_meci_fixed.xyz" 
    
    # 2. 你的 X-MACE 模型路径
    MACE_MODEL = "/root/X-MACE_2/energies_forces_meci_500.model"
    # ==========================================
    
    evaluate_generated_meci(GENERATED_XYZ, MACE_MODEL)
