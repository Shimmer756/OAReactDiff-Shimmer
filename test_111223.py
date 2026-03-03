import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

def test_mace_thermometer():
    print("🔍 正在测试 X-MACE 是否是“坏掉的温度计”...\n")
    calc = MACECalculator(
        model_paths="/root/X-MACE_2/energies_forces_meci_500.model",
        device='cuda', 
        n_energies=6, 
        default_dtype='float32'
    )
    
    # 读取你刚才修复好的答题卡里的第一个分子
    atoms = read("generated_meci_fixed.xyz", index=0)
    atoms.calc = calc
    
    # 1. 测原版
    e = atoms.get_potential_energy().flatten()
    print(f"🟢 原始生成分子 | S0: {e[0]:.2f} | S1: {e[1]:.2f} | Gap: {abs(e[1]-e[0]):.4f} eV")
    
    # 2. 搞破坏：给每个原子加上巨大的随机位移 (完全偏离 MECI)
    np.random.seed(42)
    atoms.positions += np.random.randn(*atoms.positions.shape) * 1.5
    
    e_noise = atoms.get_potential_energy().flatten()
    print(f"🔴 彻底撕碎之后 | S0: {e_noise[0]:.2f} | S1: {e_noise[1]:.2f} | Gap: {abs(e_noise[1]-e_noise[0]):.4f} eV")
    print("-" * 50)
    
    if abs(abs(e_noise[1]-e_noise[0]) - abs(e[1]-e[0])) < 0.1:
        print("❌ 实锤了！MACE 模型对结构变化完全免疫，它就是一个瞎猜固定值的坏温度计！")
        print("👉 解决方案：我们需要一个在【正常轨迹（包含大Gap和小Gap）】上训练过的 MACE 模型，而不是只在 MECI 上训练的 MACE！")
    else:
        print("✅ MACE 模型正常！Gap 发生了剧烈变化！")

if __name__ == "__main__":
    test_mace_thermometer()
