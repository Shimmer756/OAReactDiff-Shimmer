import numpy as np
from pathlib import Path

# 请替换为您的 NPZ 文件名！
NPZ_FILE_PATH = "custom_R_P_multi_molecule.npz" 

def inspect_npz_file(file_path: str):
    """加载并打印自定义 NPZ 文件的结构和关键数据信息。"""
    try:
        data = np.load(file_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}。请检查路径是否正确。")
        return
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        return

    # --- 1. 打印所有键名 ---
    print("\n==================================================")
    print(f"文件 {Path(file_path).name} 结构概览 (Keys):")
    print("--------------------------------------------------")
    print(list(data.keys()))

    # 尝试确定样本总数 (N_samples)
    try:
        num_samples = data['reactant/num_atoms'].shape[0]
        max_atoms = data['reactant/positions'].shape[1]
        print(f"\n样本总数 (N_samples): {num_samples} 个反应")
        print(f"最大原子数 (MAX_ATOMS/Padding 长度): {max_atoms}")
        print("==================================================")
    except KeyError:
        print("\n错误: 缺少核心键 (reactant/num_atoms)。")
        return
    
    # --- 2. 检查 Reactant (out[0]) ---
    print("\n--- A. Reactant (out[0]) 数据结构 ---")
    print(f"Reactant 坐标形状 (positions): {data['reactant/positions'].shape}")
    print(f"Reactant 原子数形状 (num_atoms): {data['reactant/num_atoms'].shape}")
    
    # 打印第一个样本的真实原子数和前 5 个原子位置
    print(f"\n[样本 0 详情 - Reactant]")
    print(f"原子数: {data['reactant/num_atoms'][0]}")
    print(f"前 5 个原子位置 (Positions 示例):\n{data['reactant/positions'][0, :-5:-1]}")
    print(f"前 5 个原子序数 (Charges 示例):\n{data['reactant/charges'][0, :5]}")


    # --- 3. 检查 Transition State (out[1] - 占位符) ---
    print("\n--- B. Transition State (out[1]) 占位符结构 ---")
    # 验证 TS 占位符是否与 Reactant 相同 (即 R 的拷贝)
    if np.array_equal(data['reactant/positions'], data['transition_state/positions']):
         print("✅ TS Positions 与 Reactant Positions 完全一致 (作为占位符)。")
    else:
         print("❌ TS Positions 与 Reactant Positions 不一致。")
    print(f"TS 坐标形状 (positions): {data['transition_state/positions'].shape}")
    #print(f"前 5 个原子位置 (Positions 示例):\n{data['transition_state/positions'][0, :]}")
    #print(f"前 5 个原子序数 (Charges 示例):\n{data['transition_state/charges'][0, ]}")


    # --- 4. 检查 Product (out[2]) ---
    print("\n--- C. Product (out[2]) 数据结构 ---")
    print(f"Product 坐标形状 (positions): {data['product/positions'].shape}")
    
    # 打印第一个样本的前 5 个原子位置
    print(f"\n[样本 0 详情 - Product]")
    print(f"原子数: {data['product/num_atoms'][0]}")
    print(f"前 5 个原子位置 (Positions 示例):\n{data['product/positions'][0, :5]}")


if __name__ == "__main__":
    inspect_npz_file(NPZ_FILE_PATH)
