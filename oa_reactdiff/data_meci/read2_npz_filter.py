import numpy as np
import os

# --- 配置 ---
INPUT_FILE = "custom_R_P_multi_molecule.npz"
OUTPUT_FILE = "filtered_dataset.npz"

# 定义目标剔除元素 (Atomic Number Z)
# 格式: {Z: "名称"}
BANNED_ELEMENTS = {
    15: "磷 (P)",
    16: "硫 (S)",
    17: "氯 (Cl)",
    35: "溴 (Br)",
    53: "碘 (I)",
    5:  "硼 (B)"
}

# 辅助函数：将原子序数转为元素符号（用于最后一步统计剩余元素）
def get_element_symbol(z):
    mapping = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
    return mapping.get(z, f"Z={z}")

def analyze_and_filter():
    print(f"正在加载文件: {INPUT_FILE} ...")
    try:
        data = np.load(INPUT_FILE, allow_pickle=True)
    except FileNotFoundError:
        print("错误：找不到输入文件。")
        return

    # 获取核心数据
    # 我们假设原子种类信息存在于 'reactant/charges' 中
    # 且所有关键数据的第一个维度都是样本数 (N_samples)
    try:
        charges = data['reactant/charges']     # 形状: (N, Max_Atoms)
        num_atoms = data['reactant/num_atoms'] # 形状: (N,)
        total_samples = charges.shape[0]
    except KeyError as e:
        print(f"错误：数据中缺少键 {e}，无法分析原子类型。")
        return

    print(f"原始样本总数: {total_samples}")

    # ---------------------------------------------------------
    # 任务 1: 统计包含目标元素的分子
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("任务 1: 统计目标元素")
    print("="*40)

    # 初始化计数器
    element_counts = {z: 0 for z in BANNED_ELEMENTS}
    indices_to_remove = set() # 使用集合存储要剔除的分子索引（自动去重）

    # 遍历所有分子
    for i in range(total_samples):
        # 获取当前分子的有效原子 (去除 padding)
        n = num_atoms[i]
        # 注意：这里取整并转为列表，防止浮点数干扰
        atom_types = set(charges[i, :n].astype(int))

        has_banned_in_this_mol = False

        # 检查是否包含被禁元素
        for banned_z in BANNED_ELEMENTS:
            if banned_z in atom_types:
                element_counts[banned_z] += 1
                has_banned_in_this_mol = True

        # 如果包含任意一种或多种被禁元素，标记该索引待删除
        if has_banned_in_this_mol:
            indices_to_remove.add(i)

    # 输出统计结果
    print("各元素分子数量 (独立统计):")
    for z, name in BANNED_ELEMENTS.items():
        print(f"  - 包含 {name:<6}: {element_counts[z]} 个")

    print("-" * 30)
    print(f"包含以上任意元素的分子总数 (去重后): {len(indices_to_remove)} 个")
    print(f"占比: {len(indices_to_remove)/total_samples*100:.2f}%")


    # ---------------------------------------------------------
    # 任务 2: 剔除并保存新文件
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("任务 2: 生成新数据集")
    print("="*40)

    # 计算需要保留的索引
    all_indices = set(range(total_samples))
    keep_indices = sorted(list(all_indices - indices_to_remove))

    if len(keep_indices) == 0:
        print("警告：所有分子都被剔除了！将不会保存文件。")
        return

    print(f"剔除 {len(indices_to_remove)} 个分子，保留 {len(keep_indices)} 个分子。")

    # 构建新数据字典
    filtered_data = {}

    # 遍历原始 npz 中的所有数组
    for key in data.files:
        original_arr = data[key]

        # 只有当数组的第一维等于原始样本数时，我们才进行切片过滤
        # 这可以防止错误切割元数据或其他非样本对应的数据
        if hasattr(original_arr, 'shape') and original_arr.shape[0] == total_samples:
            filtered_data[key] = original_arr[keep_indices]
        else:
            # 如果不是样本数据（比如全局配置），直接保留
            filtered_data[key] = original_arr

    # 保存压缩的 npz
    np.savez_compressed(OUTPUT_FILE, **filtered_data)
    print(f"✅ 新文件已保存至: {OUTPUT_FILE}")


    # ---------------------------------------------------------
    # 任务 3: 验证新文件
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("任务 3: 验证新文件内容")
    print("="*40)

    check_new_file(OUTPUT_FILE)

def check_new_file(filepath):
    """单独的验证函数"""
    try:
        data = np.load(filepath, allow_pickle=True)
        charges = data['reactant/charges']
        num_atoms = data['reactant/num_atoms']
        total_samples = charges.shape[0]
    except Exception as e:
        print(f"验证失败: {e}")
        return

    print(f"新文件样本数: {total_samples}")

    # 1. 检查是否还有漏网之鱼
    found_banned = False
    all_remaining_elements = set()

    for i in range(total_samples):
        n = num_atoms[i]
        atoms = set(charges[i, :n].astype(int))

        # 收集剩余元素种类
        all_remaining_elements.update(atoms)

        # 检查是否包含被禁元素
        for banned_z in BANNED_ELEMENTS:
            if banned_z in atoms:
                print(f"❌ 警告: 在索引 {i} 发现残留的 {BANNED_ELEMENTS[banned_z]}!")
                found_banned = True

    if not found_banned:
        print("✅ 验证通过：新文件中不再包含 磷, 硫, 氯, 溴, 碘, 硼。")

    # 2. 统计剩下的元素
    print("\n剩下的分子包含以下元素种类:")
    sorted_elements = sorted(list(all_remaining_elements))
    for z in sorted_elements:
        symbol = get_element_symbol(z)
        print(f"  - Z={z} ({symbol})")

if __name__ == "__main__":
    analyze_and_filter()




