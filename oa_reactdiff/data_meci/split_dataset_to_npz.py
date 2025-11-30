import numpy as np
import os

# --- 配置 ---
INPUT_FILE = "filtered_dataset.npz"  # 输入文件
TRAIN_FILE = "train_meci.npz"        # 输出训练集 (改为 .npz)
VALID_FILE = "valid_meci.npz"        # 输出验证集 (改为 .npz)
SPLIT_RATIO = 0.9  # 90% 训练，10% 验证
RANDOM_SEED = 42   # 固定随机种子

def split_and_save_npz():
    print(f"正在读取文件: {INPUT_FILE} ...")

    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到文件 {INPUT_FILE}，请确保上一步已成功运行。")
        return

    # 1. 加载原始 NPZ 数据
    # allow_pickle=True 是必须的，以防包含对象数组
    raw_npz = np.load(INPUT_FILE, allow_pickle=True)
    # 将 NpzFile 对象转换为字典，方便操作
    data_dict = {key: raw_npz[key] for key in raw_npz.files}

    # 2. 确定样本总数 N
    # 使用 'reactant/num_atoms' 作为参考
    try:
        num_samples = data_dict['reactant/num_atoms'].shape[0]
    except KeyError:
        print("错误：无法在数据中找到 'reactant/num_atoms' 来确定样本数。")
        return

    print(f"数据集总样本数: {num_samples}")

    # 3. 生成随机索引并划分
    np.random.seed(RANDOM_SEED) 
    indices = np.arange(num_samples)
    np.random.shuffle(indices) 

    # 计算切分点
    split_point = int(num_samples * SPLIT_RATIO)

    train_indices = indices[:split_point]
    valid_indices = indices[split_point:]

    print(f"划分计划: 训练集 {len(train_indices)} 个, 验证集 {len(valid_indices)} 个")

    # 4. 构建训练和验证字典
    train_data = {}
    valid_data = {}

    for key, arr in data_dict.items():
        # 检查是否为数据数组（第一维等于样本数）
        # 这种逻辑可以区分样本数据（需要切分）和元数据（全局配置，需完整复制）
        if hasattr(arr, 'shape') and arr.shape[0] == num_samples:
            train_data[key] = arr[train_indices]
            valid_data[key] = arr[valid_indices]
        else:
            # 假如有些元数据不是按样本存储的（例如全局配置），直接拷贝
            print(f"提示: Key '{key}' 似乎是元数据（不随样本切分），已完整复制。")
            train_data[key] = arr
            valid_data[key] = arr

    # 5. 保存为 NPZ 文件
    # 使用 np.savez_compressed 可以有效减小文件体积
    print(f"\n正在保存 {TRAIN_FILE} ...")
    np.savez_compressed(TRAIN_FILE, **train_data)

    print(f"正在保存 {VALID_FILE} ...")
    np.savez_compressed(VALID_FILE, **valid_data)

    # 6. 最终验证
    print("\n" + "="*30)
    print("划分完成验证")
    print("="*30)
    verify_npz(TRAIN_FILE, "训练集")
    verify_npz(VALID_FILE, "验证集")

def verify_npz(filepath, label):
    """简单的验证函数，读取 NPZ 并打印大小"""
    try:
        # 使用 np.load 读取
        data = np.load(filepath, allow_pickle=True)
        
        # 检查核心 Key
        if 'reactant/num_atoms' in data:
            n = data['reactant/num_atoms'].shape[0]
            print(f"✅ {label} ({filepath}): 读取成功，包含 {n} 个分子。")
            # 还可以打印一下文件大小
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"   文件大小: {size_mb:.2f} MB")
        else:
            print(f"⚠️ {label} ({filepath}): 读取成功，但缺少 'reactant/num_atoms' 键。")
            
    except Exception as e:
        print(f"❌ {label} ({filepath}): 读取失败 - {e}")

if __name__ == "__main__":
    split_and_save_npz()
