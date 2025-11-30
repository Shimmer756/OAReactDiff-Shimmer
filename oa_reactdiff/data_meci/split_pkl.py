import numpy as np
import pickle
import os

# --- 配置 ---
INPUT_FILE = "filtered_dataset.npz"
TRAIN_FILE = "train_meci.pkl"
VALID_FILE = "valid_meci.pkl"
SPLIT_RATIO = 0.9  # 90% 训练，10% 验证
RANDOM_SEED = 42   # 固定随机种子，保证每次运行划分结果一致

def split_and_save_pkl():
    print(f"正在读取文件: {INPUT_FILE} ...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到文件 {INPUT_FILE}，请确保上一步已成功运行。")
        return

    # 1. 加载 NPZ 数据
    # 将 NpzFile 对象转换为标准的 Python 字典，方便后续 Pickle 序列化
    npz_data = np.load(INPUT_FILE, allow_pickle=True)
    data_dict = {key: npz_data[key] for key in npz_data.files}
    
    # 2. 确定样本总数 N
    # 我们使用 'reactant/num_atoms' 作为参考，因为它是一维数组 (N,)
    try:
        num_samples = data_dict['reactant/num_atoms'].shape[0]
    except KeyError:
        print("错误：无法在数据中找到 'reactant/num_atoms' 来确定样本数。")
        return

    print(f"数据集总样本数: {num_samples}")

    # 3. 生成随机索引并划分
    np.random.seed(RANDOM_SEED) # 设置种子
    indices = np.arange(num_samples)
    np.random.shuffle(indices)  # 随机打乱

    # 计算切分点
    split_point = int(num_samples * SPLIT_RATIO)
    
    train_indices = indices[:split_point]
    valid_indices = indices[split_point:]
    
    print(f"划分计划: 训练集 {len(train_indices)} 个, 验证集 {len(valid_indices)} 个")

    # 4. 构建训练和验证字典
    train_data = {}
    valid_data = {}

    for key, arr in data_dict.items():
        # 检查该数组是否是数据数组（第一维等于样本数）
        # 如果是，则进行切片；如果是元数据（如全局配置），则直接复制
        if hasattr(arr, 'shape') and arr.shape[0] == num_samples:
            train_data[key] = arr[train_indices]
            valid_data[key] = arr[valid_indices]
        else:
            # 假设非样本对应的数据（metadata）在两个集里都需要保留
            print(f"提示: Key '{key}' 似乎不是样本数据（维度不匹配），将完整复制到两个文件中。")
            train_data[key] = arr
            valid_data[key] = arr

    # 5. 保存为 PKL 文件
    print(f"\n正在保存 {TRAIN_FILE} ...")
    with open(TRAIN_FILE, 'wb') as f:
        pickle.dump(train_data, f)
    
    print(f"正在保存 {VALID_FILE} ...")
    with open(VALID_FILE, 'wb') as f:
        pickle.dump(valid_data, f)

    # 6. 最终验证
    print("\n" + "="*30)
    print("划分完成验证")
    print("="*30)
    verify_pkl(TRAIN_FILE, "训练集")
    verify_pkl(VALID_FILE, "验证集")

def verify_pkl(filepath, label):
    """简单的验证函数，读取 PKL 并打印大小"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # 检查一个核心 Key 的长度
        n = data['reactant/num_atoms'].shape[0]
        print(f"✅ {label} ({filepath}): 读取成功，包含 {n} 个分子。")
    except Exception as e:
        print(f"❌ {label} ({filepath}): 读取失败 - {e}")

if __name__ == "__main__":
    split_and_save_pkl()
