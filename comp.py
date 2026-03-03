import pickle
from collections import Counter
import numpy as np

# 官方数据集路径
path_official = "/root/OAReactDiff-Shimmer/reactot/data/train_rpsb_all.pkl"

print(f"📥 正在加载官方数据: {path_official}")
with open(path_official, 'rb') as f:
    data_off = pickle.load(f)

# 提取 reactant 里的所有 num_atoms
if 'reactant' in data_off and 'num_atoms' in data_off['reactant']:
    num_atoms_list = data_off['reactant']['num_atoms']
    
    # 将其统一转化为列表
    if isinstance(num_atoms_list, np.ndarray):
        num_atoms_list = num_atoms_list.tolist()
    elif isinstance(num_atoms_list, int):
        num_atoms_list = [num_atoms_list]
        
    counts = Counter(num_atoms_list)
    
    print("\n" + "="*40)
    print(f"📊 官方数据集【原子数量】统计结果")
    print("="*40)
    print(f"包含的分子总数: {len(num_atoms_list)} 个")
    print(f"最小原子数: {min(num_atoms_list)}")
    print(f"最大原子数: {max(num_atoms_list)}")
    print(f"平均原子数: {np.mean(num_atoms_list):.2f}")
    
    print("\n📈 具体的尺寸分布如下:")
    # 按原子数量从小到大排序打印
    for size, count in sorted(counts.items()):
        print(f"  👉 含有 {size:2d} 个原子的分子: {count:4d} 个")
    print("="*40)
else:
    print("❌ 在官方数据集中未找到 'num_atoms' 字段！")
