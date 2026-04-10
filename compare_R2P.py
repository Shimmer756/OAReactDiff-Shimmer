import pickle
import numpy as np
from scipy.spatial.transform import Rotation

def calculate_rmsd(pos1, pos2):
    p1_centered = pos1 - np.mean(pos1, axis=0)
    p2_centered = pos2 - np.mean(pos2, axis=0)
    rotation, _ = Rotation.align_vectors(p2_centered, p1_centered)
    p1_aligned = rotation.apply(p1_centered)
    return np.sqrt(np.mean(np.sum((p1_aligned - p2_centered) ** 2, axis=1)))

def find_best_candidates():
    data_path = "reactot/data_meci/valid_rpsb_filtered_30.pkl"
    
    print(f"📂 正在加载验证集: {data_path}")
    with open(data_path, 'rb') as f:
        valid_data = pickle.load(f)

    reactants = valid_data['reactant']['positions']
    products_true = valid_data['product']['positions']
    
    total_mols = len(reactants)
    metrics = []

    for i in range(total_mols):
        r_pos = np.array(reactants[i])
        p_pos = np.array(products_true[i])
        
        # 计算 起始态 到 真实 MECI 的几何跨度
        span_rmsd = calculate_rmsd(r_pos, p_pos)
        metrics.append((i, span_rmsd))

    # 按形变剧烈程度降序排列
    metrics.sort(key=lambda x: x[1], reverse=True)

    print("\n🎯 推荐用于论文的 Top 5 高动态跨度分子候选：")
    print("="*60)
    print(f"{'索引 (Index)':<15} | {'起始态 -> 真实 MECI 的对齐 RMSD (Å)':<35}")
    print("-" * 60)
    
    for idx, span_rmsd in metrics[:5]:
        marker = "🔥 (首选)" if span_rmsd > 0.4 else ""
        print(f" {idx:<13} |  {span_rmsd:.4f} {marker}")
    print("="*60)
    print("💡 下一步：请将上述列表中 RMSD 最大的 Index 填入之前的 meci.py 脚本中运行。")

if __name__ == "__main__":
    find_best_candidates()
