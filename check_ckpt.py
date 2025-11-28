import torch
import sys
from pathlib import Path

# 1. 配置您的 Checkpoint 路径
CKPT_PATH = "checkpoint/OAReactDiff/leftnet-1-0d69b68764ba/last.ckpt"

def inspect_ckpt(path):
    print(f"📂 正在加载检查点: {path}")
    if not Path(path).exists():
        print("❌ 文件不存在！")
        return

    try:
        # 加载到 CPU
        # 添加 weights_only=False 以允许加载自定义模型类
        checkpoint = torch.load(path, map_location="cpu", weights_only=False) 
        
        print("\n====== 🛠️ 训练超参数 (Hyperparameters) ======")
        hparams = checkpoint.get("hyper_parameters", {})
        
        # 1. 检查 node_nfs (原子特征维度)
        # 这决定了模型“认为”有多少种原子
        node_nfs = hparams.get("node_nfs", "未找到")
        print(f"🔹 node_nfs: {node_nfs}")
        if isinstance(node_nfs, list) and len(node_nfs) > 0:
            dim = node_nfs[0]
            print(f"   -> 每个原子的特征维数: {dim}")
            # 3(pos) + N(cat) + 1(charge) = dim
            # N = dim - 4
            print(f"   -> 推测原子类型数量 (One-Hot): {dim - 4}")
            if dim - 4 == 5:
                print("   ✅ 对应 5 种元素 (H, C, N, O, F)")
            elif dim - 4 == 11:
                print("   ⚠️ 对应 11 种元素 (含 S, Cl...)")
            else:
                print(f"   ❓ 未知配置")

        # 2. 检查 Normalizer (归一化参数)
        # 如果这里是极小值，反归一化时就会导致坐标爆炸
        norm_val = hparams.get("norm_values", "未找到")
        print(f"🔹 norm_values: {norm_val}")
        
        # 3. 检查网络配置
        model_config = hparams.get("model_config", {})
        in_hidden = model_config.get("in_hidden_channels", "未找到")
        print(f"🔹 in_hidden_channels: {in_hidden}")

        print("\n====== 🧠 权重统计 ======")
        state_dict = checkpoint["state_dict"]
        # 检查是否有 NaN 或 Inf 的权重
        has_nan = False
        max_val = 0.0
        for k, v in state_dict.items():
            if torch.is_tensor(v):
                if torch.isnan(v).any():
                    print(f"❌ 警告: 参数 {k} 包含 NaN!")
                    has_nan = True
                if torch.isinf(v).any():
                    print(f"❌ 警告: 参数 {k} 包含 Inf!")
                
                curr_max = v.abs().max().item()
                if curr_max > max_val:
                    max_val = curr_max
        
        print(f"✅ 权重最大绝对值: {max_val:.4f}")
        if max_val > 1000:
            print("⚠️ 警告: 权重数值异常大，模型可能已发散！")
        
        if not has_nan and max_val < 100:
            print("✅ 权重数值范围看起来正常。")

    except Exception as e:
        print(f"读取失败: {e}")

if __name__ == "__main__":
    inspect_ckpt(CKPT_PATH)
