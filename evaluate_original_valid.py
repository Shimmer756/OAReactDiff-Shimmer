import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from reactot.trainer.pl_trainer import SBModule
from reactot.dataset.transition1x import ProcessedTS1x

# 配置类
class OPT:
    def __init__(self):
        self.solver = "ddpm"
        self.method = "midpoint"
        self.atol = 1e-2
        self.rtol = 1e-2

def main():
    # === 1. 配置路径 ===
    ckpt_path = "reactot-pretrained.ckpt"
    
    # 指向官方那个 55MB 的“全量”验证集文件
    # (根据你之前的 ls 记录，它在 reactot/data/ 下，或者 reactot/data0/ 下)
    data_path = "reactot/data/valid_rpsb_all.pkl" 

    if not os.path.exists(data_path):
        print(f"❌ 错误: 找不到文件 {data_path}")
        return

    # === 2. 加载模型 ===
    print(f"🔧 正在加载模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SBModule.load_from_checkpoint(ckpt_path, map_location=device)
    model.ddpm.opt = OPT()
    model.to(device)
    model.eval()

    # === 3. 加载数据 (关键修改) ===
    print(f"📂 正在加载数据: {data_path}")
    print("👉 正在启用 use_by_ind=True 以读取内置索引...")

    dataset = ProcessedTS1x(
        npz_path=data_path,
        center=True,
        device=device,
        remove_h=False,
        single_frag_only=False, #试一下
        
        # === 【核心修改】 ===
        # 开启这个开关，代码就会去读文件里的 'use_ind' 字段
        # 从而只加载那 1000 多条验证数据，过滤掉训练数据
        use_by_ind=True  
        # ==================
    )

    # === 4. 验证数据量 ===
    sample_count = len(dataset)
    print(f"📊 过滤后的样本数: {sample_count}")
    
    # 预期应该是 1000 左右 (官方测试集大小)
    # 如果还是 7000+，说明 use_ind 没生效或者文件里的 use_ind 是全量的
    if sample_count > 2000:
        print("⚠️ 警告: 样本数依然很大，use_ind 可能未生效！")
    else:
        print("✅ 样本数正常，成功提取了验证集子集。")

    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )

    # === 5. 运行评估 ===
    print("\n🚀 开始评估...")
    results, all_rmsds = model.eval_rmsd(
        dataloader,
        verbose=True,
        write_xyz=False,
    )

    print("\n" + "="*40)
    print("🏆 官方验证集评估结果 (use_ind)")
    print("="*40)
    print(f"Mean RMSD:   {results['rmsd_mean']:.4f} Å")
    print(f"Median RMSD: {results['rmsd_median']:.4f} Å")
    print("-" * 40)

if __name__ == "__main__":
    main()
