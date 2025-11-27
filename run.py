import torch
import numpy as np
from pathlib import Path
from typing import List
from torch.utils.data import DataLoader

# 导入项目模块
from oa_reactdiff.trainer.pl_trainer import DDPMModule
from oa_reactdiff.dataset.transition1x import ProcessedTS1x
from oa_reactdiff.diffusion._normalizer import FEATURE_MAPPING

# --- 1. 配置 ---
# 请确保路径正确
MODEL_CHECKPOINT_PATH = "checkpoint/OAReactDiff/leftnet-0-20f22da4eb62/ddpm-epoch=1978-val-totloss=300.81.ckpt"
CUSTOM_DATA_PATH = Path("oa_reactdiff/data/transition1x/train_addprop.pkl")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 必要的模型配置 (与训练时一致)
leftnet_config = dict(
    pos_require_grad=False, cutoff=10.0, num_layers=6, hidden_channels=196, 
    num_radial=96, in_hidden_channels=14, reflect_equiv=True, legacy=True,
    update=True, pos_grad=False, single_layer_output=True, object_aware=True,
)
# 推理时的配置 (注意 batch_size=1)
inference_config = dict(
    datadir=str(CUSTOM_DATA_PATH), remove_h=False, 
    bz=1,             # <--- 关键修改：批次大小设为 1
    num_workers=0, swapping_react_prod=False, single_frag_only=False,
    clip_grad=False, gradient_clip_val=None, ema=False, ema_decay=0.999,
)

# --- 2. 核心生成函数 ---
def generate_single_molecule(ddpm_module, data_loader):
    model = ddpm_module.ddpm.to(device)
    model.eval()
    
    print(f"🚀 开始生成单个分子...")
    
    # 只获取第一个 batch
    for batch_idx, batch in enumerate(data_loader):
        representations, conditions = batch
        
        # 移动数据到 GPU
        representations = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in r.items()} for r in representations]
        conditions = conditions.to(device)
        
        # 准备固定输入 (Reactant)
        xh_fixed = [
            torch.cat([repre[feature_type] for feature_type in FEATURE_MAPPING], dim=1)
            for repre in representations
        ]
        
        n_samples = representations[0]["size"].size(0)
        fragments_nodes = [repre["size"] for repre in representations]
        
        # 核心：只固定 Reactant (索引 0)
        frag_fixed = [0,2] 

        with torch.no_grad():
            out_samples, _ = model.inpaint(
                n_samples=n_samples,
                fragments_nodes=fragments_nodes,
                conditions=conditions,
                return_frames=1,
                resamplings=1, 
                jump_length=1, 
                xh_fixed=xh_fixed,
                frag_fixed=frag_fixed,
            )
        
        # 提取生成的产物 (Index 2)
        generated_fragments = out_samples[0]
        #gen_product_tensor = generated_fragments[2]
        gen_TS_tensor = generated_fragments[1]

        # 获取 3D 坐标 (前3列)
        # 注意：这里还是归一化后的坐标，如果您想看真实坐标，通常模型内部已经处理了，
        # 或者需要手动调用 normalizer.unnormalize (但 out_samples 通常是处理过的 xh)
        # 查看 en_diffusion.py 的 sample_p_xh_given_z0 返回的是 pos_0 (已反归一化)
        # 但是 inpaint 返回的 out_samples 是什么？
        # inpaint 返回的是 out_samples[0] = [cat(pos, cat, charge), ...]
        # 并且在 inpaint 内部调用 sample_p_xh_given_z0 时已经做过反归一化了。
        
        gen_TS_pos = gen_TS_tensor[:, :3].cpu().numpy()
        gen_TS_atom_types = gen_TS_tensor[:, 3:-1].cpu().numpy() # One-hot
        gen_TS_charges = gen_TS_tensor[:, -1].cpu().numpy() # Charge
        
        print("\n✅ 生成完成！")
        print(f"生成的产物坐标形状: {gen_TS_pos.shape}")
        print("坐标数据 (前 5 行):")
        print(gen_TS_pos[:5])
        
        # ！！！关键：生成一个后立刻退出循环！！！
        break 

# --- 3. 执行 ---
if __name__ == "__main__":
    # 加载模型
    print("正在加载模型...")
    ddpm_module = DDPMModule.load_from_checkpoint(
        checkpoint_path=MODEL_CHECKPOINT_PATH,
        map_location=device,
        model_config=leftnet_config,
        training_config=inference_config,
    )
    
    # 加载数据
    print("正在加载数据...")
    dataset = ProcessedTS1x(npz_path=str(CUSTOM_DATA_PATH), **inference_config)
    
    # DataLoader 使用 batch_size=1
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

    # 运行
    generate_single_molecule(ddpm_module, loader)
