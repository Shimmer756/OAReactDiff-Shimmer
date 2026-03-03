import torch
import os
from torch.utils.data import DataLoader
from reactot.trainer.pl_trainer import DDPMModule
from reactot.dataset.transition1x import ProcessedTS1x
from reactot.diffusion._normalizer import FEATURE_MAPPING
from reactot.diffusion._schedule import DiffSchedule, PredefinedNoiseSchedule

def set_new_schedule(
    ddpm_trainer: DDPMModule,
    timesteps: int = 250,
    device: torch.device = torch.device("cuda"),
    noise_schedule: str = "polynomial_2"
) -> DDPMModule:
    gamma_module = PredefinedNoiseSchedule(
        noise_schedule=noise_schedule,
        timesteps=timesteps,
        precision=1e-5,
    )
    schedule = DiffSchedule(
        gamma_module=gamma_module,
        norm_values=ddpm_trainer.ddpm.norm_values
    )
    ddpm_trainer.ddpm.schedule = schedule
    ddpm_trainer.ddpm.T = timesteps
    return ddpm_trainer.to(device)

def sample_meci_from_ckpt(ckpt_path, output_xyz="generated_meci.xyz", num_samples=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 正在加载权重: {ckpt_path}")
    
    model = DDPMModule.load_from_checkpoint(ckpt_path, map_location=device, strict=False)
    model = set_new_schedule(model, timesteps=150, noise_schedule="polynomial_2")
    model.eval()
    
    print("📦 正在加载验证集数据...")
    dataset = ProcessedTS1x(
        npz_path="reactot/data_meci/train_rpsb_all_centered.pkl", 
        center=True, pad_fragments=0, device="cuda", zero_charge=False, remove_h=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

    # 💡 核心探测：看看模型到底有几个编码器
    n_frags = len(model.ddpm.dynamics.encoders)
    print(f"🕵️ 模型内部架构探测: 包含 {n_frags} 个物理编码器。")
    print(f"🚀 开始生成 MECI 结构，目标生成数量: {num_samples}")
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_samples: break
            
            representations, conditions = batch
            conditions = conditions.to(device)
            
            if n_frags == 1:
                # 🎯 单片段模式：模型直接生成 MECI
                # 我们提取数据集中第1个（或者唯一一个）片段的 size
                meci_idx = 1 if len(representations) > 1 else 0
                fragments_nodes = [representations[meci_idx]["size"].to(device)]
                
                out_samples, _ = model.ddpm.sample(
                    n_samples=1,
                    fragments_nodes=fragments_nodes,
                    conditions=conditions,
                    return_frames=1,
                    timesteps=None
                )
                # 提取最后一步生成的构型
                final_pred = out_samples[0][-1] 
                
            else:
                # 🎯 双/多片段模式：模型用基态推导 MECI
                xh_fixed = [
                    torch.cat([rep[ft] for ft in FEATURE_MAPPING], dim=1).to(device)
                    for rep in representations[:n_frags]
                ]
                fragments_nodes = [rep["size"].to(device) for rep in representations[:n_frags]]
                
                out_samples, _ = model.ddpm.inpaint(
                    n_samples=1,
                    fragments_nodes=fragments_nodes,
                    conditions=conditions,
                    return_frames=1,
                    resamplings=1, 
                    jump_length=1,
                    timesteps=None,
                    xh_fixed=xh_fixed,
                    frag_fixed=[0]
                )
                final_pred = out_samples[0][1]
            
            # === 解析坐标并写入文件 ===
            pos = final_pred[:, :3].cpu().numpy()
            z = final_pred[:, -1].cpu().numpy()
            
            with open(output_xyz, "a" if i > 0 else "w") as f:
                f.write(f"{len(z)}\nGenerated_MECI_Sample_{i}\n")
                for atom_idx in range(len(z)):
                    f.write(f"{int(z[atom_idx])} {pos[atom_idx][0]:.6f} {pos[atom_idx][1]:.6f} {pos[atom_idx][2]:.6f}\n")
            
            print(f"✅ 成功生成第 {i+1}/{num_samples} 个分子")

    print(f"🎉 恭喜！所有的生成的分子已保存为: {output_xyz}")

if __name__ == "__main__":
    CKPT_PATH = "checkpoint/R2P_Finetune/last.ckpt" 
    sample_meci_from_ckpt(CKPT_PATH)
