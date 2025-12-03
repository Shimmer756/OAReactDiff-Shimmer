import torch
import copy
from uuid import uuid4
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from reactot.trainer.pl_trainer import SBModule, DDPMModule
from reactot.trainer.ema import EMACallback
from reactot.model import LEFTNet

# === 1. 配置区域 ===
class OPT:
    def __init__(self):
        self.solver = "ode"
        self.method = "midpoint"
        self.atol = 1e-2
        self.rtol = 1e-2

# 基础模型配置 (LEFTNet)
leftnet_config = dict(
    pos_require_grad=False,
    cutoff=10.0,
    num_layers=6,
    hidden_channels=196,
    num_radial=96,
    in_hidden_channels=8,
    reflect_equiv=True,
    legacy=True,
    update=True,
    pos_grad=False,
    single_layer_output=True,
    object_aware=True,
)

optimizer_config = dict(lr=1e-4, betas=[0.9, 0.999], weight_decay=0, amsgrad=True)

# 训练配置
training_config = dict(
    datadir="reactot/data/transition1x/", # 请确保这里指向你的数据目录
    remove_h=False,
    bz=14,
    num_workers=0,
    clip_grad=True,
    gradient_clip_val=None,
    ema=True,
    ema_decay=0.999,
    swapping_react_prod=False, # R->P 是有方向的，建议关闭交换
    append_frag=False,
    use_by_ind=True,
    reflection=False,
    single_frag_only=False,
    only_ts=False,
    lr_schedule_type=None,
    use_sampler=True,
    sampler_config=dict(max_num=2800, mode="node^2", shuffle=True, ddp=False)
)

# === 关键修改：针对 R->P 任务的配置 ===
fragment_names = ["R", "P"]  # 只保留 R 和 P
node_nfs = [9, 9]            # 只有两个节点特征输入
fixed_idx = [0]              # 固定索引 0 (R), 生成索引 1 (P)
idx = 1                      # 告诉模型我们要预测的是列表中的第2个元素 (P)

# 映射配置：从 R 生成 P
mapping = "R->P"
mapping_initial = "R"        # 初始状态设为 R

# 预训练权重路径
checkpoint_path = "reactot-pretrained.ckpt" 

# === 2. 权重加载与“手术”函数 ===
def load_and_adapt_checkpoint(ckpt_path):
    print(f"🔧 正在加载并适配预训练权重: {ckpt_path}")
    # 加载原始权重
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    
    new_state_dict = {}
    
    # 遍历权重进行迁移
    for k, v in state_dict.items():
        # 处理 encoder (输入层)
        if "dynamics.encoders" in k:
            if "encoders.0" in k:   # R -> R (保留)
                new_state_dict[k] = v
            elif "encoders.1" in k: # TS -> 丢弃
                print(f"   - 丢弃 TS 权重: {k}")
                continue
            elif "encoders.2" in k: # P -> 移位到 1
                new_k = k.replace("encoders.2", "encoders.1")
                print(f"   - 迁移 P  权重: {k} -> {new_k}")
                new_state_dict[new_k] = v
                
        # 处理 decoder (输出层)
        elif "dynamics.decoders" in k:
            if "decoders.0" in k:   # R -> R (保留)
                new_state_dict[k] = v
            elif "decoders.1" in k: # TS -> 丢弃
                continue
            elif "decoders.2" in k: # P -> 移位到 1
                new_k = k.replace("decoders.2", "decoders.1")
                new_state_dict[new_k] = v
                
        # 其他层 (LEFTNet 主干) -> 直接保留
        else:
            new_state_dict[k] = v
            
    return new_state_dict

# === 3. 主训练流程 ===
def main():
    seed_everything(42, workers=True)
    
    # 1. 初始化模型 (使用新的 2-fragment 配置)
    ddpm = SBModule(
        model_config=leftnet_config,
        optimizer_config=optimizer_config,
        training_config=training_config,
        node_nfs=node_nfs,          # [9, 9]
        edge_nf=0,
        condition_nf=1,
        fragment_names=fragment_names, # ["R", "P"]
        pos_dim=3,
        update_pocket_coords=True,
        condition_time=True,
        edge_cutoff=None,
        norm_values=(1., 1., 1.),
        norm_biases=(0., 0., 0.),
        noise_schedule="cosine",
        timesteps=3000,
        precision=1e-5,
        loss_type="l2",
        pos_only=True,
        process_type="TS1x",
        model=LEFTNet,
        enforce_same_encoding=None,
        scales=[1., 1.],            # 对应 R, P
        fixed_idx=fixed_idx,        # [0]
        eval_epochs=1,
        mapping=mapping,            # "R->P"
        mapping_initial=mapping_initial, # "R"
        nfe=25,
        beta_max=0.3,
        ot_ode=True,
        power=0.5,
        inv_power=1,
        sigma=0.,
        ts_guess=None,
        idx=idx                     # 1 (生成 P)
    )
    
    # 2. 加载经过“手术”的权重
    adapted_state_dict = load_and_adapt_checkpoint(checkpoint_path)
    missing, unexpected = ddpm.load_state_dict(adapted_state_dict, strict=False)
    
    print(f"\n✅ 权重加载完成。")
    print(f"   Missing keys (应为空或仅包含无关项): {len(missing)}")
    print(f"   Unexpected keys (应为空): {len(unexpected)}")
    
    ddpm.ddpm.opt = OPT() # 注入优化器配置

    # 3. 配置 Trainer
    wandb_logger = WandbLogger(
        project="ReactOT-R2P-Finetune",
        name=f"R2P-{str(uuid4())[:8]}",
        log_model=False
    )
    
    callbacks = [
        EarlyStopping(monitor="val_ep_scaled_err", patience=200, verbose=True),
        ModelCheckpoint(
            monitor="val_ep_scaled_err",
            dirpath="checkpoint/R2P_Finetune/",
            filename="r2p-{epoch:03d}-{val_ep_scaled_err:.4f}",
            save_top_k=3
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    if training_config["ema"]:
        callbacks.append(EMACallback(pl_module=ddpm, decay=training_config["ema_decay"]))

    trainer = Trainer(
        max_epochs=1000, # 微调不需要太久
        accelerator="gpu",
        devices=[0], # 假设使用单卡，多卡需调整
        strategy="auto",
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=training_config["gradient_clip_val"],
        limit_train_batches=1.0, 
        limit_val_batches=1.0,
    )

    # 4. 开始训练
    print("\n🚀 开始 R->P 任务微调...")
    trainer.fit(ddpm)

if __name__ == "__main__":
    main()
