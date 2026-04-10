import torch
# 开启全局 NaN 追踪器！一旦产生 NaN，立刻报错并定位到具体代码行！
torch.autograd.set_detect_anomaly(True)

import copy
from uuid import uuid4
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import Callback ##LYY

from reactot.trainer.pl_trainer import SBModule, DDPMModule
from reactot.trainer.ema import EMACallback
from reactot.model import LEFTNet

#LYY
from reactot.dynamics.xmace_potential import XMACEPotential



class PrintMetricsCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # 获取当前 Epoch 的指标
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        # 提取关键指标 (如果存在)
        val_loss = metrics.get("val_ep_loss", "N/A")
        val_err = metrics.get("val_ep_scaled_err", "N/A")
        val_rmsd = metrics.get("val_ep_rmsd_mean", "N/A")

        # 格式化打印
        print(f"\n[Epoch {epoch}] -----------------------------")
        print(f"  📉 Val Loss:        {val_loss:.6f}" if isinstance(val_loss, float) else f"  Val Loss: {val_loss}")
        print(f"  🎯 Val Scaled Err:  {val_err:.6f}" if isinstance(val_err, float) else f"  Val Scaled Err: {val_err}")
        print(f"  📏 Val Mean RMSD:   {val_rmsd:.6f}" if isinstance(val_rmsd, float) else f"  Val RMSD: {val_rmsd}")
        print("------------------------------------------\n")


# === 1. 配置区域 ===
class OPT:
    def __init__(self):
        # 【关键配置】使用 ddpm 求解器配合 ot_ode=True，这是最稳健的组合
        self.solver = "ddpm"   
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

optimizer_config = dict(lr=1e-5, betas=[0.9, 0.999], weight_decay=0, amsgrad=True)

# 训练配置
training_config = dict(
    # 【关键配置】指向你转换好的数据目录
    datadir="reactot/data_meci/", 
    
    remove_h=False,
    bz=4,  # 如果显存够大，可以尝试调大到 32 或 64
    num_workers=0, # 服务器上建议开启多进程读取 (例如 4 或 8)
    clip_grad=True,
    gradient_clip_val=1.0,
    ema=True,
    ema_decay=0.999,
    swapping_react_prod=False, # R->P 是有方向的，必须关闭交换
    append_frag=False,
    use_by_ind=False,
    reflection=False,
    single_frag_only=False,
    only_ts=False,
    lr_schedule_type=None,
    use_sampler=True,
    #sampler_config=dict(max_num=2800, mode="node^2", shuffle=True, ddp=False),
    sampler_config=dict(max_num=15000, mode="node^2", shuffle=True, ddp=False)
)

# === 针对 R->P 任务的配置 ===
fragment_names = ["R", "P"]  # 只保留 R 和 P
node_nfs = [9, 9]            # 只有两个节点特征输入
fixed_idx = [0]              # 固定索引 0 (R), 生成索引 1 (P)
idx = 1                      # 告诉模型我们要预测的是列表中的第2个元素 (P)

# 映射配置：从 R 生成 P
mapping = "R->P"
mapping_initial = "R"        # 初始状态设为 R

# 预训练权重路径
checkpoint_path = "reactot-pretrained.ckpt" 

#LYY_2
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


# ... 之前的代码 (如 load_and_adapt_checkpoint 函数) ...

# === 这一段是你需要新增进去的代码 ===
class PhysicsInformedSBModule(SBModule):
    def __init__(self, mace_model_path, phys_weight=0.01 , *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.physics_engine = XMACEPotential(mace_model_path=mace_model_path)
        self.phys_weight = phys_weight  
        # 【新增这一行】：从传入的参数中抓取 idx，如果没传默认设为 1 (代表 P 态)
        self.idx = kwargs.get("idx", 1)

    def training_step(self, batch, batch_idx):
        # 1. 跑原始的几何流匹配，拿到 geo_loss
        loss_dict = super().training_step(batch, batch_idx)
        # 🌟 绕开原作者花里胡哨的父类，直捣黄龙只算 Loss！绝对不抽样！
        #loss_dict = self.compute_loss(batch)
        geo_loss = loss_dict["loss"]
            

        try:
            # 2. 从流匹配过程中，拿到当前目标分子 P 的坐标图 (React-OT 格式)
            #target_dict = batch[0][self.idx]
            # 2. 🌟 从流匹配黑盒里，拿出大模型【预测出来的、带有梯度的】坐标！
            #pred_pos = loss_dict["pred_pos_unnorm"]
            
            # 2. 🌟 绕开 loss_dict！直接去模型本体身上拿坐标！
            pred_pos = self.ddpm.current_pred_pos

            # 3. 🌟 浅拷贝一份真实的 target_dict，狸猫换太子！
            # 把原本没有梯度的死坐标，换成我们带有梯度的预测坐标！
            target_dict = dict(batch[0][self.idx])
            target_dict["pos"] = pred_pos  # 核心替换！计算图在这里接通！

            # 4. 让 X-MACE 算一算当前构型的 S0-S1 能量差打分
            phys_ae, _ , gap_mean = self.physics_engine.forward_energy_only(target_dict)
            #phys_loss = torch.mean(phys_ae)
            # ====================================================
            # 🎯 拨乱反正！我们要优化的是 Gap 趋近于 0，而不是绝对能量！
            # ====================================================
            
            #phys_loss = gap_mean 
            
            #做个2次方
            phys_loss = gap_mean ** 2

            if torch.isnan(phys_loss) or torch.isinf(phys_loss):
                phys_loss = torch.tensor(0.0, device=geo_loss.device, requires_grad=True)
            else:
                # 哪怕初期模型瞎猜导致 Gap 高达 20 eV，我们也强制把惩罚截断在 5.0 以内
                # 这样物理梯度会非常温和地把分子往 Gap=0 的方向拉，绝对不会拉崩！
                phys_loss = torch.clamp(phys_loss, min=0.0, max=5.0)

            # 4. 联合 Loss
            total_loss = geo_loss + self.phys_weight * phys_loss
            
            # 打印日志到 Wandb / 进度条
            self.log("geo_loss", geo_loss, prog_bar=True)
            self.log("phys_loss", phys_loss, prog_bar=True)
            # 【核心修改】把能量差显示在进度条上，这就是你找 MECI 的核心指标！
            self.log("S1_S0_gap", gap_mean, prog_bar=True)

            return total_loss
              
        except Exception as e:
            # === 新增以下两行来显示完整报错轨迹 ===
            import traceback
            traceback.print_exc()
            # 如果因为数据解析发生意外，回退到只算结构 Loss 以防训练崩溃
            print(f"⚠️ 物理引擎遇到错误: {e}，回退至纯结构 Loss。")
            return geo_loss
        
# ====================================



# === 3. 主训练流程 ===
def main():
    seed_everything(42, workers=True)
    current_w = 0
    w_tag = f"w{current_w}" # 自动生成标签


    # === 原本的 ddpm = SBModule(...) 替换为下面这整段 ===
    ddpm = PhysicsInformedSBModule(
        mace_model_path="/root/X-MACE_2/meci_energies_forces.model", 
        phys_weight = current_w,  
        
        model_config=leftnet_config,
        optimizer_config=optimizer_config,
        training_config=training_config,
        node_nfs=node_nfs,          
        edge_nf=0,
        condition_nf=1,
        fragment_names=fragment_names, 
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
        power=0.5,                   
        ot_ode=True,                 
        
        model=LEFTNet,
        enforce_same_encoding=None,
        scales=[1., 1.],            
        fixed_idx=fixed_idx,        
        eval_epochs=1,
        mapping=mapping,            
        mapping_initial=mapping_initial, 
        nfe=25,
        beta_max=0.3,
        inv_power=1,
        sigma=0.,
        ts_guess=None,
        idx=idx                     
    )
    # ====================================================


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
    
    # 1. 监控几何精度的 Checkpoint (加了 every_n_epochs=1)
    checkpoint_geo = ModelCheckpoint(
        monitor="geo_loss",
        dirpath=f"checkpoint/R2P_Finetune/{w_tag}/",
        filename=f"best-geo-{w_tag}-" + "{epoch:03d}-{geo_loss:.4f}",
        save_top_k=3,
        mode="min",
        every_n_epochs=1,    # 🌟 显式指定：每一轮结束都进行评估和保存检查
        save_last=True,      # 只需要在其中一个里面设置即可
        save_on_train_epoch_end=True
    )

    # 2. 监控物理能隙 (Gap) 的 Checkpoint (加了 every_n_epochs=1)
    checkpoint_phys = ModelCheckpoint(
        monitor="S1_S0_gap",
        dirpath=f"checkpoint/R2P_Finetune/{w_tag}/",
        filename=f"best-phys-{w_tag}-" + "{epoch:03d}-{S1_S0_gap:.4f}",
        save_top_k=3,
        mode="min",
        every_n_epochs=1,    # 🌟 显式指定：每一轮结束都进行评估和保存检查
        save_on_train_epoch_end=True
    )

    callbacks = [
        checkpoint_geo,
        checkpoint_phys,
        LearningRateMonitor(logging_interval='step'),
        PrintMetricsCallback(),
    ]

    if training_config["ema"]:
        callbacks.append(EMACallback(pl_module=ddpm, decay=training_config["ema_decay"]))

    trainer = Trainer(
        max_epochs=200, # 全量微调建议跑久一点
        accelerator="gpu",
        devices=[0], 
        strategy="auto",
        # 👇【必须加上这一行】：强制使用 32 位浮点数，拒绝 FP16 的精度下溢！
        precision="32-true",
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=training_config["gradient_clip_val"],
        
        # 【关键配置】提高验证效率，每 5 个 Epoch 验证一次
        check_val_every_n_epoch=5,
        
        # === 【关键修改】添加梯度累积 ===
        # 因为 bz 改成了 4，这里累积 8 次，相当于有效 Batch Size = 32
        # 这样既不会爆显存，又能保证梯度的稳定性
        accumulate_grad_batches=8,
        # ============================
    )

    # 4. 开始训练
    print("\n🚀 开始 R->P 任务全量微调...")
    trainer.fit(ddpm)

if __name__ == "__main__":
    main()
