from typing import List, Optional, Tuple
from uuid import uuid4
import os
import shutil
import inspect
import torch

from .pl_trainer import DDPMModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
#from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from oa_reactdiff.trainer.ema import EMACallback
from oa_reactdiff.model import EGNN, LEFTNet


model_type = "leftnet"
version = "0"
project = "OA_Diff_MECI"
# ---EGNNDynamics---
egnn_config = dict(
    in_node_nf=8,  # embedded dim before injecting to egnn
    in_edge_nf=0,
    hidden_nf=256,
    edge_hidden_nf=64,
    act_fn="swish",
    n_layers=9,
    attention=True,
    out_node_nf=None,
    tanh=True,
    coords_range=15.0,
    norm_constant=1.0,
    inv_sublayers=1,
    sin_embedding=True,
    normalization_factor=1.0,
    aggregation_method="mean",
)
leftnet_config = dict(
    pos_require_grad=False,
    cutoff=10.0,
    num_layers=6,
    hidden_channels=196,
    num_radial=96,
    in_hidden_channels=8,
    #in_hidden_channels=14,  # modify for OAReactDiff
    reflect_equiv=True,
    legacy=True,
    update=True,
    pos_grad=False,
    single_layer_output=True,
    object_aware=True,
)

if model_type == "leftnet":
    model_config = leftnet_config
    model = LEFTNet
elif model_type == "egnn":
    model_config = egnn_config
    model = EGNN
else:
    raise KeyError("model type not implemented.")

optimizer_config = dict(
    #lr=2.5e-4,
    lr=1e-4 ,# <--- 建议降低到 1e-4 甚至 5e-5
    betas=[0.9, 0.999],
    weight_decay=0,
    amsgrad=True,
)

T_0 = 200
T_mult = 2
training_config = dict(
    #datadir="./oa_reactdiff/data/transition1x/",
    datadir="./oa_reactdiff/data_meci/", #LYY修改1
    remove_h=False,
    bz=2,
    num_workers=0,
    clip_grad=True,
    gradient_clip_val=0.5,  #LYY修改
    ema=False,
    ema_decay=0.999,
    # [关键] 您的数据只有单向 (R->P)，不需要交换
    swapping_react_prod=False,
    append_frag=False,
    use_by_ind=True,
    reflection=False,
    # [关键] 您的数据是多分子，这里设为 False 以避免过滤掉"非单片段"数据
    single_frag_only=False,
    only_ts=False,
    lr_schedule_type=None,
    lr_schedule_config=dict(
        gamma=0.8,
        step_size=100,
    ),  # step
)
training_data_frac = 1.0

node_nfs: List[int] = [9] * 3  # 3 (pos) + 5 (cat) + 1 (charge)
edge_nf: int = 0  # edge type
condition_nf: int = 1
fragment_names: List[str] = ["R", "TS", "P"]
pos_dim: int = 3
update_pocket_coords: bool = True
condition_time: bool = True
edge_cutoff: Optional[float] = None
loss_type = "l2"
pos_only = True
process_type = "MECI" #LYY修改
enforce_same_encoding = None
scales = [1.0, 0.0, 1.0] #LYY修改
fixed_idx: Optional[List] = None
eval_epochs = 10

# ----Normalizer---
norm_values: Tuple = (1.0, 1.0, 1.0)
norm_biases: Tuple = (0.0, 0.0, 0.0)

# ---Schedule---
noise_schedule: str = "cosine"
timesteps: int = 5000
precision: float = 1e-5

norms = "_".join([str(x) for x in norm_values])
run_name = f"{model_type}-{version}-" + str(uuid4()).split("-")[-1]

seed_everything(42, workers=True)
ddpm = DDPMModule(
    model_config,
    optimizer_config,
    training_config,
    node_nfs,
    edge_nf,
    condition_nf,
    fragment_names,
    pos_dim,
    update_pocket_coords,
    condition_time,
    edge_cutoff,
    norm_values,
    norm_biases,
    noise_schedule,
    timesteps,
    precision,
    loss_type,
    pos_only,
    process_type,
    model,
    enforce_same_encoding,
    scales,
    source=None,
    fixed_idx=fixed_idx,
    eval_epochs=eval_epochs,
)

config = model_config.copy()
config.update(optimizer_config)
config.update(training_config)
trainer = None

wandb_logger = None
'''
if trainer is None or (isinstance(trainer, Trainer) and trainer.is_global_zero):
    wandb_logger = WandbLogger(
        project=project,
        log_model=False,
        name=run_name,
    )
    try:  # Avoid errors for creating wandb instances multiple times
        wandb_logger.experiment.config.update(config)
        wandb_logger.watch(ddpm.ddpm.dynamics, log="all", log_freq=100, log_graph=False)
    except:
        pass
'''
ckpt_path = f"checkpoint/{project}/{run_name}"
earlystopping = EarlyStopping(
    monitor="val-totloss",
    patience=2000,
    verbose=True,
    log_rank_zero_only=True,
)

#回调=======================================
checkpoint_callback = ModelCheckpoint(
    monitor="val-totloss",
    dirpath=ckpt_path,
    filename="ddpm-{epoch:03d}-{val-totloss:.2f}",
    every_n_epochs=1,
    # --- 核心修改 ---
    save_top_k=1,
    mode="min",             # 明确指出：损失越小(min)越好
    save_last=True,         # (可选建议) 额外保存最新的一个模型 (last.ckpt)，防止中断后没法续传
)

# 2. [新增] 保存 RMSD 最小的模型
checkpoint_callback_rmsd = ModelCheckpoint(
    monitor="val-rmsd",
    dirpath=ckpt_path,
    filename="best-rmsd-{epoch:03d}-{val-rmsd:.4f}",   # 文件名带上 rmsd 值
    save_top_k=1,
    mode="min",
    # 注意: 虽然 RMSD 每 10 个 epoch 才算一次 (其他时候是 NaN)，
    # Lightning 会自动处理: 只有当 metric 是有效数值且更优时才会保存。
)

# 3. [新增] 保存 RMSD 中位数最小的模型
checkpoint_callback_median = ModelCheckpoint(
    monitor="val-rmsd-median",
    dirpath=ckpt_path,
    filename="best-median-{epoch:03d}-{val-rmsd-median:.4f}",
    save_top_k=1,
    mode="min",
)

lr_monitor = LearningRateMonitor(logging_interval="step")
callbacks = [earlystopping, 
        checkpoint_callback,
        checkpoint_callback_rmsd,  # RMSD 最佳
        checkpoint_callback_median,  # 中位数最佳
        TQDMProgressBar(), lr_monitor]
if training_config["ema"]:
    callbacks.append(EMACallback(decay=training_config["ema_decay"]))

if not os.path.isdir(ckpt_path):
    os.makedirs(ckpt_path)
# 获取 LEFTNet 类所在文件的真实路径
model_file_path = inspect.getfile(model) # model 变量是 LEFTNet 类
shutil.copy(model_file_path, f"{ckpt_path}/{model_type}.py")

print("config: ", config)

strategy = None
devices = [0]
strategy = DDPStrategy(find_unused_parameters=True)
if strategy is not None:
    devices = list(range(torch.cuda.device_count()))
if len(devices) == 1:
    strategy = "auto"
trainer = Trainer(
    max_epochs=2000,
    accelerator="gpu",
    precision="bf16-mixed", #混合精度,LYY修改
    deterministic=False,
    devices=devices,
    strategy=strategy,
    log_every_n_steps=1,
    callbacks=callbacks,
    profiler=None,
    logger=None,
    #logger=wandb_logger,
    # [修改这里] 累积 8 次 update 再更新一次权重,# 2 (bz) * 8 (accum) = 16 (Effective Batch Size)
    accumulate_grad_batches=8,
    gradient_clip_val=training_config["gradient_clip_val"],
    limit_train_batches=200,
    limit_val_batches=20,
    # max_time="00:10:00:00",
)

trainer.fit(ddpm)
trainer.save_checkpoint("pretrained-ts1x-diff.ckpt")
#trainer.save_checkpoint("custom_r2p_model.ckpt")
