import torch
import numpy as np
import os
from pytorch_lightning import Trainer, seed_everything, Callback
from pytorch_lightning.loggers import CSVLogger # 本地调试用 CSVLogger 更轻量
from reactot.trainer.pl_trainer import SBModule
from reactot.model import LEFTNet

# === 1. 核心配置 (Local Debug Mode) ===
class Config:
    def __init__(self):
        # 基础路径
        self.datadir = "reactot/data_meci/" # 指向你转换好的 .pkl 文件夹
        self.checkpoint_path = "reactot-pretrained.ckpt"
        
        # 调试参数
        self.batch_size = 2       # 极小 Batch
        self.max_epochs = 100     # 跑 100 轮确保过拟合 (Loss -> 0)
        self.nfe = 20             # 采样步数
        self.lr = 1e-4            # 微调学习率

# 优化器参数
class OPT:
    def __init__(self):
        self.solver = "ode"
        self.method = "midpoint"
        self.atol = 1e-3 # 本地调试精度可以稍微放宽
        self.rtol = 1e-3

# === 2. 辅助工具：Kabsch RMSD 计算 ===
def kabsch_rmsd(P, Q):
    """计算两个点集之间的最小 RMSD (自动对齐)"""
    P = P - np.mean(P, axis=0)
    Q = Q - np.mean(Q, axis=0)
    H = np.dot(P.T, Q)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(np.dot(Vt.T, U.T))
    E = np.eye(3)
    if d < 0: E[2, 2] = -1
    R = np.dot(Vt.T, np.dot(E, U.T))
    P_aligned = np.dot(P, R)
    return np.sqrt(np.sum((P_aligned - Q)**2) / len(P))

# === 3. 自定义回调：训练结束后立即测试并打印 ===
class SingleSampleCheckCallback(Callback):
    def on_fit_end(self, trainer, pl_module):
        print("\n\n" + "="*60)
        print("🧪 训练结束，开始单样本生成测试 (Sanity Check)")
        print("="*60)
        
        # 1. 获取一个 Batch 的数据 (从训练集)
        device = pl_module.device
        pl_module.eval()
        
        # 从 dataloader 取出第一批数据
        dataloader = trainer.train_dataloader
        batch = next(iter(dataloader))
        
        # 2. 将数据移动到设备
        # SBModule 的处理逻辑比较复杂，这里模拟 pl_trainer 里的处理
        # 转换 PyG Batch 到模型需要的 list of dict
        # 注意: 这里简化处理，直接调用 pl_module 内部的数据处理逻辑可能较难，
        # 我们直接利用 pl_module.forward 里的解包逻辑
        
        # 为了方便，我们直接手动从 batch 中提取我们需要的信息
        # batch 是一个 PyG Data 对象
        #我们需要构建 representations list
        # 这是一个 hacky 但有效的方法，模拟 dataset.collate 的逆过程
        # 但 SBModule 的输入通常已经是 List[Dict] (如果用自定义 collate) 或者 PyG Batch
        # 让我们直接利用 pl_module.eval_sample_batch 接口，它能处理这些细节
        
        # 将 batch 移到 GPU
        batch = batch.to(device)
        
        # 3. 运行生成 (Inference)
        with torch.no_grad():
            # 条件输入 (在 R->P 任务中通常是全 0 或 dummy)
            conditions = torch.zeros(batch.num_graphs, 1, dtype=torch.long, device=device)
            
            # 调用 React-OT 的评估接口
            # eval_sample_batch 会自动识别 mapping="R->P"
            # 返回: r_pos, p_pred_pos, p_true_pos, ...
            print(f"🚀 正在使用 ODE 求解器 (NFE={pl_module.nfe}) 生成结构...")
            r_pos, _, p_pos, x0_size, x0_other, rmsds = pl_module.eval_sample_batch(
                (batch, conditions),
                return_all=True
            )
            
            # 注意：在 R->P 模式下，eval_sample_batch 的返回值对应关系会发生变化
            # 查看 pl_trainer.py:
            # outputs = self.ddpm.sample(...)
            # 如果是 R->P, xs (轨迹) 的终点是 P_pred
            # 我们直接拿 p_pos (这是模型生成的) 和 batch 中的真实 P 对比
            
            # *修正*: eval_sample_batch 返回的通常是 (Start, End_Pred, End_True)
            # 在 R->P 模式下:
            #   r_pos  -> Start (R)
            #   ts_pos -> End Pred (P_pred)  <-- 变量名虽叫 ts_pos，实际是中间/生成项
            #   p_pos  -> End True (P_true)  <-- 变量名虽叫 p_pos，实际是目标项
            # 让我们通过打印 shape 来确认，或者直接看 RMSD
            
            p_pred = _ # 待确认，暂取中间项
            p_true = _ # 待确认，暂取最后项
            
            # 由于 eval_sample_batch 内部逻辑复杂，我们直接信任它返回的 `rmsds`
            # 它内部已经计算了 生成值 vs 真实值 的 RMSD
            
            final_rmsd = rmsds.mean().item()
            print(f"\n✅ 测试完成！")
            print(f"📉 当前 Batch 平均 RMSD: {final_rmsd:.4f} Å")
            
            # 4. 打印第一个分子的坐标对比
            print("\n📝 --- 单个分子坐标对比 (前5个原子) ---")
            
            # 提取第一个分子的数据
            # r_pos, p_pred, p_true 都是 (N_total, 3) 的 tensor
            # 我们只看第一个分子
            n_atoms = batch.ptr[1] - batch.ptr[0]
            start = batch.ptr[0]
            end = batch.ptr[1]
            
            # 根据 pl_trainer.py 的逻辑:
            # returns: x0_out, x1_out, x0_true ...
            # x0 是终点(P), x1 是起点(R)
            # eval_sample_batch 返回: x1_out, x0_out, x0_true
            # 对应: R, P_pred, P_true
            
            coords_R = r_pos[start:end].cpu().numpy()
            coords_P_pred = _[start:end].cpu().numpy() # 中间那个返回值
            coords_P_true = p_pos[start:end].cpu().numpy()
            
            # 手动再算一次 RMSD 验证
            my_rmsd = kabsch_rmsd(coords_P_pred, coords_P_true)
            print(f"🔍 样本 0 手动计算 RMSD: {my_rmsd:.4f} Å")
            
            atoms = ["C"] * n_atoms # 简化显示
            print(f"{'Atom':<5} {'X(True)':<10} {'X(Pred)':<10} {'Diff':<10}")
            for i in range(min(5, n_atoms)):
                diff = np.linalg.norm(coords_P_pred[i] - coords_P_true[i])
                print(f"{i:<5} {coords_P_true[i][0]:10.4f} {coords_P_pred[i][0]:10.4f} {diff:10.4f}")
            print("..." if n_atoms > 5 else "")

# 重新打补丁：为了让上面的 Callback 能拿到数据，我们需要一种稍微 hacky 的方式
# 因为 eval_sample_batch 的返回值解包比较麻烦，我们修改 Callback 直接打印 RMSD 即可
# 坐标对比我会在下面通过 print 简单展示

# === 4. 权重适配函数 (同之前) ===
def load_and_adapt_checkpoint(ckpt_path):
    print(f"🔧 加载权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    new_state_dict = {}
    for k, v in state_dict.items():
        if "dynamics.encoders" in k:
            if "encoders.0" in k: new_state_dict[k] = v
            elif "encoders.1" in k: continue # 丢弃 TS
            elif "encoders.2" in k: new_state_dict[k.replace("encoders.2", "encoders.1")] = v
        elif "dynamics.decoders" in k:
            if "decoders.0" in k: new_state_dict[k] = v
            elif "decoders.1" in k: continue # 丢弃 TS
            elif "decoders.2" in k: new_state_dict[k.replace("decoders.2", "decoders.1")] = v
        else: new_state_dict[k] = v
    return new_state_dict

# === 5. 主程序 ===
def main():
    seed_everything(42)
    cfg = Config()
    
    # 1. 检查文件
    if not os.path.exists(cfg.datadir):
        print(f"❌ 错误: 数据目录 {cfg.datadir} 不存在！请先运行 convert_npz_to_pkl.py")
        return
    if not os.path.exists(cfg.checkpoint_path):
        print(f"❌ 错误: 权重文件 {cfg.checkpoint_path} 不存在！")
        return

    # 2. 模型配置 (R -> P)
    # 关键点: fragment_names=["R", "P"], idx=1 (预测P), mapping="R->P"
    ddpm = SBModule(
        model_config=dict(
            hidden_channels=196, num_layers=6, in_hidden_channels=8,
            num_radial=96, object_aware=True, legacy=True
        ),
        optimizer_config=dict(lr=cfg.lr),
        training_config=dict(
            datadir=cfg.datadir, bz=cfg.batch_size, num_workers=0, # 0 workers for local debug
            ema=False, swapping_react_prod=False, # 调试关闭 EMA 和 Swap
            sampler_config=dict(mode="node^2", shuffle=True),
            remove_h=False, #LYY
            clip_grad=True,           # <--- 【本次修复】是否裁剪梯度
            lr_schedule_type=None,    # <--- 【预防性修复】避免下一步报 lr 调度器错误
        ),
        node_nfs=[9, 9], fragment_names=["R", "P"],
        pos_dim=3, condition_nf=1, 
        norm_values=(1., 1., 1.), norm_biases=(0., 0., 0.),
        loss_type="l2", pos_only=True,
        # 【关键修复】必须显式指定处理类型为 TS1x，否则默认找 .npz 文件
        process_type="TS1x",
        model=LEFTNet,
        fixed_idx=[0],      # 固定 R(0)
        idx=1,              # 预测 P(1)
        mapping="R->P", 
        mapping_initial="R",
        nfe=cfg.nfe,
        ot_ode=True,
        timesteps=1000
    )
    
    # 3. 加载权重
    state_dict = load_and_adapt_checkpoint(cfg.checkpoint_path)
    ddpm.load_state_dict(state_dict, strict=False)
    ddpm.ddpm.opt = OPT()

    # 4. 训练器 (Debug Mode)
    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices=1,
        overfit_batches=1,      # 【关键】只训练这 1 个 Batch，反复迭代
        check_val_every_n_epoch=10,
        log_every_n_steps=1,    # 每步都记录
        logger=CSVLogger("logs", name="local_debug"),
        callbacks=[SingleSampleCheckCallback()], # 添加我们的测试回调
        enable_checkpointing=False, # 调试模式不保存权重文件
        enable_progress_bar=True,
    )

    print(f"\n🎯 开始本地过拟合测试 (Overfitting 1 Batch)...")
    print(f"   目标: Loss 应该迅速下降，最终 RMSD 应该很小。")
    
    trainer.fit(ddpm)

if __name__ == "__main__":
    main()
