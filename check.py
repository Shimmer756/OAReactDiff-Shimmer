import torch
from reactot.trainer.pl_trainer import DDPMModule

# 加载你的权重（只加载扩散模型部分）
print("正在对模型大脑进行扫描...")
ckpt_path = "checkpoint/R2P_Finetune/last-v2.ckpt" 
#ckpt_path = "reactot-pretrained.ckpt"
model = DDPMModule.load_from_checkpoint(ckpt_path, map_location="cpu", strict=False)

nan_found = False
for name, param in model.named_parameters():
    # 扫描每一个权重矩阵
    if torch.isnan(param).any():
        print(f"💀 发现毒药！层的名字叫: {name}，里面充满了 NaN！")
        nan_found = True
        break

if nan_found:
    print("\n❌ 结论：模型在训练时已经梯度爆炸死掉了！不用再拿它去生成了。")
    print("👉 凶手大概率是：学习率（LR）开得太大，或者 MACE 的物理 Loss 传回了有毒的梯度！")
else:
    print("\n✅ 结论：模型权重非常健康！没有任何 NaN！")
    print("👉 说明问题完全出在采样（Sampling）时的初始条件设置上，我们需要修改采样代码！")
