import torch
from torch import nn
from torch.autograd import grad
from ase import Atoms
import numpy as np

from mace.data.atomic_data import AtomicData
from mace.data.utils import config_from_atoms
from mace.tools.torch_geometric.batch import Batch as MaceBatch

from mace.tools import utils 

class XMACEPotential(nn.Module):
    def __init__(
        self,
        mace_model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        alpha: float = 10.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.alpha = alpha
        self.beta = beta
        
        print(f"🔧 正在加载 X-MACE 原生 PyTorch 模型用于梯度追踪: {mace_model_path}")
        self.mace_model = torch.load(mace_model_path, map_location=self.device, weights_only=False)
        self.mace_model.eval()
        for param in self.mace_model.parameters():
            param.requires_grad = False

        self.z_table = utils.AtomicNumberTable([int(z) for z in self.mace_model.atomic_numbers])
        self.r_max = float(self.mace_model.r_max.cpu())

    def _build_mace_batch(self, input_dict):
        """将 React-OT 的 dict 数据无缝转换为 MACE 数据"""
        batch_idx = input_dict["mask"]
        pos = input_dict["pos"]
        z = input_dict["charge"].squeeze()

        n_graphs = int(batch_idx.max()) + 1
        mace_data_list = []

        for i in range(n_graphs):
            mask = (batch_idx == i)
            atoms = Atoms(
                numbers=z[mask].detach().cpu().numpy(),
                positions=pos[mask].detach().cpu().numpy()
            )
            config = config_from_atoms(atoms)
            atomic_data = AtomicData.from_config(
                config, 
                z_table=self.z_table,
                cutoff=self.r_max      
            )
            mace_data_list.append(atomic_data)

        mace_batch = MaceBatch.from_data_list(mace_data_list).to(self.device)
        mace_batch.positions = input_dict["pos"] 
        return mace_batch.to_dict()

    @torch.enable_grad()
    def forward_autograd(self, input_dict, conditions=None):
        print("🚨 警报：底层正在调用实时物理梯度！")
        input_dict["pos"].requires_grad_(True)
        
        mace_dict = self._build_mace_batch(input_dict)
        out = self.mace_model(mace_dict)
        
        energies = out['energy']
        e_s0 = energies[:, 0]
        e_s1 = energies[:, 1]

        gap_penalty = self.alpha * torch.pow(e_s1 - e_s0, 2)
        min_penalty = self.beta * e_s1
        '''
        ae = gap_penalty + min_penalty

        forces = -grad(
            outputs=torch.sum(ae),
            inputs=input_dict["pos"],
            create_graph=self.training
        )[0]
        
        gap_mean = torch.mean(torch.abs(e_s1 - e_s0))
        '''

       # 1. 计算 Gap 梯度 (目标：垂直走向交叉缝合面)
       
        # retain_graph=True 确保计算图在第一次反向传播后不被销毁
        grad_gap = grad(
            outputs=torch.sum(gap_penalty),
            inputs=input_dict["pos"],
            create_graph=self.training,
            retain_graph=True 
        )[0]

        # 2. 计算 S1 极小化梯度 (目标：顺着势能面向下走)
        grad_s1 = grad(
            outputs=torch.sum(min_penalty),
            inputs=input_dict["pos"],
            create_graph=self.training
        )[0]

        # 3. 梯度正交化 (Gram-Schmidt Projection)
        # 从 grad_s1 中剔除掉所有与 grad_gap 平行的分量，防止它们打架
        dot_product = torch.sum(grad_s1 * grad_gap)
        norm_sq = torch.sum(grad_gap * grad_gap) + 1e-8
        grad_s1_orthogonal = grad_s1 - (dot_product / norm_sq) * grad_gap

        # 4. 向量合成 (负号代表梯度的反方向，即力的方向)
        forces = - (grad_gap + grad_s1_orthogonal)

        # 5. 提取标量供日志记录
        ae = gap_penalty + min_penalty
        gap_mean = torch.mean(torch.abs(e_s1 - e_s0))  

        try:
            # 取平均向量长度 (Norm) 来代表这股力的平均强度
            mag_gap = torch.norm(grad_gap, dim=-1).mean().item()
            mag_s1 = torch.norm(grad_s1, dim=-1).mean().item()
            mag_ortho = torch.norm(grad_s1_orthogonal, dim=-1).mean().item()
            mag_final = torch.norm(forces, dim=-1).mean().item()
            
            # 计算投影系数
            proj_coef = (dot_product / norm_sq).item()
            
            print(f"\n--- 🔬 物理引擎底层监控 ---")
            print(f"⚡ S0能量: {e_s0.mean().item():.3f} eV | S1能量: {e_s1.mean().item():.3f} eV | Gap: {gap_mean.item():.3f} eV")
            print(f"🧲 引力对决 (平均梯度长度):")
            print(f"   ▶ Gap 闭合引力 (grad_gap):    {mag_gap:.6f}")
            print(f"   ▶ S1  下降引力 (grad_s1):     {mag_s1:.6f}")
            print(f"📐 正交化手术:")
            print(f"   ▶ 投影系数 (重合度):          {proj_coef:.6f}")
            print(f"   ▶ 切除平行分量后的 S1 引力:   {mag_ortho:.6f}")
            print(f"🚀 最终输出给 ODE 的物理推力:    {mag_final:.6f}")
            print(f"---------------------------")
        except Exception as e:
            print(f"监控打印出错: {e}")
        # ====================================================

        return ae.squeeze(), forces, gap_mean

    def forward_energy_only(self, input_dict, conditions=None):
        """只预测能量和Gap，避免显式计算受力导致的二阶导数爆炸"""
        mace_dict = self._build_mace_batch(input_dict)
        out = self.mace_model(mace_dict)

        energies = out['energy']
        e_s0 = energies[:, 0]
        e_s1 = energies[:, 1]

        gap_penalty = self.alpha * torch.pow(e_s1 - e_s0, 2)
        min_penalty = self.beta * e_s1
        ae = gap_penalty + min_penalty

        gap_mean = torch.mean(torch.abs(e_s1 - e_s0))

        return ae.squeeze(), None, gap_mean
