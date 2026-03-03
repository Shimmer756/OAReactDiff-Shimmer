import torch
from torch import nn
from torch.autograd import grad
from ase import Atoms
import numpy as np

# 导入 MACE 核心数据处理模块
from mace.data.atomic_data import AtomicData
from mace.data.utils import config_from_atoms
from mace.tools.torch_geometric.batch import Batch as MaceBatch
# 【新增】：导入 utils 用于生成 z_table
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

        # 【核心修复】：手动根据模型的 atomic_numbers 生成 z_table 和提取 r_max
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
                z_table=self.z_table,  # 使用我们刚刚生成的 z_table
                cutoff=self.r_max      # 使用提取的 r_max
            )
            mace_data_list.append(atomic_data)

        mace_batch = MaceBatch.from_data_list(mace_data_list).to(self.device)
        # 【关键】：保留 PyTorch 梯度
        mace_batch.positions = input_dict["pos"] 
        return mace_batch.to_dict()

    @torch.enable_grad()
    def forward_autograd(self, input_dict, conditions=None):
        """预测能量，并算出引导 MECI 的力"""
        input_dict["pos"].requires_grad_(True)
        
        mace_dict = self._build_mace_batch(input_dict)
        out = self.mace_model(mace_dict)
        
        # 从字典中提取能量
        energies = out['energy']
        e_s0 = energies[:, 0]
        e_s1 = energies[:, 1]

        # 惩罚项计算
        gap_penalty = self.alpha * torch.pow(e_s1 - e_s0, 2)
        min_penalty = self.beta * e_s1
        ae = gap_penalty + min_penalty

        forces = -grad(
            outputs=torch.sum(ae),
            inputs=input_dict["pos"],
            create_graph=self.training
        )[0]
        
        # 【新增】：计算一下批次的平均能量差，方便监控
        gap_mean = torch.mean(torch.abs(e_s1 - e_s0))

        return ae.squeeze(), forces, gap_mean

    # 【我们新增的救命函数：只算能量，绝对不碰二阶导数】
    def forward_energy_only(self, input_dict, conditions=None):
        """只预测能量和Gap，避免显式计算受力导致的二阶导数爆炸"""
        # 直接构建 MACE 数据，因为传入的 pos 已经自带扩散模型的梯度，不需要再 requires_grad_
        mace_dict = self._build_mace_batch(input_dict)
        out = self.mace_model(mace_dict)

        # 从字典中提取能量
        energies = out['energy']
        e_s0 = energies[:, 0]
        e_s1 = energies[:, 1]

        # 惩罚项计算 (这个就是我们需要的物理 Loss)
        gap_penalty = self.alpha * torch.pow(e_s1 - e_s0, 2)
        min_penalty = self.beta * e_s1
        ae = gap_penalty + min_penalty

        # 计算平均能量差，方便监控
        gap_mean = torch.mean(torch.abs(e_s1 - e_s0))

        # 中间的 force 直接返回 None，不计算！
        return ae.squeeze(), None, gap_mean
