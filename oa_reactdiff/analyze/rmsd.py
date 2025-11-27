from typing import List
import numpy as np
import math

from pymatgen.core import Molecule
from pymatgen.analysis.molecule_matcher import (
    BruteForceOrderMatcher,
    GeneticOrderMatcher,
    HungarianOrderMatcher,
    KabschMatcher,
)
from pymatgen.io.xyz import XYZ

import torch
from torch import Tensor


def xh2pmg(xh):
    mol = Molecule(
        species=xh[:, -1].long().cpu().numpy(),
        coords=xh[:, :3].cpu().numpy(),
    )
    return mol


def xyz2pmg(xyzfile):
    xyz_converter = XYZ(mol=None)
    mol = xyz_converter.from_file(xyzfile).molecule
    return mol


def rmsd_core(mol1, mol2, threshold=0.5, same_order=False):
    _, count = np.unique(mol1.atomic_numbers, return_counts=True)
    if same_order:
        bfm = KabschMatcher(mol1)
        _, rmsd = bfm.fit(mol2)
        return rmsd
    total_permutations = 1
    for c in count:
        total_permutations *= math.factorial(c)  # type: ignore
    if total_permutations < 1e4:
        bfm = BruteForceOrderMatcher(mol1)
        _, rmsd = bfm.fit(mol2)
    else:
        bfm = GeneticOrderMatcher(mol1, threshold=threshold)
        pairs = bfm.fit(mol2)
        rmsd = threshold
        for pair in pairs:
            rmsd = min(rmsd, pair[-1])
        if not len(pairs):
            bfm = HungarianOrderMatcher(mol1)
            _, rmsd = bfm.fit(mol2)
    return rmsd


def pymatgen_rmsd(
    mol1,
    mol2,
    ignore_chirality=False,
    threshold=0.5,
    same_order=False,
):
    if isinstance(mol1, str):
        mol1 = xyz2pmg(mol1)
    if isinstance(mol2, str):
        mol2 = xyz2pmg(mol2)
    rmsd = rmsd_core(mol1, mol2, threshold)
    if ignore_chirality:
        coords = mol2.cart_coords
        coords[:, -1] = -coords[:, -1]
        mol2_reflect = Molecule(
            species=mol2.species,
            coords=coords,
        )
        rmsd_reflect = rmsd_core(mol1, mol2_reflect, threshold)
        rmsd = min(rmsd, rmsd_reflect)
    return rmsd


def batch_rmsd(
    fragments_nodes: List[Tensor],
    out_samples: List[Tensor],
    xh: List[Tensor],
    idx: int = 1,
    threshold=0.5,
):
    rmsds = []
    out_samples_use = out_samples[idx]
    xh_use = xh[idx]
    nodes = fragments_nodes[idx].long().cpu().numpy()
    start_ind, end_ind = 0, 0
    for jj, natoms in enumerate(nodes):
        end_ind += natoms

        # 提取当前分子的数据----------------------------------------------------
        current_gen_xh = out_samples_use[start_ind:end_ind]
        current_true_xh = xh_use[start_ind:end_ind]

        # --- [DEBUG] 1. 检查数值有效性 ---
        gen_coords = current_gen_xh[:, :3]
        if torch.isnan(gen_coords).any() or torch.isinf(gen_coords).any():
            print(f"\n[RMSD ERROR] 分子 {jj}: 生成的坐标包含 NaN 或 Inf！")
            print(f"  - 坐标片段:\n{gen_coords[:5]}")
            rmsds.append(1.0) # 标记为失败
            start_ind = end_ind
            continue

        max_val = torch.max(torch.abs(gen_coords)).item()
        if max_val > 100.0: # 阈值设为 100 (正常分子通常 < 10)
            print(f"\n[RMSD WARNING] 分子 {jj}: 坐标数值爆炸！最大绝对值: {max_val:.2f}")
            print(f"  - 坐标片段:\n{gen_coords[:5]}")
            # 这里不跳过，尝试让它报错看看具体的数学错误

        mol1 = xh2pmg(out_samples_use[start_ind:end_ind])
        mol2 = xh2pmg(xh_use[start_ind:end_ind])
        try:
            rmsd = pymatgen_rmsd(mol1, mol2, ignore_chirality=True, threshold=threshold)
            rmsds.append(min(rmsd, 1.0)) # 正常截断
        #except:
        #    rmsd = 1.0
        #rmsds.append(min(rmsd, 1.0))
        except Exception as e:
            # --- [DEBUG] 2. 捕获并报告具体的计算错误 ---
            print(f"\n[RMSD CRASH] 分子 {jj} 计算失败！")
            print(f"  - 错误类型: {type(e).__name__}")
            print(f"  - 错误信息: {e}")
            print(f"  - 此时的最大坐标值: {max_val:.2f}")
            # print(f"  - 导致崩溃的坐标:\n{gen_coords.cpu().numpy()}")
            
            rmsds.append(1.0) # 保持 1.0 以防止训练中断，但在日志里你会看到上面的错误
        start_ind = end_ind
    return rmsds
