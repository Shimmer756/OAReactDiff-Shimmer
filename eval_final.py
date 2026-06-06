import torch
import numpy as np
import pickle
import os
from ase import Atoms
from mace.calculators import MACECalculator
from scipy.spatial.transform import Rotation
from reactot.trainer.train_rp_finetune import PhysicsInformedSBModule

class OPT:
    def __init__(self):
        self.solver = "ddpm"
        self.method = "midpoint"
        self.atol = 1e-2
        self.rtol = 1e-2

def calculate_rmsd(pos1, pos2):
    p1_centered = pos1 - np.mean(pos1, axis=0)
    p2_centered = pos2 - np.mean(pos2, axis=0)
    rotation, _ = Rotation.align_vectors(p2_centered, p1_centered)
    p1_aligned = rotation.apply(p1_centered)
    return np.sqrt(np.mean(np.sum((p1_aligned - p2_centered) ** 2, axis=1)))

def main():
    # =====================================================================
    # 🌟 1. 核心参数配置区 (在这里修改你的 w 和 ckpt)
    # =====================================================================
    phys_w = 0.2  # 物理权重
    w_tag = f"w{phys_w}"
    nfe_steps = 50 # 指定采样步数

    #    请将此处替换为你真实想要测试的 ckpt 路径
    ckpt_path = f"checkpoint/R2P_Finetune/{w_tag}/best-geo-w0.2-epoch=104-geo_loss=0.0154.ckpt"
    mace_model_path = "/root/X-MACE_2/meci_energies_forces.model"
    data_path = "reactot/data_meci/valid_rpsb_filtered_30.pkl"

    # 创建独立的保存文件夹
    save_dir = f"results_detailed_2/{w_tag}_nfe{nfe_steps}"
    os.makedirs(save_dir, exist_ok=True)
    # =====================================================================

    if not os.path.exists(ckpt_path):
        print(f"❌ 找不到检查点文件: {ckpt_path}，请确认路径。")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载 MACE 和模型
    mace_calc = MACECalculator(model_paths=mace_model_path, device=str(device), n_energies=2, default_dtype='float64')

    model = PhysicsInformedSBModule.load_from_checkpoint(ckpt_path,
            mace_model_path=mace_model_path,
            phys_weight=phys_w ,
            strict=False).to(device).double()

    model.eval()
    model.ddpm.opt = OPT()
    model.ddpm.physics_engine = model.physics_engine
    model.ddpm.phys_weight = model.phys_weight

    # 加载验证集数据
    with open(data_path, 'rb') as f:
        valid_data = pickle.load(f)

    reactants = valid_data['reactant']['positions']
    products_true = valid_data['product']['positions']
    all_charges = valid_data['reactant']['charges']
    num_samples = len(reactants)

    print(f"🚀 开始批量深度评估验证集: {num_samples} 个分子 (NFE={nfe_steps})")

    # 🌟 2. 初始化总的统计文件表头
    summary_file_path = os.path.join(save_dir, "summary_all_molecules.txt")
    with open(summary_file_path, "w") as f_sum:
        f_sum.write("Mol_Index\tR-P_RMSD(A)\tBest-R_RMSD(A)\tBest-P_RMSD(A)\tBest_Gap(eV)\tS0_Energy(eV)\tS1_Energy(eV)\n")

    atom_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}

    for i in range(num_samples):
        r_pos_np = np.array(reactants[i])
        p_pos_true = np.array(products_true[i])
        charges_np = np.array(all_charges[i])
        N = len(charges_np)

        # 🌟 计算分子真实的 起点(R) 与 终点(P) 之间的 RMSD，作为位移参考基准
        rmsd_R_P = calculate_rmsd(r_pos_np, p_pos_true)

        # 构造输入
        z_indices = torch.tensor([atom_map.get(int(c), 0) for c in charges_np], device=device)
        z_onehot = torch.nn.functional.one_hot(z_indices, num_classes=5).double()
        z_charge = torch.tensor(charges_np, dtype=torch.float64, device=device).unsqueeze(-1)
        batch_idx = torch.zeros(N, dtype=torch.long, device=device)

        repre_template = {"one_hot": z_onehot, "charge": z_charge, "size": torch.tensor([N], dtype=torch.long, device=device), "mask": batch_idx, "batch": batch_idx}
        repre_R = {**repre_template, "pos": torch.tensor(r_pos_np, dtype=torch.float64, device=device)}
        repre_P = {**repre_template, "pos": torch.tensor(p_pos_true, dtype=torch.float64, device=device)}

        conditions = torch.zeros(1, 1, dtype=torch.float64, device=device)
        x1_full = torch.tensor(r_pos_np, dtype=torch.float64, device=device).unsqueeze(0).reshape(1, -1)

        try:
            with torch.no_grad():
                # 采样，固定 NFE 为你指定的步数
                sample_out = model.ddpm.sample(x1_full, [repre_R, repre_P], conditions, ot_ode=True, nfe=nfe_steps, log_count=nfe_steps)
                trajectories = sample_out[0].detach().cpu()

            mol_best_gap = float('inf')
            mol_best_pos = None
            mol_best_S0 = None
            mol_best_S1 = None

            # 🌟 3. 准备记录该分子每一帧的数据
            frame_records = ["Frame\tGap(eV)\tS0(eV)\tS1(eV)\tRMSD_to_R(A)\tRMSD_to_P(A)\n"]

            # =====================================================================
            # 遍历并评估全部帧 (trajectories.shape[1] 即你设置的全部步数)
            # =====================================================================
            for f in range(trajectories.shape[1]):
                curr_pos = trajectories[:, f, :].numpy()

                # 计算能量
                atoms = Atoms(numbers=charges_np, positions=curr_pos)
                atoms.calc = mace_calc
                e = atoms.get_potential_energy().flatten()
                curr_s0, curr_s1 = e[0], e[1]
                gap = abs(curr_s1 - curr_s0)

                # 计算该帧构型到 R 和 P 的距离
                curr_rmsd_to_R = calculate_rmsd(curr_pos, r_pos_np)
                curr_rmsd_to_P = calculate_rmsd(curr_pos, p_pos_true)

                # 无条件追加记录当前帧数据，保留全部 50 帧
                frame_records.append(f"{f}\t{gap:.6f}\t{curr_s0:.6f}\t{curr_s1:.6f}\t{curr_rmsd_to_R:.6f}\t{curr_rmsd_to_P:.6f}\n")

                # 更新最佳结构（仅用于宏观统计总表，不影响帧数据的全量保存）
                if gap < mol_best_gap:
                    mol_best_gap = gap
                    mol_best_pos = curr_pos
                    mol_best_S0 = curr_s0
                    mol_best_S1 = curr_s1

            # 🌟 4. 提取最佳结构的 RMSD 数据 (写入宏观表使用)
            best_rmsd_to_R = calculate_rmsd(mol_best_pos, r_pos_np)
            best_rmsd_to_P = calculate_rmsd(mol_best_pos, p_pos_true)

            # 将最优帧总结果追加写入到宏观表中
            with open(summary_file_path, "a") as f_sum:
                f_sum.write(f"{i}\t{rmsd_R_P:.6f}\t{best_rmsd_to_R:.6f}\t{best_rmsd_to_P:.6f}\t{mol_best_gap:.6f}\t{mol_best_S0:.6f}\t{mol_best_S1:.6f}\n")

            # 将全部的帧过程写入单独的分子文件中（没有任何截断）
            mol_frame_path = os.path.join(save_dir, f"mol_{i}_frames.txt")
            with open(mol_frame_path, "w") as f_frame:
                f_frame.writelines(frame_records)

            if (i + 1) % 5 == 0:
                print(f"✅ 已完成 {i+1}/{num_samples} | 当前分子最优Gap: {mol_best_gap:.4f} eV")

        except Exception as e:
            print(f"❌ 分子 {i} 评估出错: {e}")

    print(f"\n🎉 评估完成！所有数据已存入: {save_dir}")
    print(f"   - 总览表: summary_all_molecules.txt")
    print(f"   - 帧数据: mol_*_frames.txt (已包含完整的 {nfe_steps} 帧)")

if __name__ == "__main__":
    main()
