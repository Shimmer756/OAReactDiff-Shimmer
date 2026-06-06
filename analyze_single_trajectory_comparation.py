import torch
import numpy as np
import pickle
import os
import csv
from ase import Atoms
from ase.io import write
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

# 🌟 新增功能函数：用于生成两帧对比的 XYZ 文件
def save_comparison_xyz(pos1, pos2, charges, filepath):
    atoms1 = Atoms(numbers=charges, positions=pos1)
    atoms2 = Atoms(numbers=charges, positions=pos2)
    # 将两个构型作为一个列表写入，生成包含两帧的 xyz 文件
    write(filepath, [atoms1, atoms2])

def main():
    # ==========================================
    # 🎯 核心配置区
    # ==========================================
    target_index = 41  # 分析的分子索引
    phys_w = 0.1      # 权重
    w_tag = f"w{phys_w}"
    nfe_steps = 200     # 采样步数

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = f"checkpoint/R2P_Finetune/{w_tag}/best-geo-w0.1-epoch=104-geo_loss=0.0146.ckpt"
    mace_model_path = "/root/X-MACE_2/meci_energies_forces.model"
    data_path = "reactot/data_meci/valid_rpsb_filtered_30.pkl"

    # 自动生成专属的保存目录
    save_dir = f"results/trajectory_analysis/{w_tag}_mol_{target_index}"
    os.makedirs(save_dir, exist_ok=True)
    
    xyz_path = os.path.join(save_dir, f"trajectory_mol_{target_index}.xyz")
    csv_path = os.path.join(save_dir, f"metrics_mol_{target_index}.csv")
    
    # 🌟 定义新增的对比文件保存路径
    start_vs_final_path = os.path.join(save_dir, f"compare_start_vs_final_mol_{target_index}.xyz")
    true_vs_final_path = os.path.join(save_dir, f"compare_true_vs_final_mol_{target_index}.xyz")

    if os.path.exists(xyz_path): os.remove(xyz_path)
    if os.path.exists(start_vs_final_path): os.remove(start_vs_final_path)
    if os.path.exists(true_vs_final_path): os.remove(true_vs_final_path)

    print(f"🔧 正在加载 MACE 和 扩散模型 (权重: {w_tag})...")
    mace_calc = MACECalculator(model_paths=mace_model_path, device=str(device), n_energies=2, default_dtype='float64')

    model = PhysicsInformedSBModule.load_from_checkpoint(ckpt_path,
            mace_model_path=mace_model_path,
            phys_weight=phys_w,
            strict=False).to(device).double()
    model.eval()
    model.ddpm.opt = OPT()
    

    # ==========================================================
    # 🌟 终极物理连线：强行把老板（model）的物理引擎和权重，塞给干活的司机（ddpm）！
    # ==========================================================
    model.ddpm.physics_engine = model.physics_engine
    model.ddpm.phys_weight = model.phys_weight
    # ==========================================================


    with open(data_path, 'rb') as f:
        valid_data = pickle.load(f)

    reactants = valid_data['reactant']['positions']
    products_true = valid_data['product']['positions']
    all_charges = valid_data['reactant']['charges']

    print(f"🚀 开始单分子逐帧轨迹提取: Index = {target_index}")
    r_pos_np = np.array(reactants[target_index])
    p_pos_true = np.array(products_true[target_index])
    charges_np = np.array(all_charges[target_index])
    N = len(charges_np)

    atom_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}

    z_indices = torch.tensor([atom_map.get(int(c), 0) for c in charges_np], device=device)
    z_onehot = torch.nn.functional.one_hot(z_indices, num_classes=5).double()
    z_charge = torch.tensor(charges_np, dtype=torch.float64, device=device).unsqueeze(-1)
    batch_idx = torch.zeros(N, dtype=torch.long, device=device)

    repre_template = {"one_hot": z_onehot, "charge": z_charge, "size": torch.tensor([N], dtype=torch.long, device=device), "mask": batch_idx, "batch": batch_idx}
    repre_R = {**repre_template, "pos": torch.tensor(r_pos_np, dtype=torch.float64, device=device)}
    repre_P = {**repre_template, "pos": torch.tensor(p_pos_true, dtype=torch.float64, device=device)}

    conditions = torch.zeros(1, 1, dtype=torch.float64, device=device)
    x1_full = torch.tensor(r_pos_np, dtype=torch.float64, device=device).unsqueeze(0).reshape(1, -1)

    trajectory_metrics = []

    try:
        with torch.no_grad():
            sample_out = model.ddpm.sample(x1_full, [repre_R, repre_P], conditions,
                                           ot_ode=True, nfe=nfe_steps, log_count=nfe_steps)
            trajectories = sample_out[0].detach().cpu()

        print(f"✅ 采样完成，共生成 {trajectories.shape[1]} 帧轨迹。开始计算物理/几何指标...")

        # 🌟 新增：追踪最优帧（最小 Gap）的变量
        best_frame_idx = -1
        min_gap = float('inf')
        best_frame_pos = None

        best_frame_idx = 0
        best_frame_pos = trajectories[:, 0, :].numpy()

        for f in range(trajectories.shape[1]):
            curr_pos = trajectories[:, f, :].numpy()
            
            # ==========================================
            # 🛡️ 提前给默认惩罚值，确保变量绝对存在！
            # ==========================================
            gap = 999.0
            frame_rmsd = 999.0 

            try:
                # 1. 算 RMSD (纯数学计算，绝对安全)
                frame_rmsd = calculate_rmsd(curr_pos, p_pos_true)

                # 2. 组装 ASE 对象并写入
                atoms = Atoms(numbers=charges_np, positions=curr_pos)
                write(xyz_path, atoms, append=True)
                atoms.calc = mace_calc
                
                # 3. 算能量 (极容易爆炸的雷区)
                e = atoms.get_potential_energy().flatten()
                gap = abs(e[1] - e[0])
                
                # 4. 如果活过了雷区，就去竞争最优帧
                if gap < min_gap:
                    min_gap = gap
                    best_frame_idx = f
                    best_frame_pos = curr_pos

                print(f"  帧 {f:02d} | 伪时间 t={f / (trajectories.shape[1] - 1):.2f} | Gap: {gap:.4f} eV | RMSD: {frame_rmsd:.4f} Å")
            
            except Exception as e_err:
                print(f"  ⚠️ 帧 {f:02d} | 评估爆炸 (极可能原子重叠)！已跳过该帧。")
            
            # ==========================================
            # 现在不管上面怎么崩，gap 和 frame_rmsd 都绝对有值！

            trajectory_metrics.append({
                'frame': f,
                'time_t': f / (trajectories.shape[1] - 1), 
                'gap_eV': gap,
                'rmsd_A': frame_rmsd
            })
            print(f"  帧 {f:02d} | 伪时间 t={trajectory_metrics[-1]['time_t']:.2f} | Gap: {gap:.4f} eV | RMSD: {frame_rmsd:.4f} Å")

        # 写入 CSV... (这部分保持不变)
        with open(csv_path, mode='w', newline='') as csv_file:
            # ... (保持原样)
            fieldnames = ['frame', 'time_t', 'gap_eV', 'rmsd_A']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in trajectory_metrics:
                writer.writerow(row)

        # ==========================================================
        # 🌟 新增功能：计算最优帧与 R (起点) 和 P (真值) 的 RMSD
        # ==========================================================
        if best_frame_pos is not None:
            rmsd_to_R = calculate_rmsd(best_frame_pos, r_pos_np)
            rmsd_to_P = calculate_rmsd(best_frame_pos, p_pos_true)
        else:
            rmsd_to_R, rmsd_to_P = 999.0, 999.0 # 防止全炸的情况报错

        # 🌟 核心修正：使用 best_frame_pos 生成对比文件，而不是最后一帧
        print(f"\n🎯 轨迹搜索完成！锁定最优帧：第 {best_frame_idx} 帧，最低 Gap = {min_gap:.4f} eV")
        print(f"📏 最优结构 (MECI) 几何偏差分析:")
        print(f"   ▶ 与反应物 (R) 的 RMSD: {rmsd_to_R:.4f} Å")
        print(f"   ▶ 与真值态 (P) 的 RMSD: {rmsd_to_P:.4f} Å")
        
        print("\n🔧 正在生成双态对比文件 (基于最优帧)...")
        save_comparison_xyz(r_pos_np, best_frame_pos, charges_np, start_vs_final_path)
        save_comparison_xyz(p_pos_true, best_frame_pos, charges_np, true_vs_final_path)

        print("\n" + "="*50)
        print(f"🎉 分子 {target_index} 轨迹及对比文件提取成功！")
        print(f"1. 完整 50 帧演化: {os.path.basename(xyz_path)}")
        print(f"2. [起点 vs 终点]: {os.path.basename(start_vs_final_path)}")
        print(f"3. [真值 vs 终点]: {os.path.basename(true_vs_final_path)}")
        print("="*50)

    except Exception as e:
        print(f"❌ 评估出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
