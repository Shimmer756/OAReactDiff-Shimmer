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
    phys_w = 0.01
    w_tag = f"w{phys_w}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = f"checkpoint/R2P_Finetune/{w_tag}/best-geo-w0.01-epoch=104-geo_loss=0.0131.ckpt"
    mace_model_path = "/root/X-MACE_2/meci_energies_forces.model"
    data_path = "reactot/data_meci/valid_rpsb_filtered_30.pkl"
    
    # 🌟 自动生成结果保存目录
    save_dir = f"results_200/evaluation/{w_tag}"
    os.makedirs(save_dir, exist_ok=True)
    
    if not os.path.exists(ckpt_path):
        print(f"❌ 找不到检查点文件: {ckpt_path}，请确认训练是否已开始。")
        return

    # 1. 加载 MACE 和模型
    mace_calc = MACECalculator(model_paths=mace_model_path, device=str(device), n_energies=2, default_dtype='float64')
    
    model = PhysicsInformedSBModule.load_from_checkpoint(ckpt_path, 
            mace_model_path=mace_model_path, 
            phys_weight=phys_w ,
            strict=False).to(device).double()
    
    model.eval()
    
    # 🌟 关键：必须加上这一行，否则模型不知道用什么 solver 采样
    model.ddpm.opt = OPT()
    
    # ==========================================================
    # 🌟 终极物理连线：强行把老板（model）的物理引擎和权重，塞给干活的司机（ddpm）！
    # ==========================================================
    model.ddpm.physics_engine = model.physics_engine
    model.ddpm.phys_weight = model.phys_weight
    # ==========================================================

    # 2. 加载验证集数据
    with open(data_path, 'rb') as f:
        valid_data = pickle.load(f)
    
    reactants = valid_data['reactant']['positions']
    products_true = valid_data['product']['positions']
    all_charges = valid_data['reactant']['charges']
    num_samples = len(reactants)

    print(f"🚀 开始批量评估验证集: {num_samples} 个分子")
    
    results = [] # 存储每个分子的 {rmsd, gap, frame_idx}
    
    # 原子映射 (保持与你单分子脚本一致)
    atom_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}

    for i in range(num_samples):
        r_pos_np = np.array(reactants[i])
        p_pos_true = np.array(products_true[i])
        charges_np = np.array(all_charges[i])
        N = len(charges_np)

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
                # 采样
                sample_out = model.ddpm.sample(x1_full, [repre_R, repre_P], conditions, ot_ode=True, nfe=200, log_count=10)
                #trajectories = sample_out[1].detach().cpu()
                trajectories = sample_out[0].detach().cpu()

            # 寻找物理最优帧
            mol_best_gap = float('inf')
            mol_best_pos = None

            for f in range(trajectories.shape[1]):
                curr_pos = trajectories[:, f, :].numpy()
                atoms = Atoms(numbers=charges_np, positions=curr_pos)
                atoms.calc = mace_calc
                e = atoms.get_potential_energy().flatten()
                gap = abs(e[1] - e[0])
                
                if gap < mol_best_gap:
                    mol_best_gap = gap
                    mol_best_pos = curr_pos

            # 计算 RMSD
            mol_rmsd = calculate_rmsd(mol_best_pos, p_pos_true)
            results.append({'rmsd': mol_rmsd, 'gap': mol_best_gap})
            
            if (i + 1) % 5 == 0:
                print(f"✅ 已完成 {i+1}/{num_samples} | 当前平均 Gap: {np.mean([r['gap'] for r in results]):.4f} eV")

        except Exception as e:
            print(f"❌ 分子 {i} 评估出错: {e}")

    # --- 🌟 3. 输出与保存区 ---
    if len(results) > 0:
        avg_rmsd = np.mean([r['rmsd'] for r in results])
        avg_gap = np.mean([r['gap'] for r in results])
        med_gap = np.median([r['gap'] for r in results])

        print("\n" + "="*50)
        print(f"📊 验证集全样本评估报告 ({w_tag})")
        print("-" * 50)
        print(f"平均 RMSD: {avg_rmsd:.4f} Å")
        print(f"平均能隙 (Mean Gap): {avg_gap:.4f} eV")
        print(f"能隙中位数 (Median Gap): {med_gap:.4f} eV")
        print("="*50)

        # 🌟 核心改进：pkl 文件名也带权重，且存入指定文件夹
        pkl_save_path = os.path.join(save_dir, f"valid_eval_results_{w_tag}.pkl")
        with open(pkl_save_path, "wb") as f:
            pickle.dump(results, f)

        # 🌟 额外福利：自动生成一个易读的文本总结报告
        report_path = os.path.join(save_dir, "summary_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Weight Tag: {w_tag}\nCheckpoint: {ckpt_path}\n")
            f.write(f"Mean Gap: {avg_gap:.6f} eV\nMedian Gap: {med_gap:.6f} eV\n")
            f.write(f"Mean RMSD: {avg_rmsd:.6f} A\n")

        print(f"\n✅ 评估完成！结果已存入目录: {save_dir}")
if __name__ == "__main__":
    main()
