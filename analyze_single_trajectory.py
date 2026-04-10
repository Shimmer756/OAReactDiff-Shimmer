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

def main():
    # ==========================================
    # 🎯 核心配置区
    # ==========================================
    target_index = 32  # 🌟 在这里填入你要分析的分子索引 (例如最优的 32)
    phys_w = 0.2       # 使用你表现最好的 w=0.2 权重
    w_tag = f"w{phys_w}"
    nfe_steps = 50     # 采样步数

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ⚠️ 请确保这里的 ckpt 路径是你 w=0.2 的真实路径
    ckpt_path = f"checkpoint/R2P_Finetune/{w_tag}/best-geo-w0.2-epoch=104-geo_loss=0.0154.ckpt" 
    mace_model_path = "/root/X-MACE_2/meci_energies_forces.model"
    data_path = "reactot/data_meci/valid_rpsb_filtered_30.pkl"

    # 自动生成专属的保存目录
    save_dir = f"results/trajectory_analysis/{w_tag}_mol_{target_index}"
    os.makedirs(save_dir, exist_ok=True)
    xyz_path = os.path.join(save_dir, f"trajectory_mol_{target_index}.xyz")
    csv_path = os.path.join(save_dir, f"metrics_mol_{target_index}.csv")

    # 如果 xyz 文件已存在，先删除，防止追加内容混乱
    if os.path.exists(xyz_path):
        os.remove(xyz_path)

    # 1. 加载 MACE 和模型
    print(f"🔧 正在加载 MACE 和 扩散模型 (权重: {w_tag})...")
    mace_calc = MACECalculator(model_paths=mace_model_path, device=str(device), n_energies=2, default_dtype='float64')

    model = PhysicsInformedSBModule.load_from_checkpoint(ckpt_path,
            mace_model_path=mace_model_path,
            phys_weight=phys_w,
            strict=False).to(device).double()
    model.eval()
    model.ddpm.opt = OPT()

    # 2. 加载验证集数据
    with open(data_path, 'rb') as f:
        valid_data = pickle.load(f)

    reactants = valid_data['reactant']['positions']
    products_true = valid_data['product']['positions']
    all_charges = valid_data['reactant']['charges']
    
    # 获取指定分子的数据
    print(f"🚀 开始单分子逐帧轨迹提取: Index = {target_index}")
    r_pos_np = np.array(reactants[target_index])
    p_pos_true = np.array(products_true[target_index])
    charges_np = np.array(all_charges[target_index])
    N = len(charges_np)

    atom_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}

    # 构造输入张量
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
            # 🌟 核心修改：log_count 必须等于 nfe_steps，这样才能拿到完整 50 帧
            sample_out = model.ddpm.sample(x1_full, [repre_R, repre_P], conditions, 
                                           ot_ode=True, nfe=nfe_steps, log_count=nfe_steps)
            trajectories = sample_out[1].detach().cpu()

        print(f"✅ 采样完成，共生成 {trajectories.shape[1]} 帧轨迹。开始计算物理/几何指标...")

        # 逐帧分析
        for f in range(trajectories.shape[1]):
            # 提取第 f 帧的坐标 (去掉 batch 维度)
            curr_pos = trajectories[:, f, :].numpy() 
            
            # 创建 ASE Atoms 对象
            atoms = Atoms(numbers=charges_np, positions=curr_pos)
            
            # 保存到 xyz 文件 (追加模式)
            write(xyz_path, atoms, append=True)

            # 计算物理 Gap
            atoms.calc = mace_calc
            e = atoms.get_potential_energy().flatten()
            gap = abs(e[1] - e[0])

            # 计算几何 RMSD
            frame_rmsd = calculate_rmsd(curr_pos, p_pos_true)

            # 记录数据
            trajectory_metrics.append({
                'frame': f,
                'time_t': f / (trajectories.shape[1] - 1), # 伪时间: 0(R) -> 1(P)
                'gap_eV': gap,
                'rmsd_A': frame_rmsd
            })

            print(f"  帧 {f:02d} | 伪时间 t={trajectory_metrics[-1]['time_t']:.2f} | Gap: {gap:.4f} eV | RMSD: {frame_rmsd:.4f} Å")

        # 3. 将指标保存为 CSV 文件
        with open(csv_path, mode='w', newline='') as csv_file:
            fieldnames = ['frame', 'time_t', 'gap_eV', 'rmsd_A']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in trajectory_metrics:
                writer.writerow(row)

        print("\n" + "="*50)
        print(f"🎉 分子 {target_index} 轨迹提取圆满成功！")
        print(f"📂 轨迹坐标已保存至: {xyz_path} (可用 PyMOL/VMD 打开观看动画)")
        print(f"📊 帧指标数据已保存至: {csv_path} (可直接读取画折线图)")
        print("="*50)

    except Exception as e:
        print(f"❌ 评估出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
