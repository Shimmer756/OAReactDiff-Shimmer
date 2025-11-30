import h5py
import numpy as np
import re
from typing import Dict, List, Any, Tuple

# --- 配置 ---
H5_FILE_PATH = 'merged_meci_dataset_onehotandcharges.h5' # 请替换为您的HDF5文件路径
NPZ_FILE_PATH = 'custom_R_P_multi_molecule.npz'
MAX_ATOMS = 150 # 根据您的数据设置

# --- 辅助函数：提取步骤数字 ---
def get_step_number(step_key: str) -> int:
    """从 'step_X' 格式的键中提取数字 X，用于排序。"""
    match = re.match(r'step_(\d+)', step_key)
    if match:
        return int(match.group(1))
    return -1 # 返回 -1 确保非step键被忽略

# --- 1. 数据提取与格式化函数 ---
def extract_ts1x_data_from_h5(h5_file_path: str, max_atoms: int) -> Tuple[Dict[str, Dict[str, List[Any]]], int]:
    """
    遍历 HDF5 文件中的所有分子，提取 R 和 P 数据，并构造 TS 占位符。
    P (产物) 的逻辑：优先使用 step_29，否则使用数字最大的 step。
    """
    
    # 存储最终数据的结构
    all_fragments_data: Dict[str, Dict[str, List[Any]]] = {
        role: {'positions': [], 'charges': [], 'num_atoms': []} 
        for role in ['reactant', 'transition_state', 'product']
    }
    
    # 跟踪成功提取的分子数量
    successful_mols = 0

    with h5py.File(h5_file_path, 'r') as f:
        molecule_ids = list(f.keys())
        
        for mol_id in molecule_ids:
            mol_group = f[mol_id]
            all_steps = list(mol_group.keys())
            
            # --- A. 确定产物 (Product) 对应的步骤名称 (P_step_name) ---
            P_step_name = None
            
            if 'step_29' in all_steps:
                P_step_name = 'step_29'
                #print(f"✅ 分子 {mol_id}: 找到目标步骤 step_29。")
            else:
                # 筛选并按数字排序所有 step_X 键
                step_keys_with_num = [(key, get_step_number(key)) for key in all_steps if get_step_number(key) != -1]
                
                if step_keys_with_num:
                    # 按数字降序排列
                    step_keys_with_num.sort(key=lambda item: item[1], reverse=True)
                    P_step_name = step_keys_with_num[0][0] # 最大的数字对应的键
                    print(f"⚠️ 分子 {mol_id}: 缺少 step_29。已使用数字最大的步骤 {P_step_name} 作为产物。")
                
            if P_step_name is None:
                print(f"警告：分子 {mol_id} 既没有 step_29 也没有其他 step_X 键，跳过。")
                continue
            
            # --- B. 定义最终提取的角色映射 ---
            final_roles = {
                'reactant': 'step_0',
                'product': P_step_name
            }
            
            if final_roles['reactant'] not in all_steps:
                 print(f"警告：分子 {mol_id} 缺少 step_0，无法构建 R/P 对，跳过。")
                 continue
                 
            # --- C. 提取 R 和 P 数据 ---
            current_r_data = {}
            for role, step_name in final_roles.items():
                
                step_data = mol_group[step_name]
                
                # 提取数据
                positions_raw = step_data['coordinates'][:]
                charges_raw = step_data['atomic_numbers'][:]
                num_atoms = len(charges_raw)
                
                # 数据填充 (Padding)
                positions_padded = np.zeros((max_atoms, 3), dtype=np.float32)
                positions_padded[:num_atoms] = positions_raw[:num_atoms]
                
                charges_padded = np.zeros(max_atoms, dtype=np.int64)
                charges_padded[:num_atoms] = charges_raw
                
                # 存储当前分子的数据
                all_fragments_data[role]['positions'].append(positions_padded)
                all_fragments_data[role]['charges'].append(charges_padded)
                all_fragments_data[role]['num_atoms'].append(num_atoms)
                
                if role == 'reactant':
                    # 存储 R 数据，用于后续作为 TS 的占位符
                    current_r_data = {'positions': positions_padded, 'charges': charges_padded, 'num_atoms': num_atoms}

            # --- D. 构造 Transition State (TS) 占位符 ---
            ts_role = 'transition_state'
            all_fragments_data[ts_role]['positions'].append(current_r_data['positions'])
            all_fragments_data[ts_role]['charges'].append(current_r_data['charges'])
            all_fragments_data[ts_role]['num_atoms'].append(current_r_data['num_atoms'])
            
            successful_mols += 1

    return all_fragments_data, successful_mols

# --- 2. 构造和保存最终 NPZ 结构 ---

if __name__ == '__main__':
    
    try:
        # 提取数据
        all_data, num_samples = extract_ts1x_data_from_h5(H5_FILE_PATH, MAX_ATOMS)
        
        if num_samples == 0:
            print("错误: 未从 HDF5 文件中提取到任何有效的反应数据。")
            exit()
            
        print(f"\n========================================================")
        print(f"✅ 成功提取 {num_samples} 个反应路径的数据。")

        # 构造最终的 NPZ 字典结构
        final_npz_data = {}
        
        for role in ['reactant', 'transition_state', 'product']:
            r_data = all_data[role]
            
            final_npz_data[f'{role}/positions'] = np.stack(r_data['positions'], axis=0)
            final_npz_data[f'{role}/charges'] = np.stack(r_data['charges'], axis=0)
            final_npz_data[f'{role}/num_atoms'] = np.array(r_data['num_atoms'], dtype=np.int64)

        # 增加 ProcessedTS1x 所需的顶层元数据
        final_npz_data['single_fragment'] = np.ones(num_samples, dtype=np.int64) 
        final_npz_data['use_ind'] = np.arange(num_samples, dtype=np.int64)

        # 保存为 NPZ 文件
        np.savez(NPZ_FILE_PATH, **final_npz_data)
        
        print(f"\n✅ 数据集已成功构建并保存为 {NPZ_FILE_PATH}")
        print(f"总样本数 (N_samples): {num_samples}")
        
    except FileNotFoundError:
        print(f"\n错误: 未找到文件 {H5_FILE_PATH}。请检查文件路径是否正确。")
    except KeyError as e:
        print(f"\n错误: HDF5 文件结构不匹配。缺少键: {e}")
    except Exception as e:
        print(f"\n发生未知错误: {e}")
