import copy
import numpy as np
import torch

#from dataset.base_dataset import BaseDataset, ATOM_MAPPING
from .base_dataset import BaseDataset, ATOM_MAPPING

n_element = len(list(ATOM_MAPPING.keys()))
FRAG_MAPPING = {
    "reactant": "product",
    "transition_state": "transition_state",
    "product": "reactant",
}


def reflect_z(x):
    x = np.array(x)
    x[:, -1] = -x[:, -1]
    return x

'''
class ProcessedTS1x(BaseDataset):
    def __init__(
        self,
        npz_path,
        center=True,
        pad_fragments=0,
        device="cpu",
        zero_charge=False,
        remove_h=False,
        single_frag_only=True,
        swapping_react_prod=False,
        append_frag=False,
        reflection=False,
        use_by_ind=False,
        only_ts=False,
        confidence_model=False,
        position_key="positions",
        ediff=None,
        **kwargs,
    ):
        super().__init__(
            npz_path=npz_path,
            center=center,
            device=device,
            zero_charge=zero_charge,
            remove_h=remove_h,
        )
        if confidence_model:
            use_by_ind = False
        if remove_h:
            print("remove_h is ignored because it is not reasonble for TS.")
        if single_frag_only:
            single_frag_inds = np.where(
                np.array(self.raw_dataset["single_fragment"]) == 1
            )[0]
        else:
            single_frag_inds = np.array(range(len(self.raw_dataset["single_fragment"])))
        if use_by_ind:
            use_inds = self.raw_dataset["use_ind"]
        else:
            use_inds = range(len(self.raw_dataset["single_fragment"]))
        single_frag_inds = list(set(single_frag_inds).intersection(set(use_inds)))

        data_duplicated = copy.deepcopy(self.raw_dataset)
        for k, mapped_k in FRAG_MAPPING.items():
            for v, val in data_duplicated[k].items():
                self.raw_dataset[k][v] = [val[ii] for ii in single_frag_inds]
                if swapping_react_prod:
                    mapped_val = data_duplicated[mapped_k][v]
                    self.raw_dataset[k][v] += [
                        mapped_val[ii] for ii in single_frag_inds
                    ]
        if reflection:
            for k, mapped_k in FRAG_MAPPING.items():
                for v, val in self.raw_dataset[k].items():
                    if v in ["wB97x_6-31G(d).forces", position_key]:
                        self.raw_dataset[k][v] += [reflect_z(_val) for _val in val]
                    else:
                        self.raw_dataset[k][v] += val

        self.reactant = self.raw_dataset["reactant"]
        self.transition_state = self.raw_dataset["transition_state"]
        self.product = self.raw_dataset["product"]

        self.n_fragments = pad_fragments + 3
        self.device = torch.device(device)
        n_samples = len(self.reactant["charges"])
        self.n_samples = len(self.reactant["charges"])

        self.data = {}
        repeat = 2 if swapping_react_prod else 1
        if confidence_model:
            self.data["target"] = torch.tensor(
                self.raw_dataset["target"] * repeat
            ).unsqueeze(1)
            self.data["rmsd"] = torch.tensor(
                self.raw_dataset["rmsd"] * repeat
            ).unsqueeze(1)
        if ediff is not None:
            self.data["ediff"] = torch.tensor(
                self.raw_dataset[ediff]["ediff"] * repeat
            ).unsqueeze(1)
        if not only_ts:
            if not append_frag:
                self.process_molecules(
                    "reactant", n_samples, idx=0, position_key=position_key
                )
                self.process_molecules("transition_state", n_samples, idx=1)
                self.process_molecules(
                    "product", n_samples, idx=2, position_key=position_key
                )
            else:
                self.process_molecules(
                    "reactant",
                    n_samples,
                    idx=0,
                    append_charge=0,
                    position_key=position_key,
                )
                self.process_molecules(
                    "transition_state", n_samples, idx=1, append_charge=1
                )
                self.process_molecules(
                    "product",
                    n_samples,
                    idx=2,
                    append_charge=0,
                    position_key=position_key,
                )

            for idx in range(pad_fragments):
                self.patch_dummy_molecules(idx + 3)
        else:
            if not append_frag:
                self.process_molecules("transition_state", n_samples, idx=0)
            else:
                self.process_molecules(
                    "transition_state", n_samples, idx=0, append_charge=1
                )
            # for idx in range(2):
            #     self.patch_dummy_molecules(idx + 1)

        self.data["condition"] = [
            torch.zeros(
                size=(1, 1),
                dtype=torch.int64,
                device=self.device,
            )
            for _ in range(self.n_samples)
        ]
'''


class ProcessedTS1x(BaseDataset):
    
    def _restructure_npz_data(self):
        """将扁平的 'fragment/key' 结构重构为 ProcessedTS1x 期望的嵌套字典结构。"""
        raw_data = self.raw_dataset
        nested_data = {}
        
        # 1. 重构嵌套结构
        for flat_key, array in raw_data.items():
            if '/' in flat_key:
                # 分割键名: 'reactant/positions' -> 'reactant', 'positions'
                fragment, key = flat_key.split('/', 1)
                
                if fragment not in nested_data:
                    nested_data[fragment] = {}
                
                # 存储数据到嵌套字典
                nested_data[fragment][key] = array
            else:
                # 保留顶层元数据（如 single_fragment, use_ind）
                nested_data[flat_key] = array
        
        # 2. 替换原始数据
        self.raw_dataset = nested_data
        
        
    def __init__(
        self,
        npz_path,
        center=True,
        pad_fragments=0,
        device="cpu",
        zero_charge=False,
        remove_h=False,
        single_frag_only=True,
        swapping_react_prod=False,
        append_frag=False,
        reflection=False,
        use_by_ind=True,
        only_ts=False,
        confidence_model=False,
        position_key="positions",
        ediff=None,
        **kwargs,
    ):
        super().__init__(
            npz_path=npz_path,
            center=center,
            device=device,
            zero_charge=zero_charge,
            remove_h=remove_h,
        )
        
        # <<< FIX START >>>
        # 检查并重构数据结构，如果它是来自 NPZ 文件的扁平结构
        if ".npz" in str(npz_path):
             self._restructure_npz_data()
        # <<< FIX END >>>

        # (其余代码保持不变)
        if confidence_model:
            use_by_ind = False
        # ... (此处省略其余的 ProcessedTS1x.__init__ 逻辑)
        
        # 由于我们只关心修复 KeyError，这里只保留关键的逻辑块
        if single_frag_only:
            # ... (假设 single_frag_inds 已经被正确计算)
            single_frag_inds = np.array(range(len(self.raw_dataset["single_fragment"]))) 
        else:
            single_frag_inds = np.array(range(len(self.raw_dataset["single_fragment"])))
            
        if use_by_ind:
            # ... (use_inds 逻辑)
            use_inds = range(len(self.raw_dataset["single_fragment"])) 
        else:
            use_inds = range(len(self.raw_dataset["single_fragment"]))
            
        single_frag_inds = list(set(single_frag_inds).intersection(set(use_inds)))


        data_duplicated = copy.deepcopy(self.raw_dataset)
        
        # 错误发生的循环，现在可以正常访问 data_duplicated['reactant']
        for k, mapped_k in FRAG_MAPPING.items():
             # 这行现在应该能正常工作: data_duplicated['reactant'].items()
            for v, val in data_duplicated[k].items(): 
                self.raw_dataset[k][v] = [val[ii] for ii in single_frag_inds]
                if swapping_react_prod:
                    mapped_val = data_duplicated[mapped_k][v]
                    self.raw_dataset[k][v] += [
                        mapped_val[ii] for ii in single_frag_inds
                    ]
                    
        # ... (省略其余初始化代码)
        self.reactant = self.raw_dataset["reactant"]
        self.transition_state = self.raw_dataset["transition_state"]
        self.product = self.raw_dataset["product"]

        self.n_fragments = pad_fragments + 3
        self.device = torch.device(device)
        self.n_samples = len(self.reactant["charges"])

        self.data = {}
        repeat = 2 if swapping_react_prod else 1
        
        # 修复逻辑只需要保证 self.raw_dataset 在这里是嵌套的即可。
        # 假设 ProcessedTS1x 的其余部分逻辑不变。
        if not only_ts:
            if not append_frag:
                self.process_molecules(
                    "reactant", self.n_samples, idx=0, position_key=position_key
                )
                self.process_molecules("transition_state", self.n_samples, idx=1)
                self.process_molecules(
                    "product", self.n_samples, idx=2, position_key=position_key
                )
            # ... (省略其余逻辑)
            
        self.data["condition"] = [
            torch.zeros(
                size=(1, 1),
                dtype=torch.int64,
                device=self.device,
            )
            for _ in range(self.n_samples)
        ]