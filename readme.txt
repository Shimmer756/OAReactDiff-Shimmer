#  analyse里面的rmsd是最新的，适应老任务。

#  model
需要修改的部分，只有diffusion/en_diffusion.py
其中对于维度有一个硬编码，z_t = [_z_t[:, : 3 + 5 + 1] for _z_t in z_t] #其中5为原子的维度，如果有增加，需要更改
