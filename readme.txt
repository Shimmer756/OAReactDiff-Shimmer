#  analyse里面的rmsd是最新的，适应老任务。

#  model
需要修改的部分，只有diffusion/en_diffusion.py
其中对于维度有一个硬编码，z_t = [_z_t[:, : 3 + 5 + 1] for _z_t in z_t] #其中5为原子的维度，如果有增加，需要更改

#train
python -m oa_reactdiff.trainer.train_ts1x 2>&1 | tee training.log

可以修改train_ts1x.py里面的version号，以建立新的checkpoint

#run
通过run2.py直接可以进行生成。其中ckpt换成训练之后的
