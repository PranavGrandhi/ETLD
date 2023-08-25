import os
import sys
import numpy as np
import torch
from etld.dataload import ResNet_DataLoader

rd = ResNet_DataLoader(repeats=10, min_res_delta=5, mask_threshold=8, max_seq_len=np.inf, provide_mask='dist')

rd.build_ResNet_train_dataloader(datasets=['dyr_ecoli'], rootdir='./', savedir='./', limit_one_dataloader=100)
dataloader = np.load('./train_dataloader_0_mask8.npy', allow_pickle=True).item()
print(dataloader['dyr_ecoli'][0].shape, dataloader['dyr_ecoli'][1].shape)


rd.build_ResNet_tm_dataloader(datasets=['dyr_ecoli'], rootdir='./', savedir='./', limit_one_dataloader=100)
dataloader = np.load('./tm_dataloader_0_mask8.npy', allow_pickle=True).item()
print(dataloader['dyr_ecoli'].shape)