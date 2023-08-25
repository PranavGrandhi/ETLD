import torch
import numpy as np
import os, sys
import torch.nn as nn
import matplotlib.pyplot as plt
from etld.resnet import ResNet, prec_step
from etld.contact_prec import ContprecFromCscore, ContAllPrecAcc, ContactRangeContPrecAcc
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

datadir = './data/'
checkpoint_dir = './ck/'
save_dir = './data/'
try:
    os.mkdir(save_dir)
except:
    pass

# Load ResNet Model
model = ResNet(r=10)
model.to(device=device)

# load dataloader
dataloader = np.load(datadir + 'train_dataloader_0_mask8.npy', allow_pickle=True).item()
datasets = list(dataloader.keys())
datasets = sorted(datasets)

# prec
for d in datasets:
    print('>' + d)
    tm, mask = dataloader[d]
    prec_ids = np.zeros(mask.shape)
    tm, mask = torch.tensor(tm, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
    seq_len = mask.shape[0]

    for n in range(5):  # range(5)
        checkpoint = checkpoint_dir + f'train_{n}.ck.tar'
        ckpt = torch.load(checkpoint)
        model.load_state_dict(ckpt['net'])
        prec_id_n = prec_step(model, tm, device)
        prec_ids += prec_id_n

        prec_ids = prec_ids / 5
        #prec_ids = np.where(prec_ids > 0.9999999, 1, 0)

    seqlen = mask.shape[0]
    topL1 = int(seqlen / 1)
    topL5 = int(seqlen / 5)

    method = 'ResNet'
    contprec_path = save_dir + f'{d}_contprec.csv'

    ContprecFromCscore(prec_ids, contprec_path, mask=datadir + '{}.mask'.format(d), delta=5)
    print('(Prec) all, short, medium, long:', ContAllPrecAcc(contactfile=contprec_path, topN=int(seq_len / 5)))
    print('(Prec) long:', ContactRangeContPrecAcc(contactfile=contprec_path, contact_range='long', topN=int(seq_len / 5)))

    plt.subplot(121)
    plt.imshow(prec_ids)
    plt.title('ResNet')

    plt.subplot(122)
    pdbcontact = np.loadtxt(datadir + f'{d}_pdb.contact')
    plt.imshow(pdbcontact)
    plt.title('PdbContact')
    plt.show()       

