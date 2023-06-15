import os
import sys
sys.path.append('../modules')
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataload import DataLoader
from model import ETLD, Loadmodel
from contact_prec import TM_to_Cscore, ContprecFromCscore

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set which gpu to use
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = 'dhfr'
msapath = './dhfr.fa'
pdbcontact_path = './dhfr_pdb.contact'    # The distance between two C-alpha atoms in the PDB is less than 8 A, 
checkpoint_dir = './run/'
save_dir = './run/'
cscore_path = save_dir + 'cscore.mat'
contprec_path = save_dir + 'contprec.csv'

## dataloader
dataloader = DataLoader(
        dataset=dataset,
        alignment_file=msapath,
        focus_key='',
        calc_weights=True,
        theta=0.2,
        times_of_seq_len=0,
        aa_isupper=False,
        remove_unknown=False,
)

etld = ETLD(
    d=256, d_inner=256, h=8, c=2, caa=2,
    src_vocab_size=dataloader.vocab_size,
    tgt_vocab_size=dataloader.vocab_size,
    src_seq_len=dataloader.seq_len,
    tgt_seq_len=dataloader.seq_len,
    pad_index=dataloader.pad_index,
    dropout=0.1,
    device=device
)

aver_Cscore = np.zeros((dataloader.seq_len - 1, dataloader.seq_len - 1))
for r in range(10):
    checkpoint = checkpoint_dir + '{}_{:0>3d}.tar'.format(dataset, r)
    load_model = Loadmodel(checkpoint, etld)
    Cscore_r = TM_to_Cscore(load_model, symmetry=True)
    aver_Cscore = aver_Cscore + Cscore_r
aver_Cscore /= 10
np.savetxt(cscore_path, aver_Cscore)

ContprecFromCscore(aver_Cscore, contprec_path, delta=5)
plt.subplot(121)
plt.imshow(aver_Cscore)
plt.title('Cscore')

plt.subplot(122)
pdbcontact = np.loadtxt(pdbcontact_path)
plt.imshow(pdbcontact)
plt.title('PdbContact')
plt.show()