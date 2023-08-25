import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import etld
from etld.dataload import DataLoader
from etld.model import ETLD, Loadmodel
from etld.contact_prec import TM_to_Cscore, ExtractTM, ContprecFromCscore, ContAllPrecAcc, ContactRangeContPrecAcc

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set which gpu to use
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#set parameters
model_params={
    'd':256, 
    'd_inner':256, 
    'h':8, 
    'c':2, 
    'caa':2,
    'r': 10,
    'e': 10,
}

dataset = 'dyr_ecoli'
msapath = './dyr_ecoli.fa'
pdbcontact_path = './dyr_ecoli_pdb.contact'    # The distance between two C-alpha atoms in the PDB is less than 8 A, 
checkpoint_dir = './dyr_ecoli/'
save_dir = './dyr_ecoli/'
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
    d=model_params['d'], 
    d_inner=model_params['d_inner'], 
    h=model_params['h'], 
    c=model_params['c'], 
    caa=model_params['caa'],
    src_vocab_size=dataloader.vocab_size,
    tgt_vocab_size=dataloader.vocab_size,
    src_seq_len=dataloader.seq_len,
    tgt_seq_len=dataloader.seq_len,
    pad_index=dataloader.pad_index,
    dropout=0.1,
    device=device
)

aver_Cscore = np.zeros((dataloader.seq_len - 1, dataloader.seq_len - 1))
for r in range(model_params['r']):
    checkpoint = checkpoint_dir + '{}_{:0>3d}.tar'.format(dataset, r)
    load_model = Loadmodel(checkpoint, etld)
    Cscore_r = TM_to_Cscore(load_model, symmetry=True)
    TM_r = ExtractTM(load_model)
    np.save(checkpoint_dir + '{}_{:0>3d}.tm.npy'.format(dataset, r), TM_r)  # save TM
    aver_Cscore = aver_Cscore + Cscore_r
aver_Cscore /= model_params['r']
np.savetxt(cscore_path, aver_Cscore)

ContprecFromCscore(aver_Cscore, contprec_path, mask='dyr_ecoli.mask', delta=5)
print('(Prec) all, short, medium, long:', ContAllPrecAcc(contactfile=save_dir + 'contprec.csv', topN=int((dataloader.seq_len - 1) / 5)))
print('(Prec) long:', ContactRangeContPrecAcc(contactfile=save_dir + 'contprec.csv', contact_range='long', topN=int((dataloader.seq_len - 1) / 5)))

plt.subplot(121)
plt.imshow(aver_Cscore)
plt.title('Cscore')

plt.subplot(122)
pdbcontact = np.loadtxt(pdbcontact_path)
plt.imshow(pdbcontact)
plt.title('PdbContact')
plt.show()