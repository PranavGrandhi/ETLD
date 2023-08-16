import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from etld.dataload import DataLoader
from etld.muteffect_prec import Create_Mutant_From_DMScsv, MutEffect_Prec_From_Ckdir, Spearmanr
from etld.logger import Logger

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set which gpu to use
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set model params
model_params={
    'd':256, 
    'd_inner':32, 
    'h':4, 
    'c':2, 
    'caa':2,
    'r': 10,
    'e': 10,
}

dataset = 'blat_ecolx'
msafile = './BLAT_ECOLX_1_b0.5.aln'
dmsfile = 'BLAT_ECOLX_Ranganathan2015.csv'
checkpoint_dir = './blat_ecolx/'
save_dir = './blat_ecolx/'
logfile = save_dir + 'mutprec_example.log'

#get expr_keys
expr_keys = []
fr = open(dmsfile, 'r')
for row in fr:
    if row.startswith('# Experimental data columns:'):
        for ek in row[28:].strip().split(','):
            expr_keys.append(ek.strip()) 
fr.close()

sys.stdout = Logger(logfile, sys.stdout, mode='w') 

print('MUTANT_CSV', dataset, dmsfile)

#creat mutant sequence
create_mutant_file = dataset + '_mutant.fa'
if create_mutant_file not in os.listdir(save_dir):
    create_mutant_file = save_dir + create_mutant_file
    print('Create Mutant File {} From {}'.format(create_mutant_file, dmsfile))
    fr = open(msafile, 'r')
    wt_key = fr.readline()[:-1]
    wt_seq = fr.readline()[:-1]
    fr.close()
               
    Create_Mutant_From_DMScsv(
        dmsfile=dmsfile, 
        wt_key=wt_key, 
        wt_seq=wt_seq,
        mutant_key='mutant', 
        mutantfile=create_mutant_file,
        aa_isupper=True, 
        alphabet_type='protein',
        need_mutant_reindex=True
    )
else:
    create_mutant_file = save_dir + create_mutant_file

# Mutation Efects Prediction
mutant_dataloader = DataLoader(
        dataset=dataset,
        alignment_file=create_mutant_file,
        focus_key='',
        calc_weights=False,
        aa_isupper=True,
        remove_unknown=True,
)

MutEffect_Prec_From_Ckdir(
    mutant_dataloader, 
    checkpoint_dir, 
    model_params=model_params,
    N_pred_iterations=1, 
    minibatch_size=256,
    save_dir=checkpoint_dir,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

                    
for ek in expr_keys:
    Spearmanr(
        save_dir=checkpoint_dir, 
        dataset=dataset, 
        repeats=model_params['r'], 
        dmsfile=dmsfile, 
        expr_key=ek, 
        prec_key='0'
    )
