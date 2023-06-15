import os
import torch
import torch.nn as nn
import numpy as np


def ExtractTM(load_model, outfile=None):

    tm = load_model.state_dict()['TM'].cpu()  # [channal, multi_head, src_seq_len, tgt_seq_len]

    tm = tm.detach().cpu().numpy()

    if outfile is not None:
        np.save(outfile, tm[:, :, :, :])

    return tm[:, :, :, :]

def TM_to_Cscore(load_model, symmetry=True, outfile=None):

    tm = load_model.state_dict()['TM'].cpu()  # [channal, multi_head, src_seq_len, tgt_seq_len]

    tm = tm.abs()
    tm = torch.pow(tm, 2)
    tm = (tm - tm[:, :, 0, 0][:, :, np.newaxis, np.newaxis])
    tm = torch.sum(tm, dim=0)
    tm = torch.sum(tm, dim=0)
    
    Cscore = (tm - torch.mean(tm, dim=1, keepdim=True)) \
          / (torch.std(tm, dim=1, keepdim=True) + 1e-9)    # put font or behand sum 

    if symmetry:
        Cscore = (Cscore + Cscore.T) / 2

    if outfile is not None:
        np.savetxt(outfile, Cscore[1:, 1:])

    return Cscore[1:, 1:].detach().numpy()

def ContprecFromCscore(cscore, outfile, delta=5):
    """Consider only the contact between amino acids with site spacing greater than delta"""

    if type(cscore) is str:
        if os.path.exists(cscore):
            Cscore = np.loadtxt(cscore)
    elif type(cscore) is np.ndarray:
        Cscore = cscore

    row, col = Cscore.shape

    contact_dict = {}
    index = 0
    for n in range(row - delta):
        for m in range(n + delta, col):
            contact_dict[index] = [n + 1, m + 1, m - n, Cscore[n, m]]
            index += 1
    contact_dict = dict(sorted(contact_dict.items(), key=lambda x: x[1][3], reverse=True))

    fw = open(outfile, 'w')
    fw.write('res1,res2,delta,Cscore\n')
    for k, v in contact_dict.items():
        fw.write(','.join(map(str, v)))
        fw.write('\n')

    fw.close()

