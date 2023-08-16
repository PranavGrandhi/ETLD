import os
import torch
import torch.nn as nn
import numpy as np


def ExtractTM(load_model, outfile=None):
    '''Extracting TM from the trained model
    '''

    tm = load_model.state_dict()['TM'].cpu()  # [channal, multi_head, src_seq_len, tgt_seq_len]

    tm = tm.detach().cpu().numpy()

    if outfile is not None:
        np.save(outfile, tm[:, :, :, :])

    return tm[:, :, :, :]

def TM_to_Cscore(load_model, symmetry=True, outfile=None):
    '''Deriving contact matrix (C) from the trained model
    '''    

    tm = load_model.state_dict()['TM'].cpu()  # [channal, multi_head, src_seq_len, tgt_seq_len]

    tm = tm.abs()
    tm = torch.pow(tm, 2)
    tm = (tm - tm[:, :, 0, 0][:, :, np.newaxis, np.newaxis])    # when the source MSA is the same as the target MSA, tm[:, :, 0, 0] = 0. 
    tm = torch.sum(tm, dim=0)
    tm = torch.sum(tm, dim=0)

    Cscore_0 = (tm - torch.mean(tm, dim=0, keepdim=True)) \
          / (torch.std(tm, dim=0, keepdim=True) + 1e-9)    # put font or behand sum     
    Cscore_1 = (tm - torch.mean(tm, dim=1, keepdim=True)) \
          / (torch.std(tm, dim=1, keepdim=True) + 1e-9)    # put font or behand sum 
    
    Cscore = Cscore_0 + Cscore_1    # or Cscore = Cscore_1 in the paper; they have similar accuracy.

    if symmetry:
        Cscore = (Cscore + Cscore.T) / 2

    if outfile is not None:
        np.savetxt(outfile, Cscore[1:, 1:])

    return Cscore[1:, 1:].detach().numpy()

def ContprecFromCscore(cscore, outfile, mask=None, delta=5):
    """Consider only the contact between amino acids with site spacing greater than delta"""

    if type(cscore) is str:
        if os.path.exists(cscore):
            Cscore = np.loadtxt(cscore)
    elif type(cscore) is np.ndarray:
        Cscore = cscore

    if mask is not None:
        if type(mask) is str:
            if os.path.exists(mask):
                Mask = np.loadtxt(mask)
        elif type(cscore) is np.ndarray:
            Mask = mask

        assert Cscore.shape == Mask.shape
    row, col = Cscore.shape

    contact_dict = {}
    index = 0
    for n in range(row - delta):
        for m in range(n + delta, col):
            if mask is None:
                contact_dict[index] = [n + 1, m + 1, m - n, Cscore[n, m]]
            else:
                contact_dict[index] = [n + 1, m + 1, m - n, Cscore[n, m], bool(Mask[n, m])]
            index += 1
    contact_dict = dict(sorted(contact_dict.items(), key=lambda x: x[1][3], reverse=True))

    head = 'res1,res2,delta,Cscore\n' if mask is None else 'res1,res2,delta,Cscore,Real\n'
    fw = open(outfile, 'w')
    fw.write(head)
    for k, v in contact_dict.items():
        fw.write(','.join(map(str, v)))
        fw.write('\n')

    fw.close()

def ContAllPrecAcc(contactfile, topN):
    '''Calculation of prediction accuracy for the first topN contacts.
    '''
    TP, FP = 0, 0
    delta_6_11_TP, delta_6_11_FP = 0, 0 # short-range
    delta_12_24_TP, delta_12_24_FP = 0, 0   # medium-range
    delta_24_TP, delta_24_FP = 0, 0 # long-range
    delta_6_11_prec = delta_12_24_prec = delta_24_prec = 0

    fr = open(contactfile, 'r')
    head = fr.readline()
    if 'Real' not in head:
        print('Column "Real" is missing from the {}.'.format(contactfile))
        return 0
    
    for row in fr.readlines()[:topN]:
        data = row.strip().split(',')
        i, j, delta, score, true_contact = int(data[0]) - 1, int(data[1]) - 1, \
                                           int(data[2]), float(data[3]), data[4]

        if delta >= 6 and delta < 12:
            if true_contact.lower() == 'true':
                TP += 1
                delta_6_11_TP += 1
            else:
                FP += 1
                delta_6_11_FP += 1
        elif delta >= 12 and delta <= 24:
            if true_contact.lower() == 'true':
                TP += 1
                delta_12_24_TP += 1
            else:
                FP += 1
                delta_12_24_FP += 1
        elif delta > 24:
            if true_contact.lower() == 'true':
                TP += 1
                delta_24_TP += 1
            else:
                FP += 1
                delta_24_FP += 1
    prec = TP / (TP + FP)
    if (delta_6_11_TP + delta_6_11_FP) > 0:
        delta_6_11_prec = delta_6_11_TP / (delta_6_11_TP + delta_6_11_FP)

    if (delta_12_24_TP + delta_12_24_FP) > 0:
        delta_12_24_prec = delta_12_24_TP / (delta_12_24_TP + delta_12_24_FP)

    if (delta_24_TP + delta_24_FP) > 0:
        delta_24_prec = delta_24_TP / (delta_24_TP + delta_24_FP)

    return prec, delta_6_11_prec, delta_12_24_prec, delta_24_prec

def ResRangeContPrecAcc(contactfile, res_range, topN):
    '''Calculation of prediction accuracy for the first topN contacts in the residue interval range.
    '''

    # residue interval range: [lower, upper)
    if type(res_range) is int:
        lower, upper = res_range, 1000000 
    elif type(res_range) is list:
        lower, upper = res_range[0], res_range[1]

    TP, FP = 0, 0

    fr = open(contactfile, 'r')
    head = fr.readline()
    if 'Real' not in head:
        print('Column "Real" is missing from the {}.'.format(contactfile))
        return 0
    
    for row in fr.readlines():
        data = row.strip().split(',')
        i, j, delta, score, true_contact = int(data[0]) - 1, int(data[1]) - 1, \
                                           int(data[2]), float(data[3]), data[4]

        if delta >= lower and delta < upper :
            if true_contact.lower() == 'true':
                TP += 1
            else:
                FP += 1

        if TP + FP >= topN:
            break
    fr.close()

    prec = TP / (TP + FP)

    return prec

def ContactRangeContPrecAcc(contactfile, contact_range, topN):
    '''Calculation of prediction accuracy for the first topN short/medium/long/medium-long contacts.
    '''

    if contact_range == 'short':
        res_range = [6, 12]
    elif contact_range == 'medium':
        res_range = [12, 24]
    elif contact_range == 'long':
        res_range = [24, np.inf]
    elif contact_range == 'medium-long':
        res_range = [12, np.inf]
    else:
        raise ValueError("contact_range must be one of 'short', 'medium', 'long', or 'medium-long'.")  

    prec = ResRangeContPrecAcc(contactfile, res_range, topN)

    return prec

