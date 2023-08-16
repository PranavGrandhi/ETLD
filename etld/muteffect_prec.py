import os, sys
import pandas as pd
import numpy as np
import scipy.stats as stats
import torch
from etld.dataload import DataLoader
from etld.model import ETLD, Loadmodel, create_padding_mask

# create mutant sequences FASTA file from the DMS file
def Create_Mutant_From_DMScsv(dmsfile, wt_key, wt_seq,
                              mutant_key='mutant', mutantfile='mutant.fa',
                              aa_isupper=True, alphabet_type='protein',
                              need_mutant_reindex=True
                              ):

    '''
    mutant_tuple_list : ['a0B', 'a0b;A1C']

    dmsfile: DMS file
    wt_key: the key of the wild sequence 
    wt_seq: the sequence of the wild sequence 
    mutant_key: default:'mutant'
    mutantfile: output mutant sequences file, default: 'mutant.fa',
    aa_isupper: whether focus on the capital amino acid, default:True
    alphabet_type: when the sequence is protein, using 'protein',
    need_mutant_reindex: whether change the index of mutant, default: True   
    '''

    df_measurement = pd.read_csv(dmsfile, comment='#', delimiter=',')
    multi_mutant_tuple_list = df_measurement[mutant_key].tolist()

    if alphabet_type.lower() == 'protein':
        alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    elif alphabet_type.lower() == 'rna':
        alphabet = 'ACGU'
    elif alphabet_type.lower() == 'dna':
        alphabet = 'ACGT'
    else:
        alphabet = alphabet_type.upper()

    if need_mutant_reindex:
        # [NAME]/[start]-[end]
        focus_loc = wt_key.split('/')[-1]
        start, end = focus_loc.split('-')
        focus_start_loc = int(start)
    else:
        focus_start_loc = 0

    # focus on the amnio acid in the alphabet; and the capital when aa_isupper is True
    if aa_isupper:
        focus_cols = [i for i, s in enumerate(wt_seq) if s.isupper() and s in alphabet]
    else:
        focus_cols = [i for i, s in enumerate(wt_seq) if s.upper() in alphabet]

    wt_seq = [wt_seq[idx].upper() for idx in focus_cols]

    # focus cols related to wt_seq without supper trimmed;
    # wt_seq cols related to wt_seq with supper trimmed: range(len(focus cols))
    # ori_seq_cols related to all wt seq from database

    focus_ori_seq_cols_to_focus_wt_seq_cols \
        = {idx + focus_start_loc: i for i, idx in enumerate(focus_cols)}    # original_sequence_idx: wt_seq_cols
    focus_ori_seq_cols = list(focus_ori_seq_cols_to_focus_wt_seq_cols.keys())

    # check mut & delete not match the condition
    check_multi_mutant_tuple_list = []
    for multi_mut in multi_mutant_tuple_list:
        multi_mut_to_list = multi_mut.split(';')
        temp_multi_mut_to_tuple = []
        for mut in multi_mut_to_list:
            pos, wt_aa, mut_aa = int(mut[1:-1]), mut[0].upper(), mut[-1].upper()
            if (pos in focus_ori_seq_cols) and (wt_aa in alphabet) and (mut_aa in alphabet) \
                and (wt_seq[focus_ori_seq_cols_to_focus_wt_seq_cols[pos]] == wt_aa):
                temp_multi_mut_to_tuple.append((pos, wt_aa, mut_aa))
            else:
                # print(multi_mut, 'is not a accessible mutant type !')
                break
        if len(temp_multi_mut_to_tuple) == len(multi_mut_to_list):
            check_multi_mutant_tuple_list.append(temp_multi_mut_to_tuple)

    fw = open(mutantfile, 'w')
    fw.write('>' + wt_key + '\n')
    fw.write(''.join(wt_seq) + '\n')
    for multi_mut in check_multi_mutant_tuple_list:
        descriptor = ''
        wt_seq_copy = wt_seq[:]
        for (pos, wt_aa, mut_aa) in multi_mut:
            # Make a descriptor
            descriptor += wt_aa + str(pos) + mut_aa + ';'

            # Mutate
            wt_seq_copy[focus_ori_seq_cols_to_focus_wt_seq_cols[pos]] = mut_aa
        descriptor = descriptor[:-1]

        # Write
        fw.write('>' + descriptor + '\n')
        fw.write(''.join(wt_seq_copy) + '\n')
    fw.close()

# mutation effect prediction from the loadmodel
def MutEffect_Prec_From_Loadmodel(loadmodel, dataloader: DataLoader, precfile, N_pred_iterations=1,
              minibatch_size=256):

    '''
    loadmodel: etld that have load the checkpoint
    dataloader: mutant_dataloader: DataLoader
    precfile: output mutprec_file
    N_pred_iterations=1: the number of calculations, one is enough
    minibatch_size=256: batch size when predict the muteffect of mutants
    '''
    device = loadmodel.device
    train_tensors = dataloader.train_tensors
    train_tensors = train_tensors.to(device)

    loadmodel.eval()

    # Then make the one hot sequence
    mutant_sequences_one_hot = dataloader.one_hots_from_seqs()    # [nseqs, seq_len, vocab_size]

    prediction_matrix = np.zeros(
        (dataloader.nseqs, N_pred_iterations)   # [nseqs, itaractions]
    )
    batch_order = np.arange(dataloader.nseqs)

    for j in range(N_pred_iterations):
        np.random.shuffle(batch_order)
        for i in range(0, dataloader.nseqs, minibatch_size):
            batch_index = batch_order[i:i + minibatch_size]

            train_batch_tensors = train_tensors[batch_index, :]

            train_padding_mask = create_padding_mask(train_batch_tensors, dataloader.pad_index).to(device)
            batch_preds, _ = loadmodel(train_batch_tensors, train_batch_tensors, train_padding_mask, TM_mask=None) # [b, seq_len, vocab_size]

            batch_preds = batch_preds.detach().cpu().numpy()
            prediction_matrix[batch_index, j] = np.sum(
                    np.sum(mutant_sequences_one_hot[batch_index, :, :] * batch_preds[:, :, :], axis=-1), axis=-1)

    # nseqs, niterations
    delta_E = prediction_matrix - prediction_matrix[0, :][np.newaxis, :]

    if precfile is not None:
        fw = open(precfile, 'w')
        fw.write('mutant,')
        fw.write(','.join([str(i) for i in range(N_pred_iterations)]))
        fw.write('\n')
        for j, key in enumerate(dataloader.keys):
            fw.write(key)
            for i in range(N_pred_iterations):
                fw.write(','+str('{:>8.3}'.format(delta_E[j, i])))
            fw.write('\n')
        fw.close()

    return delta_E

# mutation effect prediction from checkpoints in the ckdir
def MutEffect_Prec_From_Ckdir(dataloader, checkpoint_dir, model_params={},
                  N_pred_iterations=1, minibatch_size=256,
                  save_dir='./',
                  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    '''
    dataloader: mutant dataloader: Dataloader
    checkpoint_dir:
    model_params: the model params when training etld
    N_pred_iterations=1: the number of calculations, one is enough
    minibatch_size=256: batch size when predict the muteffect of mutants
    save_dir='./': save dir of output mutprec.csv
    device: device of etld, default:torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    '''

    # Load Model
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

    etld = etld.to(device)
    dataset = dataloader.dataset
    for r in range(model_params['r']):
        ck = checkpoint_dir + '{}_{:0>3d}.tar'.format(dataset, r)
        prec_csv = save_dir + 'mutprec_{}_{:0>3d}.csv'.format(dataset, r)
        loadmodel = Loadmodel(ck, etld)

        MutEffect_Prec_From_Loadmodel(loadmodel, dataloader, prec_csv, N_pred_iterations,
              minibatch_size)

# calculate the spearmanr between the muteffect_prec_values and expr_values
def Spearmanr(save_dir, dataset, repeats, dmsfile, expr_key, prec_key='0'):
    '''
    save_dir: the save dir of the mutprec.csv
    repeats: model_params['r']
    dmsfile: DMS csv 
    expr_key: the selected expr key in the DMS csv 
    prec_key: prec_key in the mutprec.csv
    '''

    df_expr = pd.read_csv(dmsfile, comment='#', delimiter=',')

    expr_mutant_list = df_expr.mutant.tolist()
    expr_value_list = df_expr[expr_key].tolist()
    expr_mutant_to_expr_value = {mutant.upper(): expr_value_list[i] for i, mutant in enumerate(expr_mutant_list)}

    aver_pred_values = np.empty(1)
    for r in range(repeats):
        pred_file = save_dir + f'mutprec_{dataset}_{r:0>3}.csv'
        df_pred = pd.read_csv(pred_file)
        pred_values = np.array(df_pred[prec_key].tolist()[1:])
        pred_values = (pred_values - np.min(pred_values)) / (np.max(pred_values) - np.min(pred_values))

        if r == 0:
            pred_mutant_list = df_pred.mutant.tolist()[1:]
            non_null_index = [i for i, mutant in enumerate(pred_mutant_list) if not np.isnan(expr_mutant_to_expr_value[mutant])]
            non_null_mutant = [mutant for i, mutant in enumerate(pred_mutant_list) if not np.isnan(expr_mutant_to_expr_value[mutant])]

            non_null_expr_value_list = [expr_mutant_to_expr_value[mutant] for mutant in non_null_mutant]
            aver_pred_values = pred_values[non_null_index]

        else:
            aver_pred_values += pred_values[non_null_index]

    aver_pred_values /= repeats
    spearmanr_r, spearmanr_p = stats.spearmanr(non_null_expr_value_list, aver_pred_values.tolist())
    print('SPEARMANR', dataset, '| expr_key', expr_key, '| N', str(len(non_null_mutant)), '| Spearmanr', spearmanr_r, '| p-val', spearmanr_p)

