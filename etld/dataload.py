import numpy as np
from collections import defaultdict
import numba as nb
from numba import njit
from numba import prange
import torch
import os

from etld.utils import mask_from_rrdist

nb.config.NUMBA_DEFAULT_NUM_THREADS = 32

class DataLoader():
    """Create a dataloader for an MSA

    Parameters:
    dataset: the name of the dataset
    alignment_file: MSA file
    focus_key: reference sequence, default the fisrt one
    calc_weights=False: using the weights or not
    theta=0.2: Hamming distance cutoff, if calc_weights=Ture
    times_of_seq_len=0: "times_of_seq_len * seq_len" is the number of training sequences in one epoch; 0 means use all;
    aa_isupper=False: select upper sites only; aa_isupper=Ture, will also remove '-'
    remove_unknown=False

    """
    def __init__(self,
                 dataset='',
                 alignment_file='',
                 focus_key='',
                 calc_weights=False,
                 theta=0.2,
                 times_of_seq_len=0,
                 aa_isupper=False,
                 remove_unknown=False
                 ):

        self.dataset = dataset
        self.alignment_file = alignment_file
        self.focus_key = focus_key
        self.calc_weights = calc_weights
        self.theta = theta
        self.times_of_seq_len = times_of_seq_len

        self.aa_isupper = aa_isupper
        self.remove_unknown = remove_unknown


        self.alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        self.alphabet_size = len(self.alphabet)

        self.gen_basic_alignment()
        self.gen_full_alignment()

        self.feature_tensors = None
        self.feature_size = None

        print('DATA_PARAMS', self.dataset, 'DataSet      ', f'{self.dataset:>8}')
        print('DATA_PARAMS', self.dataset, 'Theta        ', f'{self.theta:>8}')
        print('DATA_PARAMS', self.dataset, 'Ts_of_Seq_Len', f'{self.times_of_seq_len:>8}')
        print('DATA_PARAMS', self.dataset, 'Neff         ', f'{self.Neff:>8}')
        print('DATA_PARAMS', self.dataset, 'Seq_Nums     ', f'{self.nseqs:>8}')
        print('DATA_PARAMS', self.dataset, 'Seq_Length   ', f'{self.seq_len:>8}')
        print('DATA_PARAMS', self.dataset, 'Alphabet_Size', f'{self.alphabet_size:>8}')
        print('DATA_PARAMS', self.dataset, 'Vocab_Size   ', f'{self.vocab_size:>8}')

    def gen_basic_alignment(self):

        self.pad_index = 0  # padding or unknow amino acids
        self.start_index = 1  # the start marker
        self.vocab_size = self.alphabet_size + 2

        self.itos = {i + 2: s for i, s in enumerate(self.alphabet)}
        self.itos[self.pad_index] = '-'
        self.itos[self.start_index] = '<start>'
        self.stoi = {s: i for i, s in self.itos.items()}


        self.key_to_seq = defaultdict(str)
        self.keys = []

        have_key = False
        if open(self.alignment_file).read()[0] == '>':
            have_key = True
        fr = open(self.alignment_file, 'r')
        if have_key:
            key = ''
            for row in fr:
                row = row.strip()
                if row.startswith('>'):
                    key = row[1:]
                    self.keys.append(key)
                else:
                    self.key_to_seq[key] += row
        else:
            key = 0
            for row in fr:
                row = row.strip()
                if row != '':
                    self.keys.append(key)
                    self.key_to_seq[key] = row
                    key += 1
        fr.close()

        if type(self.focus_key) is int:
            self.focus_key = self.keys[self.focus_key]
        elif self.focus_key == '' or self.focus_key not in self.keys:
            self.focus_key = self.keys[0]

        self.focus_seq = self.key_to_seq[self.focus_key]
        if self.aa_isupper:
            self.focus_cols = [i for i, s in enumerate(self.focus_seq) if s.isupper()]
        else:
            self.focus_cols = [i for i, s in enumerate(self.focus_seq)]

        self.focus_seq_trimmed = [self.focus_seq[i] for i in self.focus_cols]

    def gen_full_alignment(self):

        # Get only focus columns, upper the seq and replace '.' with '-'
        for key, seq in self.key_to_seq.items():
            seq = seq.replace('.', '-')
            self.key_to_seq[key] = [seq[i].upper() for i in self.focus_cols]

        # Remove sequences with unknown characters
        if self.remove_unknown:
            keys_to_remove = []
            for key, seq in self.key_to_seq.items():
                for s in seq:
                    if s != '-' and s not in self.alphabet:
                        keys_to_remove.append(key)
                        break
            for key in keys_to_remove:
                if key != self.focus_key:
                    del self.key_to_seq[key]

            self.keys = list(self.key_to_seq.keys())

        # sequence to index
        self.nseqs = len(self.key_to_seq)

        seqs = [self.key_to_seq[key] for key in self.keys]

        self.seqs_to_index = []
        for seq in seqs:
            self.seqs_to_index.append(
                [self.start_index] + \
                [self.stoi[s] if s in self.stoi else self.pad_index for s in seq]
            )
        self.seq_len = len(self.seqs_to_index[0]) # total sequence length include the start marker

        self.train_tensors = torch.tensor(self.seqs_to_index,
                                     dtype=torch.long)  # train_seqs to train_tensors

        # Calculate the seq weights
        self.weights = np.ones(self.nseqs)

        if self.calc_weights:
            if self.theta >= 0 and self.theta <= 1:
                self.weights = self.weightCal(np.array(self.seqs_to_index)[:, 1:], self.theta, pad_index=self.pad_index)

        self.weights = torch.tensor(self.weights,
                                     dtype=torch.float32)  # train_seqs to train_tensors
        self.weights = self.weights.unsqueeze(1)
        self.weights *= self.nseqs  # rescale weights by multiply the number of sequences 

        self.Neff = torch.sum(self.weights) / self.nseqs   # the effective numbers

    def one_hots_from_seqs(self, ids=[]):

        if type(ids) is list:
            if ids == []:
                ids = list(range(self.nseqs))
        elif type(ids) is int:
            ids = [ids]

        sequences_to_one_hot = np.zeros((len(ids), self.seq_len, self.vocab_size))

        for i in range(len(ids)):
            for j in range(self.seq_len):
                k = self.seqs_to_index[i][j]
                sequences_to_one_hot[ids[i], j, k] = 1.0

        return np.squeeze(sequences_to_one_hot)

    @staticmethod
    @njit(nogil=True, parallel=True)
    def weightCal(x_train, theta, pad_index=0):

        nseqs, nsites = x_train.shape
        weights = np.ones(nseqs)
        for s in prange(nseqs - 1):
            for t in prange(s + 1, nseqs):
                id = 0
                non_pad_sites = 0
                for i in range(nsites):
                    if x_train[s, i] != pad_index and x_train[t, i] != pad_index:
                        non_pad_sites += 1
                        if x_train[s, i] == x_train[t, i]: id += 1
                if id >= (1. - theta) * non_pad_sites:
                    weights[s] += 1
                    weights[t] += 1

        for n in range(nseqs):
            weights[n] = 1 / weights[n]
        
        return weights

class ResNet_DataLoader():

    def __init__(self, repeats, min_res_delta=5, mask_threshold=8, max_seq_len=np.inf, provide_mask='dist') -> None:
        
        self.r = repeats
        self.min_res_delta = min_res_delta
        self.mask_threshold = mask_threshold
        self.max_seq_len = max_seq_len
        self.provide_mask = provide_mask
        pass

    def get_tm(self, tmdir, dataset, repeats):
        tms = []
        for r in range(repeats):
            tm_npy_path = tmdir + dataset + f'_{r:0>3}.tm.npy'
            tm = np.load(tm_npy_path)
            if tm.shape[-1] > self.max_seq_len:                
                return None
            else:
                tms.append(tm)

        tms = np.array(tms)
        tms_r, tms_h, tms_c, tms_n, tms_m = tms.shape
        tms = tms.reshape((tms_r, tms_h * tms_c, tms_n, tms_m))[:, :, :, :]

        return tms
    
    def get_mask_from_rrdist(self, distfile, min_res_delta, mask_threshold):

        mask = mask_from_rrdist(distfile, min_res_delta=min_res_delta, mask_threshold=mask_threshold)

        return mask

    def get_mask_from_mask(self, maskfile):
        try:
            mask = np.loadtxt(maskfile)
        except:
            mask = np.load(mask)
        
        return mask
    
    def build_ResNet_train_dataloader(self, datasets, rootdir, savedir, limit_one_dataloader=100):
        '''TM npy and its distfile are in the same dir -- rootdir/{dataset}/
        '''

        try:
            os.mkdir(savedir)
        except:
            pass

        dataloader = {}
        dataloader_count = 0
        N = len(datasets)
        for n in range(N):
            dataset = datasets[n]
            loaddir = rootdir + dataset + '/'
            tm = self.get_tm(loaddir, dataset, self.r)
            if tm is None:
                if n == len(datasets) - 1:
                    np.save(savedir + f'train_dataloader_{dataloader_count}_mask{self.mask_threshold}.npy', dataloader, allow_pickle=True)
                    print(f'train_dataloader_{dataloader_count}_mask{self.mask_threshold}.npy', '{} has been saved'.format(len(dataloader))) 
                else:
                    continue
            else:
                
                if self.provide_mask == 'mask':
                    maskfile = loaddir + dataset + '.mask'
                    mask = self.get_mask_from_mask(maskfile)
                elif self.provide_mask == 'dist':
                    maskfile = loaddir + dataset + '.dist'
                    mask = self.get_mask_from_rrdist(maskfile, self.min_res_delta, self.mask_threshold)
                
                if len(dataloader) < limit_one_dataloader - 1 and n != (len(datasets) - 1):
                    dataloader[dataset] = (tm, mask)
                else:
                    dataloader[dataset] = (tm, mask)
                    np.save(savedir + f'train_dataloader_{dataloader_count}_mask{self.mask_threshold}.npy', dataloader, allow_pickle=True)
                    print(f'train_dataloader_{dataloader_count}_mask{self.mask_threshold}.npy', '{} has been saved'.format(len(dataloader)))
                    dataloader_count += 1
                    dataloader = {}

    def build_ResNet_tm_dataloader(self, datasets, rootdir, savedir, limit_one_dataloader=100):
        '''TM npy in rootdir/{dataset}/
        '''

        try:
            os.mkdir(savedir)
        except:
            pass

        dataloader = {}
        dataloader_count = 0
        N = len(datasets)
        for n in range(N):
            dataset = datasets[n]
            loaddir = rootdir + dataset + '/'
            tm = self.get_tm(loaddir, dataset, self.r)
            if tm is None:
                if n == len(datasets) - 1:
                    np.save(savedir + f'tm_dataloader_{dataloader_count}_mask{self.mask_threshold}.npy', dataloader, allow_pickle=True)
                    print(f'tm_dataloader_{dataloader_count}_mask{self.mask_threshold}.npy', '{} has been saved'.format(len(dataloader))) 
                else:
                    continue
            else:                
                
                if len(dataloader) < limit_one_dataloader - 1 and n != (len(datasets) - 1):
                    dataloader[dataset] = tm
                else:
                    dataloader[dataset] = tm
                    np.save(savedir + f'tm_dataloader_{dataloader_count}_mask{self.mask_threshold}.npy', dataloader, allow_pickle=True)
                    print(f'tm_dataloader_{dataloader_count}_mask{self.mask_threshold}.npy', '{} has been saved'.format(len(dataloader)))
                    dataloader_count += 1
                    dataloader = {}
  