import numpy as np
from collections import defaultdict
import numba as nb
from numba import njit
from numba import prange
import torch

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





