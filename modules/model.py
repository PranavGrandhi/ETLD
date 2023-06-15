import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
from time import strftime, gmtime
import random
import copy

# 1. Create a mask for TM
def creat_TM_mask(src_seq_len, tgt_seq_len, windowsize=1):
    """Create a mask for the transformation matrix whhen the sorce MSA and target MSA are the same.

    Default windowsize = 1 to keep the diagonal value constant at 0.

    """
    if src_seq_len != tgt_seq_len or windowsize is None:
        mask = np.ones((src_seq_len, tgt_seq_len))
        return mask
    
    else:
        half_window = windowsize // 2
        mask = np.ones((src_seq_len, tgt_seq_len))
        for i in range(src_seq_len):
            if i - half_window < 0:

                mask[i, 0:(half_window + 1)] = 0
            elif i + half_window >= src_seq_len:
                mask[i, i - half_window:] = 0

            else:
                mask[i, (i - half_window): (i + half_window + 1)] = 0

        return mask[np.newaxis, np.newaxis, :, :]

# 2. Create masks for padding
def create_padding_mask(seq, pad_index=0):
    """seq: [batch_size, seq_len]

    Default pad_index = 0

    """

    seq = torch.not_equal(seq, torch.tensor(pad_index))

    return seq[:, :, np.newaxis]

# 3. Embedding
class TokenEmbedding(nn.Embedding):  # [batch_size, seq_len] -> [batch_size, seq_len, embed_size]
    def __init__(self, vocab_size, embed_size=512, pad_index=0):
        super(TokenEmbedding, self).__init__(vocab_size, embed_size, pad_index)
    
class HybridEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_size, dropout=0.1, pad_index=0):

        super(HybridEmbedding, self).__init__()

        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size, pad_index=pad_index)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence):  # [batch_size, seq_len]-> [b, seq_len, embed_size]

        x = self.token(sequence)

        return self.dropout(x)

# 4. Point Wise Feed Forward Network
class PointWiseFeedForwardNetwork(nn.Module):
    """Refer to FFN in Transformer[1].

    [1] Vaswani, Ashish , et al. "Attention Is All You Need." arXiv (2017).

    """
    def __init__(self, d, d_inner, dropout=0.1):

        super(PointWiseFeedForwardNetwork, self).__init__()

        self.sequential = nn.Sequential(
            nn.Linear(d, d_inner),  # [batch_size, seq_len, d] -> [batch_size, seq_len, d_inner]
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d),  # [batch_size, seq_len, d_inner] -> [batch_size, seq_len, d]
        )

    def forward(self, x):

        return self.sequential(x)

# 5. Sublayer Connection
class SublayerConnection(nn.Module):
    """A residual connection followed by a layer norm"""

    def __init__(self, d, dropout=0.1):
        super(SublayerConnection, self).__init__()

        self.norm = nn.LayerNorm(normalized_shape=d, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer=None):
        if sublayer is not None:
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            return x + self.dropout(self.norm(x))

# 6. Encoder
class Encoder(nn.Module):
    def __init__(self, d, d_inner, src_vocab_size,
                 pad_index=0, dropout=0.1):
        super(Encoder, self).__init__()

        self.embed_size = d
        self.embedding = HybridEmbedding(
            vocab_size=src_vocab_size,
            embed_size=d,
            dropout=0.1,
            pad_index=pad_index
        )

        self.ffn = PointWiseFeedForwardNetwork(d=d, d_inner=d_inner, dropout=dropout)

        self.sublayer = SublayerConnection(d=d, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, 
                src_padding_mask=None):  # src: [batch_size, src_seq_len, embed_size]; src_padding_mask: [batch_size, 1, 1, src_seq_len]

        src = self.embedding(src)  # [batch_size,src_seq_len] -> [batch_size, src_seq_len, embed_size]
        src *= torch.sqrt(torch.tensor(self.embed_size, dtype=torch.float32))    # weighted by sqrt(d)

        if src_padding_mask is not None:
            src *= src_padding_mask

        x = src
        x = self.sublayer(x, self.ffn)

        return self.dropout(x)

# 7. Decoder
class Decoder(nn.Module):
    def __init__(self, d, d_inner, dropout=0.1):
        super(Decoder, self).__init__()

        self.ffn = PointWiseFeedForwardNetwork(d=d, d_inner=d_inner, dropout=dropout)
        self.sublayer = SublayerConnection(d=d, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input):
        x = dec_input  # [batch_size, tgt_seq_len, embed_size]

        x = self.sublayer(x, self.ffn)

        return self.dropout(x)

# 8. ETLD
class ETLD(nn.Module):
    """ETLD Model

    Parameters:

    d: embedding vector dimension
    d_inner: the inner-layer dimension of FFN
    h: the number of multi-heads
    src_vocab_size: vocab size of the source MSA
    tgt_vocab_size: vocab size of the target MSA
    src_seq_len: sequence length in source MSA
    tgt_seq_len: sequence length in target MSA
    device: Pytorch work environment, default: torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    c=2: the transformation layer channel. c=0: linear; c=1: non-linear; c=2: both linear and non-linear
    caa=2: the amino acid transformation layer channel. caa=-1: no; caa=0: linear; caa=1: non-linear; caa=2: both linear and non-linear
    pad_index=0: padding index
    dropout=0.1: drouout rate

    """

    def __init__(self, d, d_inner, h,
                 src_vocab_size, tgt_vocab_size,
                 src_seq_len, tgt_seq_len,
                 device,
                 c=2,
                 caa=2,
                 pad_index=0,
                 dropout=0.1,              
                 ):

        super(ETLD, self).__init__()

        self.device = device

        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.tm_channel = c
        self.multi_head = h
        self.embed_size = d
        self.depth = int(self.embed_size / self.multi_head)

        self.tgt_vocab_size = tgt_vocab_size

        self.aatm_channel = caa

        self.encoder = Encoder(
            d, d_inner, src_vocab_size,
            pad_index, dropout
        )

        self.decoder = Decoder(
            d, d_inner, dropout
        )

        self.linear = nn.Linear(d, tgt_vocab_size)

        self.logsoftmax = nn.LogSoftmax(dim=-1)

        if self.tm_channel == 0 or self.tm_channel == 1:
            self.TM = nn.Parameter(torch.empty(size=(1, self.multi_head, src_seq_len, tgt_seq_len)))
            nn.init.xavier_uniform_(self.TM.data, gain=1.414)
        elif self.tm_channel == 2:
            self.TM = nn.Parameter(torch.empty(size=(2, self.multi_head, src_seq_len, tgt_seq_len)))
            nn.init.xavier_uniform_(self.TM.data, gain=1.414)

        nn.init.xavier_uniform_(self.TM.data, gain=0.)

        if self.aatm_channel == 0 or self.aatm_channel == 1:
            self.AATM = nn.Parameter(torch.empty(size=(1, tgt_vocab_size, tgt_vocab_size)))
            nn.init.xavier_uniform_(self.AATM.data, gain=0.5)

        elif self.aatm_channel == 2:
            self.AATM = nn.Parameter(torch.empty(size=(2, tgt_vocab_size, tgt_vocab_size)))
            nn.init.xavier_uniform_(self.AATM.data, gain=0.5)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_padding_mask=None, TM_mask=None):

        # src: [batch_size, src_seq_len]; tgt: [batch_size, tgt_seq_len]
        # src_mask: [batch_size, 1, 1, src_seq_len]; tgt_mask: [batch_size, 1, 1, tgt_seq_len]
        
        batch_size, _ = src.size()

        TM = self.TM   # [2, multi_head, src_seq_len, tgt_seq_len]
        if TM_mask is not None:
            TM = TM * TM_mask

        enc_output = self.encoder(src, src_padding_mask)  # [batch_size, src_seq_len, embed_size]
        enc_output = enc_output.view(batch_size, self.src_seq_len, self.multi_head, self.depth)
        
        if self.tm_channel == 0:
            # [1, batch_size, src_seq_len, multi_head, depth]
            enc_output = enc_output.view(1, batch_size, self.src_seq_len, self.multi_head, self.depth)
        elif self.tm_channel == 1:
            # [1, batch_size, src_seq_len, multi_head, depth]
            enc_output = F.gelu(enc_output)
            enc_output = enc_output.view(1, batch_size, self.src_seq_len, self.multi_head, self.depth)
        elif self.tm_channel == 2:
            # [2, batch_size, src_seq_len, multi_head, depth]
            enc_output = torch.stack([F.gelu(enc_output), enc_output], dim=0)
        enc_output = enc_output.permute(1, 0, 3, 4, 2) # [batch_size, 2, multi_head, depth, src_seq_len]

        dec_input = torch.matmul(enc_output, TM)
        dec_input = dec_input.permute(1, 0, 4, 2, 3)  # [2, batch_size, tgt_seq_len, multi_head, depth]
        dec_input = torch.sum(dec_input, dim=0)   # [batch_size, tgt_seq_len, multi_head, depth]
        dec_input = dec_input.view(batch_size, self.tgt_seq_len, -1) # [batch_size, tgt_seq_len, embed_size]

        dec_output = self.decoder(dec_input)

        linear_output = self.linear(dec_output) # [batch_size, tgt_seq_len, tgt_vocab_size]

        if self.aatm_channel in [0, 1, 2]:
            if self.aatm_channel  == 0:
                linear_output = linear_output.view(1, batch_size, self.tgt_seq_len, self.tgt_vocab_size)
            elif self.aatm_channel  == 1:
                linear_output = F.gelu(linear_output)
                linear_output = linear_output.view(1, batch_size, self.tgt_seq_len, self.tgt_vocab_size)
            elif self.aatm_channel  == 2:
                linear_output = torch.stack([F.gelu(linear_output), linear_output])

            linear_output = linear_output.permute(1, 0, 2, 3) # [batch_size, 2, tgt_seq_len, tgt_vocab_size]
            linear_output = torch.matmul(linear_output, self.AATM,)
            linear_output = torch.sum(linear_output, dim=1)

        return self.logsoftmax(linear_output), tgt  # [batch_size, tgt_seq_len, tgt_vocab_size], [batch_size, tgt_seq_len]


# 9. Optimizer and LrSchedule
def Optim(model, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-9,
                                 weight_decay=1e-2)  # weight_decay is useful
    return optimizer

class CustomSchedule(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, d, warm_steps=4000):
        self.optimizer = optimizer
        self.d = d
        self.warm_steps = warm_steps

        super(CustomSchedule, self).__init__(optimizer)

    def get_lr(self) -> float:
        arg1 = self._step_count ** (-0.5)
        arg2 = self._step_count * (self.warm_steps ** -1.5)
        dynamic_lr = (self.d ** (-0.5)) * min(arg1, arg2)

        return [dynamic_lr for group in self.optimizer.param_groups]

class ScheduleOptim():

    def __init__(self, optimizer, d, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d, -0.5)

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps
        ])

    def _update_learning_rate(self):
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def get_last_lr(self):
        return [group['lr'] for group in self._optimizer.param_groups]


# 10. Loss And Accuracy
loss_object = nn.NLLLoss(ignore_index=0, reduction='none')  

def mask_loss_func(pred, tgt, weights, pad_index=0):  # [batch_size, tgt_seq_len], [batch_size, tgt_seq_len, tgt_vocab_size]

    _loss = loss_object(pred.transpose(1, 2), tgt)  # [batch_size, tgt_seq_len]; Note, pred is former, and target is later
    _loss = _loss * weights

    mask = torch.logical_not(tgt.eq(pad_index)).type(_loss.dtype)  # [batch_size, tgt_seq_len]
    _loss *= mask

    return _loss.sum() / mask.sum().item()

def mask_acc_func(pred, tgt, pad_index=0):  # [batch_size, tgt_seq_len], [batch_size, tgt_seq_len, tgt_vocab_size]

    _pred = pred.argmax(dim=-1)  # [batch_size, tgt_seq_len, tgt_vocab_size] -> [batch_size, tgt_seq_len]
    corrects = _pred.eq(tgt)  # [batch_size, tgt_seq_len] dtype: bool

    mask = torch.logical_not(tgt.eq(pad_index))  # [b, tgt_seq_len] dtype: bool

    corrects *= mask

    return corrects.sum().float() / mask.sum().item()

# 11. Train

def train_step(model, optim_schedule, src, tgt, weights, TM_mask, pad_index):

    device = model.device

    src, tgt, weights = src.to(device), tgt.to(device), weights.to(device)

    src_padding_mask = create_padding_mask(src, pad_index).to(device)

    model.train()

    optim_schedule.zero_grad()

    predictions, tgt = model(src, tgt, src_padding_mask, TM_mask)
    loss = mask_loss_func(predictions, tgt, weights, pad_index)
    accuracy = mask_acc_func(predictions, tgt, pad_index)

    loss.backward()
    optim_schedule.step_and_update_lr()

    # limit TM to a given range
    # for n, p in model.named_parameters():
    #    if n == 'TM':
    #        p.data.clamp_(-1., 1.)

    return loss.item(), accuracy.item()


def Train(model, optim_schedule, dataloader, checkpoint,
          batch_size, epochs, trainlog='log.csv', printepoch=False):
    
    """Here the source MSA and the target MSA are the same.
    Then the target sequence is the input sequence.
    """

    prename = dataloader.dataset
    base_src_tensors = dataloader.train_tensors
    base_weights = dataloader.weights
    seq_len = dataloader.seq_len
    times_of_seq_len = dataloader.times_of_seq_len

    if times_of_seq_len <= 0:
        seq_nums_of_each_epoch = dataloader.nseqs
    else:
        seq_nums_of_each_epoch = int(times_of_seq_len * seq_len)

    # 1. create TM_mask, which can be None
    TM_mask = creat_TM_mask(src_seq_len=dataloader.seq_len, tgt_seq_len=dataloader.seq_len)
    TM_mask = torch.tensor(TM_mask).to(model.device, dtype=torch.float32)

    # 2. train
    train_start_time = time.time()
    df_history = pd.DataFrame(columns=['epoch', 'lr', 'loss', 'acc'])

    pad_index = dataloader.pad_index
    best_acc = 0.

    idx = list(np.arange(base_src_tensors.size(0)))
    
    
    for epoch in range(epochs):
        torch.manual_seed(np.random.randint(1000000))
        epoch_start_time = time.time()

        train_loss_sum = 0.
        train_acc_sum = 0.
        if dataloader.nseqs > seq_nums_of_each_epoch:
            select_idx = random.sample(idx, seq_nums_of_each_epoch)
            src_tensors = base_src_tensors[select_idx]
            weights = base_weights[select_idx]
            idx = list(np.arange(seq_nums_of_each_epoch))
        else:
            src_tensors = base_src_tensors
            weights = base_weights
        np.random.shuffle(idx)
        
        tgt_tensors = src_tensors
        for train_batch, i in enumerate(range(0, src_tensors.size(0), batch_size), start=1):
            batch_index = idx[i:i + batch_size]
            train_loss, train_acc = train_step(model, optim_schedule, src_tensors[batch_index, :],
                                                   tgt_tensors[batch_index, :], weights[batch_index], TM_mask, pad_index)

            train_loss_sum += train_loss
            train_acc_sum += train_acc
        train_loss_sum /= dataloader.Neff

        lr = optim_schedule.get_last_lr()[0]
        record = (epoch, lr, train_loss_sum / train_batch, train_acc_sum / train_batch)
        df_history.loc[epoch] = record

        # Save Model
        current_acc_avg = train_acc_sum / train_batch
        current_loss_avg = train_loss_sum / train_batch

        epoch_time = time.time() - epoch_start_time
        if printepoch:
            print(f'TRAIN {prename} '
                  f'| epoch {epoch:3d} | time {epoch_time:5.2f}s '
                  f'| lr {lr:5.3f} '
                  f'| train_loss {current_loss_avg:5.2f} '
                  f'| train_acc {current_acc_avg:5.2f} ')
        if current_acc_avg > best_acc:

            model_sd = copy.deepcopy(model.state_dict())

            torch.save({
                'loss': train_loss_sum / train_batch,
                'epoch': epoch,
                'net': model_sd,
            }, checkpoint)

            best_acc = current_acc_avg

    time_elapsed = time.time() - train_start_time
    df_history.to_csv(trainlog, index=False)
    print(f'TRAIN_END {prename} '
          f'| training_complete_in ', strftime('%H:%M:%S', gmtime(time_elapsed)),
          f'| acc {best_acc:5.2f}')


# 12. Load Model
def Loadmodel(checkpoint, model):
    ckpt = torch.load(checkpoint)
    model.load_state_dict(ckpt['net'])
    return model

