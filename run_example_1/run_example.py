import os
import sys
sys.path.append('../modules')
import torch
from dataload import DataLoader
from model import ETLD, Optim, ScheduleOptim, Train
from logger import Logger

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set which gpu to use

# save dir
dataset = 'dhfr'
save_dir = './run/'
try:
    os.mkdir(save_dir)
except:
    pass
logfile = save_dir + 'run_example.log'

# Train
sys.stdout = Logger(logfile, sys.stdout, mode='w')   # add or rewrite logfile
print('ROOT_DIR', os.path.abspath(save_dir))

## dataloader
dataloader = DataLoader(
        dataset=dataset,
        alignment_file='./dhfr.fa',
        focus_key='',
        calc_weights=True,
        theta=0.2,
        times_of_seq_len=0,
        aa_isupper=False,
        remove_unknown=False,
)
# Pytorch work environment
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# repeats = 10, epochs = 10
for r in range(10):
    checkpoint = save_dir + '{}_{:0>3d}.tar'.format(dataset, r)
    trainlog = save_dir + '{}_{:0>3d}.csv'.format(dataset, r)

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

    etld = etld.to(device)
    # 4. train
    optimizer = Optim(etld, lr=0.001)
    optim_schedule = ScheduleOptim(
            optimizer,
            d=256,
            n_warmup_steps=100
        )
    Train(
        etld, optim_schedule, dataloader,
        batch_size=64,
        epochs=10,
        checkpoint=checkpoint,
        trainlog=trainlog,
        printepoch=True
    )

