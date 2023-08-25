import os
import sys
import torch
from etld.dataload import DataLoader
from etld.model import ETLD, Optim, ScheduleOptim, Train
from etld.logger import Logger

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set which gpu to use
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
batch_size = 64

# save dir
dataset = 'dyr_ecoli'
save_dir = './dyr_ecoli/'
msapath = './dyr_ecoli.fa'
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
        alignment_file=msapath,
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
for r in range(model_params['r']):
    checkpoint = save_dir + '{}_{:0>3d}.tar'.format(dataset, r)
    trainlog = save_dir + '{}_{:0>3d}.csv'.format(dataset, r)

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
    # 4. train
    optimizer = Optim(etld, lr=0.001)
    optim_schedule = ScheduleOptim(
            optimizer,
            d=model_params['d'],
            n_warmup_steps=100
        )
    Train(
        etld, optim_schedule, dataloader,
        batch_size=batch_size,
        epochs=model_params['e'],
        checkpoint=checkpoint,
        trainlog=trainlog,
        printepoch=False
    )

