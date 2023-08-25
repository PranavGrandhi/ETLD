import os
import sys
sys.path.append('../modules')
import torch
from resnet import ResNet, Optim, ScheduleOptim, Train
from logger import Logger

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set which gpu to use

# Pytorch work environment
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dir
dataloader_dir = './dataloader/'    # dir of training dataloader
train_dataloader_paths = [dataloader_dir + f'train_dataloader_{n}_mask8.npy' for n in range(6)] # path of training dataloader

save_dir = './ck/'
try:
    os.mkdir(save_dir)
except:
    pass
logfile = save_dir + 'run_example.log'
sys.stdout = Logger(logfile, sys.stdout, mode='w')   # add or rewrite logfile
print('ROOT_DIR', os.path.abspath(save_dir))
# Randomly train the model 5 times under different training initial conditions    
for n in range(5):

    sub_save_dir = save_dir + f'{n}/'
    try:
        os.mkdir(sub_save_dir)
    except:
        pass
    print('Traing {}; Checkpoint save: {}'.format(n, sub_save_dir))

    # train
    model = ResNet()
    model.to(device=device)
    optimizer = Optim(model.parameters(), lr=0.01)
    optim_schedule = ScheduleOptim(
                optimizer,
                0.1,
                n_warmup_steps=0.1
    )

    Train(
        model, 
        optim_schedule, 
        train_dataloader_paths, 
        epochs=20,
        device=device,
        checkpoint_dir=save_dir,
        logfile=save_dir + 'log.csv', 
        printepoch=True, 
        )
    del optimizer
    del model

