import torch 
import torch.nn as nn
from torch.nn import functional as F
import time
import pandas as pd
import numpy as np
import copy
from time import strftime, gmtime
import os, sys

# ResNet
class Conv_Residual(nn.Module):
    def __init__(self, in_channels, out_channels, acfunc, stride=1, use_1x1conv=False) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=9, padding=4, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.ac1 = acfunc

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=5, padding=2, stride=stride)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.ac2 = acfunc

        if use_1x1conv:
            self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=1, stride=stride)  
        else:
            self.conv0 = None       

    def forward(self, X):
        Y = self.ac1(self.bn1(self.conv1(X)))

        if self.conv0:
            X = self.conv0(X)
        
        Y = self.bn2(self.conv2(Y))
        Y = self.ac2(Y + X)

        return Y

# Block
def Conv_block(in_channels, out_channels, num_residuals, acfunc, first_block=False):
    blocks = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blocks.append(Conv_Residual(in_channels, out_channels, acfunc, stride=1, use_1x1conv=True))
        
        else:
            blocks.append(Conv_Residual(out_channels, out_channels, acfunc, stride=1, use_1x1conv=False))
    
    return blocks

class ResNet(nn.Module):
    def __init__(self, r) -> None:
        super().__init__()
        
        self.init_conved = nn.Sequential(nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(8), nn.GELU())
        self.encode1 = nn.Sequential(*Conv_block(8, 8, 4, acfunc=nn.GELU(), first_block=True))
        self.encode2 = nn.Sequential(*Conv_block(8, 1, 4, acfunc=nn.GELU(), first_block=False))

        self.end_conved = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(1), nn.GELU())

        self.linear = nn.Linear(r, 1)

        self.activation = nn.LogSigmoid()


    def forward(self, x):
        x = x.abs()
        x = x[:, :, 1:, 1:] - x[:, :, 0, 0][:, :, np.newaxis, np.newaxis]
        x = (x - torch.mean(x, dim=-1, keepdim=True)) / (torch.std(x, dim=-1, keepdim=True) + 1e-9)

        encode_out = x.permute(0, 1, 3, 2)
        encode_out = self.init_conved(encode_out)
        encode_out = self.encode1(encode_out)
        encode_out = self.encode2(encode_out)
        encode_out = self.end_conved(encode_out)

        decode_out = encode_out.permute(2, 3, 1, 0)
        decode_out = self.linear(decode_out)
        decode_out = decode_out.squeeze(dim=-1).squeeze(dim=-1)
        decode_out = (decode_out + decode_out.T) / 2

        if self.activation is not None:
            decode_out = self.activation(decode_out)
            decode_out = (decode_out + decode_out.T) / 2

        return decode_out

# Init weights
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight, gain=1.5)  # useful
    
# Optimizer
def optim(model, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-9,
                                 weight_decay=1e-2)  # weight_decay is useful
    return optimizer

# Schedule
class ScheduleOptim():

    def __init__(self, optimizer, init_lr, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = init_lr

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

# Loss Function
loss_object = nn.CrossEntropyLoss()

# Train
def train_step(model, optim_schedule, src, tgt, device):

    src, tgt = src.to(device), tgt.to(device)
    model.train()

    optim_schedule.zero_grad()

    predictions = model(src)

    loss = loss_object(predictions, tgt)

    loss.mean().backward()
    optim_schedule.step_and_update_lr()

    return loss.item()

def prec_step(model, src, device):

    src = src.to(device)
    model.eval()

    predictions = model(src)

    return predictions.detach().cpu().numpy()

# multi dataloaders and no eval_dataloader
def Train(model, optim_schedule, train_dataloader_paths, epochs, device,
          checkpoint_dir='./', logfile='log.csv', printepoch=True):
    # init weights
    init_weights(model)

    # 4. train
    train_start_time = time.time()
    df_history = pd.DataFrame(columns=['epoch', 'lr', 'loss'])
    torch.manual_seed(np.random.randint(1000000))
    for epoch in range(epochs):
        
        epoch_start_time = time.time()

        train_loss_sum = 0.
        train_batch = 0
        for tdpath in train_dataloader_paths:
            train_dataloader = np.load(tdpath, allow_pickle=True).item().values()

            for src_tensors, tgt_tensors in train_dataloader:
                src_tensors, tgt_tensors = torch.tensor(src_tensors, dtype=torch.float32), torch.tensor(tgt_tensors, dtype=torch.float32)
                train_loss = train_step(model, optim_schedule, src_tensors, tgt_tensors, device)
                train_loss_sum += train_loss
                train_batch += 1


        lr = optim_schedule.get_last_lr()[0]
        record = (epoch, lr, train_loss_sum / train_batch) 
        df_history.loc[epoch] = record

        # Save Model
        current_loss_avg = train_loss_sum / train_batch
        epoch_time = time.time() - epoch_start_time
        if printepoch:
            print(f'TRAIN '
                  f'| epoch {epoch:3d} | time {epoch_time:5.2f}s '
                  f'| lr {lr:5.3f} '
                  f'| train_loss {current_loss_avg:5.2f} '                 
                )

    # save the end epoch
    model_sd = copy.deepcopy(model.state_dict())
    torch.save({
        'loss': train_loss_sum,
        'epoch': epoch,
        'net': model_sd,
    }, checkpoint_dir + f'epoch_{epoch}.tar')


    time_elapsed = time.time() - train_start_time
    df_history.to_csv(logfile, index=False)
    print(f'TRAIN_END '
          f'| training_complete_in ', strftime('%H:%M:%S', gmtime(time_elapsed)),
        )
