import os
import numpy as np
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
import dask
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from torch import optim
from timeit import default_timer

from dataset import *
from model import *
from metrics import *

import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_dataset(args):
    dataset = args['dataset']
    train_dataset = STDataset(
                 region_name = dataset['region_name'],
                 folder_path= dataset['folder_path'],
                 reference_file = dataset['reference_file'], 
                 label_path = dataset['label_path'],
                 lat_min = dataset['lat_min'],
                 lat_max = dataset['lat_max'], 
                 lon_min = dataset['lon_min'],
                 lon_max = dataset['lon_max'],
                 challenge = args['challenge_name'],
                 add_time = True, 
                 if_train=True,
                 seq_len=dataset['seq_len']
                 )
    test_dataset = STDataset(
                 region_name = dataset['region_name'],
                 folder_path= dataset['folder_path'],
                 reference_file = dataset['reference_file'], 
                 label_path = dataset['label_path'],
                 lat_min = dataset['lat_min'],
                 lat_max = dataset['lat_max'], 
                 lon_min = dataset['lon_min'],
                 lon_max = dataset['lon_max'],
                 challenge = args['challenge_name'],
                 add_time = True, 
                 if_train=False,
                 seq_len=dataset['seq_len']
                 )
    
    return train_dataset, test_dataset

def get_loader(train_dataset, test_dataset, args):

    loader = args['dataloader']

    train_size = int(len(train_dataset) * 0.9)
    val_size = len(train_dataset) - train_size 
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=loader['train_bs'], shuffle=False, num_workers=loader['num_workers'])
    validate_loader = DataLoader(val_dataset, batch_size=loader['val_bs'], shuffle=False, num_workers=loader['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=loader['test_bs'], shuffle=False, num_workers=loader['num_workers'])

    return train_loader, validate_loader, test_loader


def process_data(inputs, targets, k, lstm = False):
    '''
    downsample to 1*1

    reshape torch.Size([bs, var, lat, lon]) --> torch.Size([bs, n])
    '''
    if lstm:
        # print('input, target: ', inputs.shape, targets.shape)
        # 下采样前将lstm的5维张量换为4维
        bs_1, seq, var, lat, lon = inputs.shape
        bs_2, depth = targets.shape[0], targets.shape[2]
        inputs = inputs.view(bs_1 * seq, var, lat, lon)
        targets = targets.view(bs_2 * seq, depth, lat, lon)

    # 下采样到0.5*0.5
    # inputs, targets, new_lat, new_lon = down_sample(inputs, targets)

    # reshape
    if lstm:
        # (bs, n, seq, var)
        inputs = inputs.view(bs_1, seq, var, lat, lon)  # 还原为原始的 5D 形式
        targets = targets.view(bs_2, seq, depth, lat, lon)
        inputs = inputs.reshape(bs_1, seq, -1)
        targets = targets[:, :, 0:k, ...].reshape(bs_2, seq, -1)
    else:
        # (bs, n)
        bs_1 = inputs.shape[0]
        bs_2 = targets.shape[0]
        inputs = inputs.reshape(bs_1, -1)
        targets = targets[:, 0:k, ...].reshape(bs_2, -1)
    

    return inputs, targets

def down_sample(inputs, targets):
    '''
    0.25*0.25下采样到1*1

    in: (bs, var, lat, lon)
    out: (bs, var, lat, lon)
    '''
    # 下采样到1*1
    if_reshape = False
    if inputs.dim() == 5:
        # rehsape为插值所需的四维：(bs, seq, var, lat, lon) --> (bs*seq, var, lat, lon)
        if_reshape = True
        bs_1, seq, var, lat, lon = inputs.shape
        bs_2, depth = targets.shape[0], targets.shape[2]
        inputs = inputs.view(bs_1 * seq, var, lat, lon)
        targets = targets.view(bs_2 * seq, depth, lat, lon)

    lat, lon = inputs.shape[-2], inputs.shape[-1]
    new_lat, new_lon = int(lat / 2), int(lon / 2)  # 目标尺寸
    new_size = (new_lat, new_lon)
    inputs = F.interpolate(inputs, size=new_size, mode='bilinear', align_corners=False)  # `mode='bilinear'` 对于2D数据
    targets = F.interpolate(targets, size=new_size, mode='bilinear', align_corners=False)

    if if_reshape:
        # reshape回原形式：(bs*seq, var, lat, lon) --> (bs, seq, var, lat, lon)
        inputs = inputs.view(bs_1, seq, var, new_lat, new_lon)  # 还原为原始的 5D 形式
        targets = targets.view(bs_2, seq, depth, new_lat, new_lon)

    return inputs, targets, new_lat, new_lon

def train_model(model, train_loader, validate_loader, criterion, device, args):
    '''
    train model

    k: depth
    pat: early stopping patience
    '''
    n_epochs = args['epochs']
    k = args['model']['out_dim']

    train_losses = []  # loss history
    val_losses = []
    min_val_loss = torch.inf  # best validation loss
    total_time = 0  # total training time
    pat = args['pat']

    opt = args['optimizer']['name']
    lr = args['optimizer']['lr']
    optimizer = getattr(optim, opt)(model.parameters(), lr=lr)

    model_name = args['model']['model_name']
    saved_model_name = args["challenge_name"] + '_' + args['dataset']['region_name'] + '_' + model_name
    saved_path = os.path.join(args['saved_dir'], saved_model_name)
    
    for epoch in tqdm(range(n_epochs)):
        model.train()
        t1 = default_timer()  # start time
        running_loss = 0.0

        # start training
        for inputs, targets, _, _, _ in train_loader:
            if model_name == 'FNN':
                inputs, targets = process_data(inputs, targets, k, lstm = False)  # [bs, var, lat, lon] --> [bs, n]
            elif model_name == 'LSTM':
                inputs, targets = process_data(inputs, targets, k, lstm = True)   # [bs, seq, var, lat, lon] --> [bs, seq, n]  torch.Size([1, 5, 16200])
            elif model_name == 'UNET':
                inputs, targets, _, _ = down_sample(inputs, targets)  # [bs, var, new_lat, new_lon]  torch.Size([1, 12, 27, 50]
                inputs, targets = inputs, targets[:, 0:k, ...]
            elif model_name == 'Earthformer':  # [1, 5, 12, 108, 200]
                inputs, targets, _, _ = down_sample(inputs, targets)
                inputs, targets = inputs.permute(0, 1, 3, 4, 2), targets[:, 0:k, ...].permute(0, 1, 3, 4, 2)
            else:
                raise NotImplementedError
            
            inputs, targets = inputs.to(device) , targets.to(device)  
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        t2 = default_timer()  # end time
        total_time += (t2-t1)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # start validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, _, _, _ in validate_loader:
                if model_name == 'FNN':
                    inputs, targets = process_data(inputs, targets, k, lstm = False)   # [bs, var, lat, lon] --> [bs, n]
                elif model_name == 'LSTM':
                    inputs, targets = process_data(inputs, targets, k, lstm = True)   # [bs, var, lat, lon] --> [bs, n]
                elif model_name == 'UNET':
                    inputs, targets, _, _ = down_sample(inputs, targets)  
                    inputs, targets = inputs , targets[:, 0:k, ...]
                elif model_name == 'Earthformer':
                    inputs, targets, _, _ = down_sample(inputs, targets)
                    inputs, targets = inputs.permute(0, 1, 3, 4, 2), targets[:, 0:k, ...].permute(0, 1, 3, 4, 2)
                else:
                    raise NotImplementedError

                inputs, targets = inputs.to(device) , targets.to(device)  
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(validate_loader.dataset)
        val_losses.append(val_loss)

        # save best model
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save({"epoch": epoch + 1, "loss": min_val_loss,
                # "model_state_dict": model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
                }, saved_path + "-best.pt")

        print(f'Epoch {epoch+1}/{n_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Time: {t2-t1:.2f}s')
        if epoch > pat and all(val_loss >= loss for loss in val_losses[-pat:]):
            print('Early stopping triggered')
            break

    # save loss history
    loss_history = np.array(train_losses)
    if not os.path.exists('./log/loss/'):
        os.makedirs('./log/loss/')
    np.save('./log/loss/' + args["challenge_name"] + '_' + args['dataset']['region_name'] + '_' + model_name + '_loss_history.npy', loss_history)
    print("avg_time : {0:.5f}".format(total_time / (args["epochs"])))
    

def train_model_2(model, train_loader, validate_loader, criterion, device, args):
    '''
    train model

    k: depth
    pat: early stopping patience
    '''
    n_epochs = args['epochs']
    k = args['model']['out_dim']

    train_losses = []  # loss history
    val_losses = []
    min_val_loss = torch.inf  # best validation loss
    total_time = 0  # total training time
    pat = args['pat']

    opt = args['optimizer']['name']
    lr = args['optimizer']['lr']
    optimizer = getattr(optim, opt)(model.parameters(), lr=lr)

    model_name = args['model']['model_name']
    saved_model_name = args["challenge_name"] + '_' + args['dataset']['region_name'] + '_' + model_name
    saved_path = os.path.join(args['saved_dir'], saved_model_name)
    
    for epoch in tqdm(range(n_epochs)):
        model.train()
        t1 = default_timer()  # start time
        running_loss = 0.0

        # start training
        for inputs, targets, mask, _, _ in train_loader:
            inputs, targets, mask = inputs.to(device) , targets.to(device), mask.to(device)
            optimizer.zero_grad()
            loss, pred, info = model.train_one_step(inputs, targets, mask, criterion, k)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        t2 = default_timer()  # end time
        total_time += (t2-t1)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # start validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, mask, _, _ in validate_loader:
                inputs, targets, mask = inputs.to(device) , targets.to(device), mask.to(device)
                loss, pred, info = model.train_one_step(inputs, targets, mask, criterion, k)

                val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(validate_loader.dataset)
        val_losses.append(val_loss)

        # save best model
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save({"epoch": epoch + 1, "loss": min_val_loss,
                # "model_state_dict": model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
                }, saved_path + "-best.pt")

        print(f'Epoch {epoch+1}/{n_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Time: {t2-t1:.2f}s')
        if epoch >= pat and all(val_loss >= loss for loss in val_losses[-pat:]):
            print('Early stopping triggered')
            break

    # save loss history
    loss_history = np.array(train_losses)
    if not os.path.exists('./log/loss/'):
        os.makedirs('./log/loss/')
    np.save('./log/loss/' + args["challenge_name"] + '_' + args['dataset']['region_name'] + '_' + model_name + '_loss_history.npy', loss_history)
    print("avg_time : {0:.5f}".format(total_time / (args["epochs"])))
