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
import csv

from dataset import *
from model import *
from metrics import *

import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_dataset(args):
    dataset = args['dataset']
    if args['model']['model_name'] in ['AT_GRU', 'MLPE']:
        train_dataset = STDataset(
                    region_name = dataset['region_name'],
                    folder_path= dataset['folder_path'],
                    task = args['challenge_name'],
                    if_train=True,
                    seq_len=dataset['seq_len']
                 )
        test_dataset = STDataset(
                    region_name = dataset['region_name'],
                    folder_path= dataset['folder_path'],
                    task = args['challenge_name'],
                    if_train=False,
                    seq_len=dataset['seq_len']
                    )
    else:
        train_dataset = STDataset(
                    region_name = dataset['region_name'],
                    folder_path= dataset['folder_path'],
                    task = args['challenge_name'],
                    if_train=True,
                    seq_len=dataset['seq_len']
                    )
        test_dataset = STDataset(
                    region_name = dataset['region_name'],
                    folder_path= dataset['folder_path'],
                    task = args['challenge_name'],
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

def get_trainloader(args):
    dataset = args['dataset']
    if args['model']['model_name'] in ['AT_GRU', 'MLPE']:
        train_dataset = STDataset(
                    region_name = dataset['region_name'],
                    folder_path= dataset['folder_path'],
                    task = args['challenge_name'],
                    if_train=True,
                    seq_len=dataset['seq_len']
                 )
    else:
        train_dataset = STDataset(
                    region_name = dataset['region_name'],
                    folder_path= dataset['folder_path'],
                    task = args['challenge_name'],
                    if_train=True,
                    seq_len=dataset['seq_len']
                    )
    loader = args['dataloader']

    train_size = int(len(train_dataset) * 0.9)
    val_size = len(train_dataset) - train_size 
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=loader['train_bs'], shuffle=False, num_workers=loader['num_workers'])
    validate_loader = DataLoader(val_dataset, batch_size=loader['val_bs'], shuffle=False, num_workers=loader['num_workers'])

    return train_loader, validate_loader

def get_testloader(args):
    dataset = args['dataset']
    loader = args['dataloader']

    # get dataset
    if args['model']['model_name'] in ['AT_GRU', 'MLPE']:
        test_dataset = STDataset(
                    region_name = dataset['region_name'],
                    folder_path= dataset['folder_path'],
                    task = args['challenge_name'],
                    if_train=False,
                    seq_len=dataset['seq_len']
                    )
    else:
        test_dataset = STDataset(
                    region_name = dataset['region_name'],
                    folder_path= dataset['folder_path'],
                    task = args['challenge_name'],
                    if_train=False,
                    seq_len=dataset['seq_len']
                    )

    # get loader
    test_loader = DataLoader(test_dataset, batch_size=loader['test_bs'], shuffle=False, num_workers=loader['num_workers'])

    return test_loader


def get_model(train_loader, args):
    k = args['model']['out_dim']
    inputs, targets, _,  = next(iter(train_loader))

    print(args['model']['model_name'])
    
    if args['model']['model_name'] == 'LSTM':
        bs, seq_x, _, _, _ = inputs.shape
        bs, _ = targets.shape[0], targets.shape[1]

        inputs = inputs.reshape(bs, seq_x, -1)
        targets = targets[:, :, 0:k, ...].reshape(bs, 1, -1)

        input_dim = inputs.shape[-1]
        output_dim = targets.shape[-1]

        n_units1 = args['model']['n_units1']
        n_units2 = args['model']['n_units2']
        dropout_fraction = args['model']['dropout_fraction']
        activ = args['model']['activ']

        print(f'load LSTM, in_dim={input_dim}, out_dim={output_dim}')

        model = LSTMModel(input_dim, output_dim, n_units1, n_units2, dropout_fraction, activ)

    elif args['model']['model_name'] == 'FNN':
        bs_1 = inputs.shape[0]
        bs_2 = targets.shape[0]
        inputs = inputs.reshape(bs_1, -1)
        targets = targets[:, 0:k, ...].reshape(bs_2, -1)

        in_dim = inputs.shape[1]
        out_dim = targets.shape[1]

        n_units1 = args['model']['n_units1']
        n_units2 = args['model']['n_units2']
        dropout_fraction = args['model']['dropout_fraction']
        activ = args['model']['activ']

        print(f'load FNN, in_dim={in_dim}, out_dim={out_dim}')

        model = FFNN(in_dim, out_dim, n_units1, n_units2, dropout_fraction, activ)

    elif args['model']['model_name'] == 'UNET':
        n_channels = args['model']['in_dim']
        n_classes = args['model']['out_dim']

        print(f'load UNET, in_dim={n_channels}, out_dim={n_classes}')

        model = UNet(n_channels, n_classes, bilinear=False)

    elif args['model']['model_name'] == 'Earthformer':
        inputs = inputs.permute(0, 1, 3, 4, 2)  # 
        targets = targets[:, 0:k, ...].permute(0, 1, 3, 4, 2)
        input_shape = inputs.shape[1:]
        target_shape = targets.shape[1:]
        
        base_units = args['model']['base_units']
        downsample_scale = args['model']['downsample_scale']

        print(f'load Earthformer, in_dim={input_shape}, out_dim={target_shape}')

        model = TransformerModel(input_shape,
                        target_shape,
                        base_units=base_units,
                        downsample_scale=downsample_scale,
                        pos_embed_type = "t+h+w",
                        z_init_method='nearest_interp',
                        block_units=None)
        
    elif args['model']['model_name'] == 'AT_GRU':
        num_layers=args['model']['num_layers']
        rnn_hidden_size=args['model']['rnn_hidden_size']
        encoder_input_size=args['model']['encoder_input_size']
        encoder_hidden_size=args['model']['encoder_hidden_size']
        # depth=args['model']['out_dim']

        bs_2 = targets.shape[0]
        # inputs = inputs.reshape(bs_1, -1)
        targets = targets[:, 0:k, ...].reshape(bs_2, -1)

        out_dim = targets.shape[1]
        print(f'load AT_GRU, in_dim={encoder_input_size}, out_dim={out_dim}')

        model = AT_GRU(num_layers, rnn_hidden_size, encoder_input_size, encoder_hidden_size, out_dim)

    elif args['model']['model_name'] == 'MLPE':
        # input_size = args['model']['in_dim']
        bs_1 = inputs.shape[0]
        bs_2 = targets.shape[0]
        inputs = inputs.reshape(bs_1, -1)
        targets = targets[:, 0:k, ...].reshape(bs_2, -1)

        input_size = inputs.shape[1]
        out_dim = targets.shape[1]
        print(f'load IMLP, in_dim={input_size}, out_dim={out_dim}')

        model = MLPE(input_size, out_dim)
        
    return model


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
        for inputs, targets, mask in train_loader:
            inputs, targets, mask, lat = inputs.to(device) , targets.to(device), mask.to(device), lat.to(device)
            optimizer.zero_grad()
            loss, pred, info = model.train_one_step(inputs, targets, mask, lat, criterion, k)

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
            for inputs, targets, mask, lat, _ in validate_loader:
                inputs, targets, mask, lat = inputs.to(device) , targets.to(device), mask.to(device), lat.to(device)
                loss, pred, info = model.train_one_step(inputs, targets, mask, lat, criterion, k)

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

        print(f'Epoch {epoch+1}/{n_epochs}, Training Loss: {epoch_loss:.5f}, Validation Loss: {val_loss:.5f}, Time: {t2-t1:.2f}s')
        if epoch >= pat and all(val_loss >= loss for loss in val_losses[-pat:]):
            print('Early stopping triggered')
            break

    # save loss history
    loss_history = np.array(train_losses)
    if not os.path.exists('./log/loss/'):
        os.makedirs('./log/loss/')
    np.save('./log/loss/' + args["challenge_name"] + '_' + args['dataset']['region_name'] + '_' + model_name + '_loss_history.npy', loss_history)
    avg_time = total_time / (args["epochs"])
    np.save('./log/loss/' + args["challenge_name"] + '_' + args['dataset']['region_name'] + '_' + model_name + '_loss_time.npy', avg_time)
    print("avg_time : {0:.5f}".format(avg_time))


def test_model(model, test_loader, minmax, device, args, metric_names=['MSE', 'RMSE', 'MaxError','NRMSE', 'R2']):
    '''
    test model

    '''
    k = args['model']['out_dim']  # k: depth

    # 创建字典，用于存储结果
    res_list_all = {}
    res_list_depth = {}
    res_list_seq = {}
    for name in metric_names:
        res_list_all[name] = []  

    preds = []
    
    # load model
    model_name = args['model']['model_name']
    saved_model_name = args["challenge_name"] + '_' + args['dataset']['region_name'] + '_' + model_name
    saved_path = os.path.join(args['saved_dir'], saved_model_name)
    print(f"Test mode, load checkpoint from {saved_path}-best.pt")
    checkpoint = torch.load(saved_path + "-best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    # start testing
    model.eval()
    t1 = default_timer()  # start time

    with torch.no_grad():
        for inputs, targets, mask in test_loader:
            inputs, targets, mask = inputs.to(device) , targets.to(device), mask.to(device)
            res, pred, info = model.train_one_step(inputs, targets, mask, minmax, calculate_res, metric_names)
            # print(res)
            preds.append(pred)

            for name in metric_names:
                res_list_all[name].append(res[name])

    for name in metric_names:
        # 在seq维度拼接起来
        res_list_all[name] = torch.stack(res_list_all[name])  # [num_batches, ...]
        # print(name, res_list_all[name].shape)
        # 在拼接的维度求均值
        if name == 'MaxError':
            res_list_depth[name] = torch.max(res_list_all[name], dim=0).values  # [bs, depth]
            # res_list_seq[name] = torch.max(res_list_all[name], dim=1).values  # [bs, seq]
            res_list_all[name] = torch.max(res_list_all[name])  
        else:
            res_list_depth[name] = torch.mean(res_list_all[name], dim=0)  # [depth]
            # res_list_seq[name] = torch.mean(res_list_all[name], dim=1)  # [seq]
            res_list_all[name] = torch.mean(res_list_all[name])  

    t2 = default_timer()  # end time
    inference_time = (t2-t1)/len(test_loader.dataset)
    
    for name in metric_names:
        print(f"average {name}: {res_list_all[name]}")
        print(f"depth {name}: {res_list_depth[name]}")
        # print(f"seq_len {name}: {res_list_seq[name]}")
    print(res_list_all)
    print("Testing time: {}".format(inference_time))

    # # Write results to CSV
    # csv_file_path = os.path.join("./", "test_results.csv")
    # with open(csv_file_path, mode='a', newline='') as file:
    #     writer = csv.writer(file)
    #     # Write headers
    #     writer.writerow([model_name, "Average Value", "Depth Values"])
        
    #     # Write data for each metric
    #     for name in metric_names:
    #         average_value = res_list_all[name].item()
    #         depth_values = res_list_depth[name].cpu().numpy()
    #         writer.writerow([name, average_value, depth_values.tolist()])  # 将深度值转换为列表写入
    # Write results to CSV
    csv_file_path = os.path.join("./", "results_Gulf.csv")
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write headers if the file is empty
        if file.tell() == 0:
            writer.writerow(["Model Name", "Metric Name", "Average Value", "Depth Values", "Inference Time"])

        # Write data for each metric
        for name in metric_names:
            average_value = res_list_all[name].item() if hasattr(res_list_all[name], 'item') else res_list_all[name]
            depth_values = res_list_depth[name].cpu().numpy() if hasattr(res_list_depth[name], 'cpu') else res_list_depth[name]
            writer.writerow([model_name, name, average_value, depth_values.tolist(), inference_time])


    return preds, res_list_depth


def plot_depth(pred_unet, pred_earthformer, pred_earthformer_new, pred_earthformer_kan, target, lat, lon):
    """
    计算2019年1月每个深度的某个地点的真实及预测值，并绘图

    pred: torch.Size([36, 54, 100])
    target: torch.Size([36, 54, 100])

    plot
    """
    pred_earthformer = pred_earthformer[:, lat, lon].numpy()
    pred_unet = pred_unet[:, lat, lon].numpy()
    pred_earthformer_new = pred_earthformer_new[:, lat, lon].numpy()
    pred_earthformer_kan = pred_earthformer_kan[:, lat, lon].numpy()
    target = target[:, lat, lon].numpy()  # [depth]

    depth = [   0,    5,   10,   15,   20,   25,   30,   35,   40,   45,   50,   55,
          60,   65,   70,   80,   90,  100,  125,  150,  175,  200,  225,  250,
         275,  300,  350,  400,  450,  500,  550,  600,  700,  800,  900, 1000]
    
    depth = [-x for x in depth]
    

    # Sample data for demonstration (replace this with actual data)
    unet = pred_unet
    earthformer = pred_earthformer
    truth = target
    new = pred_earthformer_new
    kan = pred_earthformer_kan
    

    # Create subplots
    fig, ax = plt.subplots(figsize=(6, 8))

    # Plotting all cases (replace with your actual data)
    # def plot_nitrate_profile(ax, title):
    print(truth, unet, earthformer)
    ax.plot(truth, depth, label="truth", linestyle='--', color='black')
    ax.plot(unet, depth, label="UNet", color='#736bba')
    ax.plot(earthformer, depth, label="Earthformer", color='#7edace')
    ax.plot(new, depth, label="New_Earthformer", color='#d5dca2')
    ax.plot(kan, depth, label="Kan_Earthformer", color='#86b47a')

    ax.legend()
    
    ax.set_xscale('log')
    ax.set_xlabel('Nitrate Concentration (mmol/m³)')
    ax.set_ylabel("Depth (m)")
    ax.set_title("(a)")

    ax.set_xlim([4.7, 5.5])
    ax.set_ylim([-1000, 0])
    ax.grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

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

# def train_model(model, train_loader, validate_loader, criterion, device, args):
#     '''
#     train model
#     原来的没有使用train one epoch

#     k: depth
#     pat: early stopping patience
#     '''
#     n_epochs = args['epochs']
#     k = args['model']['out_dim']

#     train_losses = []  # loss history
#     val_losses = []
#     min_val_loss = torch.inf  # best validation loss
#     total_time = 0  # total training time
#     pat = args['pat']

#     opt = args['optimizer']['name']
#     lr = args['optimizer']['lr']
#     optimizer = getattr(optim, opt)(model.parameters(), lr=lr)

#     model_name = args['model']['model_name']
#     saved_model_name = args["challenge_name"] + '_' + args['dataset']['region_name'] + '_' + model_name
#     saved_path = os.path.join(args['saved_dir'], saved_model_name)
    
#     for epoch in tqdm(range(n_epochs)):
#         model.train()
#         t1 = default_timer()  # start time
#         running_loss = 0.0

#         # start training
#         for inputs, targets, _, _, _ in train_loader:
#             if model_name == 'FNN':
#                 inputs, targets = process_data(inputs, targets, k, lstm = False)  # [bs, var, lat, lon] --> [bs, n]
#             elif model_name == 'LSTM':
#                 inputs, targets = process_data(inputs, targets, k, lstm = True)   # [bs, seq, var, lat, lon] --> [bs, seq, n]  torch.Size([1, 5, 16200])
#             elif model_name == 'UNET':
#                 inputs, targets, _, _ = down_sample(inputs, targets)  # [bs, var, new_lat, new_lon]  torch.Size([1, 12, 27, 50]
#                 inputs, targets = inputs, targets[:, 0:k, ...]
#             elif model_name == 'Earthformer':  # [1, 5, 12, 108, 200]
#                 inputs, targets, _, _ = down_sample(inputs, targets)
#                 inputs, targets = inputs.permute(0, 1, 3, 4, 2), targets[:, 0:k, ...].permute(0, 1, 3, 4, 2)
#             else:
#                 raise NotImplementedError
            
#             inputs, targets = inputs.to(device) , targets.to(device)  
#             optimizer.zero_grad()
#             outputs = model(inputs)
            
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)
        
#         t2 = default_timer()  # end time
#         total_time += (t2-t1)
#         epoch_loss = running_loss / len(train_loader.dataset)
#         train_losses.append(epoch_loss)

#         # start validation
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for inputs, targets, _, _, _ in validate_loader:
#                 if model_name == 'FNN':
#                     inputs, targets = process_data(inputs, targets, k, lstm = False)   # [bs, var, lat, lon] --> [bs, n]
#                 elif model_name == 'LSTM':
#                     inputs, targets = process_data(inputs, targets, k, lstm = True)   # [bs, var, lat, lon] --> [bs, n]
#                 elif model_name == 'UNET':
#                     inputs, targets, _, _ = down_sample(inputs, targets)  
#                     inputs, targets = inputs , targets[:, 0:k, ...]
#                 elif model_name == 'Earthformer':
#                     inputs, targets, _, _ = down_sample(inputs, targets)
#                     inputs, targets = inputs.permute(0, 1, 3, 4, 2), targets[:, 0:k, ...].permute(0, 1, 3, 4, 2)
#                 else:
#                     raise NotImplementedError

#                 inputs, targets = inputs.to(device) , targets.to(device)  
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#                 val_loss += loss.item() * inputs.size(0)

#         val_loss = val_loss / len(validate_loader.dataset)
#         val_losses.append(val_loss)

#         # save best model
#         if val_loss < min_val_loss:
#             min_val_loss = val_loss
#             torch.save({"epoch": epoch + 1, "loss": min_val_loss,
#                 # "model_state_dict": model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
#                 "model_state_dict": model.state_dict(),
#                 "optimizer_state_dict": optimizer.state_dict()
#                 }, saved_path + "-best.pt")

#         print(f'Epoch {epoch+1}/{n_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Time: {t2-t1:.2f}s')
#         if epoch > pat and all(val_loss >= loss for loss in val_losses[-pat:]):
#             print('Early stopping triggered')
#             break

#     # save loss history
#     loss_history = np.array(train_losses)
#     if not os.path.exists('./log/loss/'):
#         os.makedirs('./log/loss/')
#     np.save('./log/loss/' + args["challenge_name"] + '_' + args['dataset']['region_name'] + '_' + model_name + '_loss_history.npy', loss_history)
#     print("avg_time : {0:.5f}".format(total_time / (args["epochs"])))
    

# def train_model_2(model, train_loader, validate_loader, criterion, device, args):
#     '''
#     train model

#     k: depth
#     pat: early stopping patience
#     '''
#     n_epochs = args['epochs']
#     k = args['model']['out_dim']

#     train_losses = []  # loss history
#     val_losses = []
#     min_val_loss = torch.inf  # best validation loss
#     total_time = 0  # total training time
#     pat = args['pat']

#     opt = args['optimizer']['name']
#     lr = args['optimizer']['lr']
#     optimizer = getattr(optim, opt)(model.parameters(), lr=lr)

#     model_name = args['model']['model_name']
#     saved_model_name = args["challenge_name"] + '_' + args['dataset']['region_name'] + '_' + model_name
#     saved_path = os.path.join(args['saved_dir'], saved_model_name)
    
#     for epoch in tqdm(range(n_epochs)):
#         model.train()
#         t1 = default_timer()  # start time
#         running_loss = 0.0

#         # start training
#         for inputs, targets, mask, _, _ in train_loader:
#             inputs, targets, mask = inputs.to(device) , targets.to(device), mask.to(device)
#             optimizer.zero_grad()
#             loss, pred, info = model.train_one_step(inputs, targets, mask, criterion, k)

#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)
        
#         t2 = default_timer()  # end time
#         total_time += (t2-t1)
#         epoch_loss = running_loss / len(train_loader.dataset)
#         train_losses.append(epoch_loss)

#         # start validation
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for inputs, targets, mask, _, _ in validate_loader:
#                 inputs, targets, mask = inputs.to(device) , targets.to(device), mask.to(device)
#                 loss, pred, info = model.train_one_step(inputs, targets, mask, criterion, k)

#                 val_loss += loss.item() * inputs.size(0)

#         val_loss = val_loss / len(validate_loader.dataset)
#         val_losses.append(val_loss)

#         # save best model
#         if val_loss < min_val_loss:
#             min_val_loss = val_loss
#             torch.save({"epoch": epoch + 1, "loss": min_val_loss,
#                 # "model_state_dict": model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
#                 "model_state_dict": model.state_dict(),
#                 "optimizer_state_dict": optimizer.state_dict()
#                 }, saved_path + "-best.pt")

#         print(f'Epoch {epoch+1}/{n_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Time: {t2-t1:.2f}s')
#         if epoch >= pat and all(val_loss >= loss for loss in val_losses[-pat:]):
#             print('Early stopping triggered')
#             break

#     # save loss history
#     loss_history = np.array(train_losses)
#     if not os.path.exists('./log/loss/'):
#         os.makedirs('./log/loss/')
#     np.save('./log/loss/' + args["challenge_name"] + '_' + args['dataset']['region_name'] + '_' + model_name + '_loss_history.npy', loss_history)
#     print("avg_time : {0:.5f}".format(total_time / (args["epochs"])))
