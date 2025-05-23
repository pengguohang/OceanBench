import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from timeit import default_timer
from tqdm import tqdm
from torch import optim
import argparse
import yaml
import csv

from dataset import *
from model import *
from metrics import *
from utils import get_dataset, get_loader, get_model


def train_model(model, train_loader, validate_loader, criterion, minmax, device, args):
    '''
    train model

    pat: early stopping patience
    '''
    n_epochs = args['epochs']

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
            inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
            optimizer.zero_grad()
            loss, pred, info = model.train_one_step(inputs, targets, mask, minmax, criterion)

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
            for inputs, targets, mask in validate_loader:
                inputs, targets, mask = inputs.to(device) , targets.to(device), mask.to(device)
                loss, pred, info = model.train_one_step(inputs, targets, mask, minmax, criterion)

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
            # print(res)inputs, targets, mask, minmax, criterion
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


    return preds, res_list_all





def main(args):
    device = torch.device(args['device'] if torch.cuda.is_available() else 'cpu')

    train_data, test_data = get_dataset(args)
    train_loader, val_loader, test_loader = get_loader(train_data, test_data, args)

    print('len(train_loader): ', len(train_loader))
    print('len(validate_loader)', len(val_loader))
    print('len(test_loader)', len(test_loader))

    data = next(iter(train_loader))
    for i in range(len(data)):
        print(data[i].shape)

    model = get_model(train_loader, args)
    model = model.to(device)

    # denormalization
    minmax = train_data.get_Minmax()

    if args['if_testing']:
        test_model(model, test_loader, minmax, device, args, metric_names=['MSE', 'RMSE', 'MaxError','NRMSE', 'R2', 'L2RE'])
    else:
        print('start training-----------------------------------------------------')
        criterion = nn.MSELoss()
        train_model(model, train_loader, val_loader, criterion, minmax, device, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    parser.add_argument("--test", action='store_true', help='test mode otherwise train mode')
    parser.add_argument("--model", type=str, default='', help="model name")
    parser.add_argument("--epochs", type=int, default=0, help="100")
    parser.add_argument("--region", type=str, default='', help="Gulf, ")
    parser.add_argument("--task", type=str, default='', help="S, T")

    cmd_args = parser.parse_args()
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)

    args['if_testing'] = cmd_args.test
    if len(cmd_args.model) > 0:
        args['model']['model_name'] = cmd_args.model
    if len(cmd_args.region) > 0:
        args['dataset']['region_name'] = cmd_args.region
    if len(cmd_args.task) > 0:
        args['challenge_name'] = cmd_args.task
    if cmd_args.epochs > 0:
        args['epochs'] = cmd_args.epochs

    print(args)
    main(args)

    

