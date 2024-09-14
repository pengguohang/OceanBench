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

from dataset import *
from model import *
from metrics import *
from utils import get_dataset, get_loader, get_model, train_model, test_model


def main(args):
    device = torch.device(args['device'] if torch.cuda.is_available() else 'cpu')

    train_data, test_data = get_dataset(args)
    train_loader, val_loader, test_loader = get_loader(train_data, test_data, args)

    print('len(train_loader): ', len(train_loader))
    print('len(validate_loader)', len(val_loader))
    print('len(test_loader)', len(test_loader))

    model = get_model(train_loader, args)
    model = model.to(device)

    if args['if_testing']:
        test_model(model, test_loader, device, args, metric_names=['MSE', 'RMSE', 'MaxError',])
    else:
        criterion = nn.MSELoss()
        train_model(model, train_loader, val_loader, criterion, device, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    parser.add_argument("--test", action='store_true', help='test mode otherwise train mode')

    cmd_args = parser.parse_args()
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)

    args['if_testing'] = cmd_args.test

    main(args)

    

