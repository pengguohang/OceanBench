import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import yaml
import argparse
# import metrics
from timeit import default_timer
from functools import reduce

def main(args):
    train_data, val_data, test_data, test_ms_data = get_dataset(args)
    train_loader, val_loader, test_loader, test_ms_loader = get_dataloader(train_data, val_data, test_data, test_ms_data, args)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    parser.add_argument("--test", action='store_true', help='test mode otherwise train mode')
    parser.add_argument("--no_denorm", action='store_true', help='no denorm in test mode')
    parser.add_argument("--continue_training", action='store_true', help='continue training')
    parser.add_argument("-c", "--case_name", type=str, default="", help="For the case, if no value is entered, the yaml file shall prevail")

    cmd_args = parser.parse_args()
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)
    args['if_training'] = not cmd_args.test
    args['continue_training'] = cmd_args.continue_training
    if len(cmd_args.case_name) > 0:
        args['dataset']['case_name'] = cmd_args.case_name

    setup_seed(args["seed"])

    # set use_norm
    use_norm_default=True
    if args["flow_name"] in ["TGV","Darcy"]:
        use_norm_default = False
    args["use_norm"] = args.get("use_norm", use_norm_default)
    args["if_denorm"] = not cmd_args.no_denorm
    
    print(args, flush=True)
    main(args)