import torch
import os
import pandas as pd

def MSE(pred, target):
    """return mean square error

    pred: model output tensor of shape  (batch_size, seq_le, depth, n)
    target: ground truth tensor of shape (batch_size, seq_le, depth, n)
    """
    assert pred.shape == target.shape
    # temp_shape = [0, len(pred.shape)-1] 
    # temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
    # pred = pred.permute(temp_shape) # (batch_size, seq_le, lat, lon, depth) -> (batch_size, depth, seq_le, lat, lon)
    # target = target.permute(temp_shape) # (batch_size, seq_le, lat, lon, depth) -> (batch_size, depth, seq_le, lat, lon)
    # bs, seq, depth = pred.shape[0], pred.shape[1], pred.shape[2]
    # errors = pred.reshape([nb, nc, ns, -1]) - target.reshape([nb, nc, ns, -1]) # (batch_size, depth, seq_le * lat * lon)
    errors = pred - target
    res = torch.mean(errors**2, dim=-1)
    return res # (bs, seq_len, depth)

def RMSE(pred, target):
    """return root mean square error

    pred: model output tensor of shape (batch_size, seq_len, depth, n)
    target: ground truth tensor of shape (batch_size, seq_len, depth, n)
    """
    return torch.sqrt(MSE(pred, target)) # (bs, depth, seq_len)

def NRMSE(pred, target):
    """return normalized root-mean-square error
    
    avoid the scale dependency
    NRMSE serves as a relative measure, and does not possess significant numerical meaning on its own
    RMSE divided by the mean of the corresponding labeling values.

    pred: model output tensor of shape (batch_size, seq_len, depth, n)
    target: ground truth tensor of shape (batch_size, seq_len, depth, n)
    # label里面有0值，没处理好
    """

    assert pred.shape == target.shape
    # temp_shape = [0, len(pred.shape)-1] 
    # temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
    # target_1 = target.permute(temp_shape) # (batch_size, seq_le, lat, lon, depth) -> (batch_size, depth, seq_le, lat, lon)
    # nb, nc, ns = target_1.shape[0], target_1.shape[1], target_1.shape[2]
    # target_1 = target_1.reshape(nb, nc, ns, -1)
    # target_mean = torch.mean(target_1, dim=3)
    target_mean = torch.mean(target, dim=-1)  # 掩码处为零值，导致res有nan
    res = RMSE(pred, target) / target_mean

    return res # (batch_size, seq_len, depth)


def L2RE(pred, target):
    """l2 relative error (nMSE in PDEBench)

    pred: model output tensor of shape (batch_size, seq_le, lat, lon, depth)
    target: ground truth tensor of shape (batch_size, seq_le, lat, lon, depth)
    """
    assert pred.shape == target.shape
    temp_shape = [0, len(pred.shape)-1]
    temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
    pred = pred.permute(temp_shape) # (bs, x1, ..., xd, t, v) -> (bs, v, x1, ..., xd, t)
    target = target.permute(temp_shape) # (bs, x1, ..., xd, t, v) -> (bs, v, x1, ..., xd, t)
    nb, nc = pred.shape[0], pred.shape[1]
    errors = pred.reshape([nb, nc, -1]) - target.reshape([nb, nc, -1]) # (bs, v, x1*x2*...*xd*t)
    res = torch.sum(errors**2, dim=2) / torch.sum(target.reshape([nb, nc, -1])**2, dim=2)
    return torch.sqrt(res) # (bs, v)

def MaxError(pred, target):
    """return max error in a batch

    pred: model output tensor of shape (batch_size, seq_le, depth, n)
    target: ground truth tensor of shape (batch_size, seq_le, depth, n)
    """
    errors = torch.abs(pred - target)
    # nc = errors.permute(0, 2, 1).shape[-1]  # seq_len
    # res, _ = torch.max(errors.reshape([-1, nc]), dim=0) # retain the last dim
    res, _ = torch.max(errors, dim=-1)
    return res # (batch_size, seq_le, depth)

def calculate_res(pred, target, metric_names):
    """
    计算指标
    in: (batch_size, seq_len, depth, n)
    out: (bs, seq_len, depth)
    """
    # metric_names=['MSE', 'RMSE', 'NRMSE', 'MaxError',]
    res_list = {}

    for metric in metric_names:
        res = eval(metric)(pred, target)
        res_list[metric] = res
        
    return res_list

# def res_all(pred, target, metric_names):
#     """
#     计算所有深度的平均指标
#     in: (batch_size, seq_len, depth, n)
#     out: 1*1
#     """
#     # metric_names=['MSE', 'RMSE', 'NRMSE', 'MaxError',]
#     res_list = res(pred, target, metric_names)

#     for metric in metric_names:
#         if metric == 'MaxError':
#             res_list[metric] = torch.max(res_list[metric]).unsqueeze(0)
#         else:
#             res_list[metric] = res_list[metric].mean().unsqueeze(0)
#     return res_list

# def res_depth(pred, target, metric_names):
#     """
#     计算每个深度的平均指标
#     in: (batch_size, seq_len, depth, n)
#     out: (depth)
#     """
#     # metric_names=['MSE', 'RMSE', 'NRMSE', 'MaxError',]
#     res_list = res(pred, target, metric_names)  # (bs, seq_len, depth)

#     for metric in metric_names:
#         depth = res_list[metric].shape[-1]
#         res_metric = res_list[metric].reshape(-1, depth)  # (bs*seq_len, depth)
#         if metric == 'MaxError':
#             res_list[metric] = torch.max(res_metric, dim=0).values
#         else:
#             res_list[metric] = torch.mean(res_metric, dim=0)
#     return res_list

# def res_seq(pred, target, metric_names):
#     """
#     计算每个月份的平均指标
#     in: (batch_size, seq_len, depth, n)
#     out: (seq_len)
#     """
#     # metric_names=['MSE', 'RMSE', 'NRMSE', 'MaxError',]
#     res_list = res(pred, target, metric_names)  # (bs, seq_len, depth)

#     for metric in metric_names:
#         seq_len = res_list[metric].shape[1]
#         res_metric = res_list[metric].permute(0, 2, 1).reshape(-1, seq_len)  # (bs*seq_len, depth)
#         if metric == 'MaxError':
#             res_list[metric] = torch.max(res_metric, dim=0).values
#         else:
#             res_list[metric] = torch.mean(res_metric, dim=0)
#     return res_list

# 不同月份的RMSE
# 不同深度的RMSE
# 整体平均RMSE