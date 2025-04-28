import torch
import os
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

def MSE(pred, target, valid_count):
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
    # print(pred.shape, valid_count.shape, errors.shape)
    res = torch.sum(errors**2, dim=-1) / valid_count
    return res # (bs, seq_len, depth)

def RMSE(pred, target, valid_count):
    """return root mean square error

    pred: model output tensor of shape (batch_size, seq_len, depth, n)
    target: ground truth tensor of shape (batch_size, seq_len, depth, n)
    """
    return torch.sqrt(MSE(pred, target, valid_count)) # (bs, depth, seq_len)

def NRMSE(pred, target, valid_count):
    """return normalized root-mean-square error
    
    avoid the scale dependency
    NRMSE serves as a relative measure, and does not possess significant numerical meaning on its own
    RMSE divided by the mean of the corresponding labeling values.

    pred: model output tensor of shape (batch_size, seq_len, depth, n)
    target: ground truth tensor of shape (batch_size, seq_len, depth, n)
    """

    assert pred.shape == target.shape
    target_mean = torch.mean(target, dim=-1)  # 掩码处为零值，导致res有nan
    
    # 创建一个掩码，过滤掉 NaN 值
    nan_mask = ~torch.isnan(target)  # True 表示非 NaN 值
    # 初始化 max/min 值为一个大的负值和正值
    min_values = torch.full_like(target, float('inf'))
    max_values = torch.full_like(target, float('-inf'))
    # 用掩码更新非 NaN 值
    target_min = torch.where(nan_mask, target, min_values).min(dim=-1)[0]
    target_max = torch.where(nan_mask, target, max_values).max(dim=-1)[0]
    target_minmax = target_max - target_min
    
    res = RMSE(pred, target, valid_count) / target_minmax

    return res # (batch_size, seq_len, depth)

def L2RE(pred, target, valid_count):
    """return normalized root-mean-square error
    
    avoid the scale dependency
    NRMSE serves as a relative measure, and does not possess significant numerical meaning on its own
    RMSE divided by the mean of the corresponding labeling values.

    pred: model output tensor of shape (batch_size, seq_len, depth, n)
    target: ground truth tensor of shape (batch_size, seq_len, depth, n)
    """

    assert pred.shape == target.shape
    
    res = RMSE(pred, target, valid_count) / torch.sqrt( torch.sum(target*target, dim=-1) / valid_count )

    return res # (batch_size, seq_len, depth)


def R2(pred, target, valid_count):
    """
    计算 R2 决定系数
    
    pred: 模型预测输出张量，形状为 (batch_size, seq_len, depth, n)
    target: 真实标签张量，形状为 (batch_size, seq_len, depth, n)
    """
    assert pred.shape == target.shape

    # 计算平方误差 (residual sum of squares)
    errors = pred - target
    ss_res = torch.sum(errors**2, dim=-1)  # 沿最后一个维度求和，得到 [batch_size, seq_len, depth]

    # 计算目标值的总平方和 (total sum of squares)
    target_mean = torch.mean(target, dim=-1, keepdim=True)  
    ss_tot = torch.sum((target - target_mean)**2, dim=-1)  # [batch_size, seq_len, depth]

    # 计算 R2 值
    r2 = 1 - (ss_res / ss_tot)  # [batch_size, seq_len, depth]

    return r2  # 返回形状为 (batch_size, seq_len, depth) 的 R2 值
    

# def L2RE(pred, target):
#     """l2 relative error (nMSE in PDEBench)

#     pred: model output tensor of shape (batch_size, seq_le, lat, lon, depth)
#     target: ground truth tensor of shape (batch_size, seq_le, lat, lon, depth)
#     """
#     assert pred.shape == target.shape
#     temp_shape = [0, len(pred.shape)-1]
#     temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
#     pred = pred.permute(temp_shape) # (bs, x1, ..., xd, t, v) -> (bs, v, x1, ..., xd, t)
#     target = target.permute(temp_shape) # (bs, x1, ..., xd, t, v) -> (bs, v, x1, ..., xd, t)
#     nb, nc = pred.shape[0], pred.shape[1]
#     errors = pred.reshape([nb, nc, -1]) - target.reshape([nb, nc, -1]) # (bs, v, x1*x2*...*xd*t)
#     res = torch.sum(errors**2, dim=2) / torch.sum(target.reshape([nb, nc, -1])**2, dim=2)
#     return torch.sqrt(res) # (bs, v)

def MaxError(pred, target, valid_count):
    """return max error in a batch

    pred: model output tensor of shape (batch_size, seq_le, depth, n)
    target: ground truth tensor of shape (batch_size, seq_le, depth, n)
    """
    errors = torch.abs(pred - target)
    # nc = errors.permute(0, 2, 1).shape[-1]  # seq_len
    # res, _ = torch.max(errors.reshape([-1, nc]), dim=0) # retain the last dim
    res, _ = torch.max(errors, dim=-1)
    return res # (batch_size, seq_le, depth)

def calculate_res(pred, target, metric_names, valid_count):
    """
    计算指标
    in: (batch_size, seq_len, depth, n)
    out: (bs, seq_len, depth)
    """
    res_list = {}

    for metric in metric_names:
        res = eval(metric)(pred, target, valid_count)
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