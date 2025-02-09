import netCDF4 as nc
import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset, IterableDataset
import xarray as xr
import torch.nn.functional as F

# 选择子区域
# 归一化与逆归一化

class STDataset(Dataset):
    def __init__(self,
                 region_name = 'Gulf',
                 folder_path='/home/data2/pengguohang/My_Ocean/challenge/1993_2019_data/',
                 label_path='/home/data2/pengguohang/My_Ocean/challenge/1993_2019_data/label.nc',
                 task='S',  # S T Chl 
                 if_train=True,
                 seq_len=5,
                 ):

        if region_name == 'Gulf':
            # 均选取了核心区域
            self.lat_min, self.lat_max = 23, 50
            self.lon_min, self.lon_max = -80, -30
        elif region_name == 'Pacific':
            self.lat_min, self.lat_max = -50, 0  
            self.lon_min, self.lon_max = -150, -70   
        elif region_name == 'Indian':
            self.lat_min, self.lat_max = -20, 10  
            self.lon_min, self.lon_max = 40, 100  

        else: 
            print('Invilid regian name!!!')

        key = task
        self.input, self.label, self.lat, self.lon, self.depth, self.minmax = self.get_data(folder_path, key)

        self.input = self.input.permute(1, 0, 2, 3)  # [time, var, lat, lon]
        self.label = self.label.permute(1, 0, 2, 3)  # [time, depth, lat, lon]
        self.lat = torch.tensor(self.lat)
        self.lon = torch.tensor(self.lon)
        self.depth = torch.tensor(self.depth)
        # print('depth: ', self.depth)

        # 将lat和lon合并到input中
        time = self.input.shape[0]
        lat = self.input.shape[2]
        lon = self.input.shape[3]
        expand_lat = ( (self.lat-self.lat.min()) / (self.lat.max() - self.lat.min())).unsqueeze(0).unsqueeze(-1).repeat(time, 1, 1, lon)
        expand_lon = ( (self.lon-self.lon.min()) / (self.lon.max() - self.lon.min())).unsqueeze(0).unsqueeze(0).repeat(time, 1, lat, 1)
        self.input = torch.cat((self.input, expand_lat, expand_lon), dim=1)
        # print('add lat: ', self.input.shape, expand_lat.shape)

        # 将时间合并到input中
        time_series = np.array(range(time))
        jd1 = torch.cos( torch.tensor(2*np.pi*(time_series/12)+1) )
        jd2 = torch.sin( torch.tensor(2*np.pi*(time_series/12)+1) )
        jd1 = jd1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, lat, lon)
        jd2 = jd2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, lat, lon)
        # print('add time: ', self.input.shape, jd1.shape)
        # print('jd1, jd2', jd1.shape, jd2.shape)
        self.input = torch.cat((self.input, jd1), dim=1)
        self.input = torch.cat((self.input, jd2), dim=1)

        # Create mask: if any input variable or label at a grid point is NaN, set mask to False
        input_nan_mask = torch.isnan(self.input)
        label_nan_mask = torch.isnan(self.label)
        input_nan_mask = input_nan_mask.any(dim=(0, 1))  # [time, var, lat, lon]
        label_nan_mask = label_nan_mask.any(dim=(0, 1))
        self.mask = (~label_nan_mask) & (~input_nan_mask)  # [lat, lon], NaN

        # Time-series data: Add seq_len dimension, create sliding window sequences
        if seq_len > 0:
            num = self.input.shape[0] - seq_len
            self.input = torch.stack([self.input[i:i + seq_len] for i in range(num)], dim=0)
            self.label = torch.stack([self.label[i + seq_len - 1:i + seq_len] for i in range(num)], dim=0)

        # Split data into train: 199401-201712, test: 199301-199312, 201801-201901
        test_len = 12
        train_len = self.input.shape[0] - test_len
        if if_train:
            self.input = self.input[test_len + 1:train_len, ...]
            self.label = self.label[test_len + 1:train_len, ...]
        else:
            input1 = self.input[:test_len + 1, ...]  # First 12 months
            label1 = self.label[:test_len + 1, ...]
            input2 = self.input[train_len:, ...]  # Last 12 months
            label2 = self.label[train_len:, ...]
            self.input = torch.cat((input1, input2), dim=0)
            self.label = torch.cat((label1, label2), dim=0)

        print('Shape of variables:', self.input.shape, self.label.shape, self.lat.shape, self.lon.shape, self.depth.shape)
        self.mask_x = self.mask.unsqueeze(0).unsqueeze(-1).repeat(self.input.shape[0],1,1, self.input.shape[1]).reshape(-1, self.input.shape[1])
        self.input = self.input.permute(0,2,3,1).reshape(-1, self.input.shape[1])
        self.mask_y = self.mask.unsqueeze(0).unsqueeze(-1).repeat(self.label.shape[0],1,1, self.label.shape[1]).reshape(-1, self.label.shape[1])
        self.label = self.label.permute(0,2,3,1).reshape(-1, self.label.shape[1])
        

        print('shape of variable: ', self.input.shape, self.label.shape, self.mask_x.shape, self.mask_y.shape, self.depth.shape)

        
    def get_data(self, folder_path, key):
        """
        Extract input data and concatenate
        folder_path: data folder path

        return: (var, time, lat, lon)
        """
        input_path = os.path.join(folder_path, 'input.nc')
        label_path = os.path.join(folder_path, 'label.nc')
        input = xr.open_dataset(input_path)
        label = xr.open_dataset(label_path)
        
        # 归一化input
        input_data = torch.tensor(input['data'].values)
        
        masked_input = input_data.clone()
        masked_input[torch.isnan(masked_input)] = -float('inf')  # 将 NaN 替换为无穷小以忽略它们
        max_input, _ = torch.max(masked_input.view(masked_input.shape[0], -1), dim=1)
        masked_input[torch.isnan(input_data)] = float('inf')  # 将 NaN 替换为无穷大以忽略它们
        min_input, _ = torch.min(masked_input.view(masked_input.shape[0], -1), dim=1)
        max_input[0], min_input[0] = 1, 0  # 第一个为mdt数据，不归一化

        max_input, min_input = max_input.view(max_input.shape[0], 1, 1, 1).expand_as(input_data), min_input.view(max_input.shape[0], 1, 1, 1).expand_as(input_data)
        input_minmax = (input_data - min_input) / (max_input - min_input) 

        # 归一化label
        label_data = torch.tensor(label[key].values).permute(1, 0, 2, 3)

        masked_label = label_data.clone()
        masked_label[torch.isnan(masked_label)] = -float('inf')
        max_label, _ = torch.max(masked_label.reshape(masked_label.shape[0], -1), dim=1)
        masked_label[torch.isnan(label_data)] = float('inf')
        min_label, _ = torch.min(masked_label.reshape(masked_label.shape[0], -1), dim=1)

        max_label, min_label = max_label.view(label_data.shape[0], 1, 1, 1).expand_as(label_data), min_label.view(label_data.shape[0], 1, 1, 1).expand_as(label_data)
        label_minmax = (label_data - min_label) / (max_label - min_label) 

        # 存储最值
        minmax = [min_input, max_input, min_label, max_label]

        # get sub
        input_sub, lat_sub, lon_sub = self.get_sub(input_minmax, input['lat'].values, input['lon'].values)
        label_sub, _, _ = self.get_sub(label_minmax, input['lat'].values, input['lon'].values)

        return input_sub, label_sub, lat_sub, lon_sub, label['depth'].values, minmax
    
    def get_sub(self, data, latitude, longitude):
        """
        提取子区域的数据

        input:
        lat_min, lat_max, lon_min, lon_max: 子区域范围
        data: 原始数据
        latitude, longitude: 经纬度数据

        return: subset_data, subset_lat, subset_lon
        """
        # 找到对应的索引
        # print('in sub: ', data.shape, latitude.shape, longitude.shape)
        lat_indices = np.where((latitude >= self.lat_min) & (latitude <= self.lat_max))[0]
        lon_indices = np.where((longitude >= self.lon_min) & (longitude <= self.lon_max))[0]
        # 提取子集数据
        subset_data = data[:, :, lat_indices, :][:, :, :, lon_indices]
        # print(subset_data.shape)
        # 提取相应的经纬度数组
        subset_lat = latitude[lat_indices]
        subset_lon = longitude[lon_indices]

        return subset_data, subset_lat, subset_lon

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        inputs = self.input[idx]   # (var, lat, lon) or (seq, var, lat, lon)
        label = self.label[idx]    # (dept, lat, lon) or (seq, dept, lat, lon)
        lat = self.lat
        lon = self.lon
        depth = self.depth
        mask_x = self.mask_x[idx] 
        mask_y = self.mask_y[idx]
        
        return inputs.float(), label.float(), mask_x, mask_y, lon


# class STDataset_points(Dataset):
#     def __init__(self,
#                  region_name = 'Gulf',
#                  folder_path='/home/data2/pengguohang/My_Ocean/challenge/1993_2019_data/',
#                  label_path='/home/data2/pengguohang/My_Ocean/challenge/1993_2019_data/label.nc',
#                  task='S',  # S T Chl 
#                  if_train=True,
#                  seq_len=5,
#                  ):

#         if region_name == 'Gulf':
#             self.lat_min, self.lat_max = 23, 50
#             self.lon_min, self.lon_max = -80, -30
#         elif region_name == 'Pacific':
#             self.lat_min, self.lat_max = -50, 0   # 南太平洋的纬度范围
#             self.lon_min, self.lon_max = -150, -70  # 南太平洋的经度范围
#         elif region_name == 'Indian':
#             self.lat_min, self.lat_max = -20, 10  # 更核心的纬度范围
#             self.lon_min, self.lon_max = 40, 100  # 更核心的经度范围
#         else: 
#             print('Invilid regian name!!!')
#         key = task
#         self.input, self.label, self.lat, self.lon, self.depth, self.minmax = self.get_data(folder_path, key)

#         self.input = self.input.permute(1, 0, 2, 3)  # [time, var, lat, lon]
#         self.label = self.label.permute(1, 0, 2, 3)  # [time, depth, lat, lon]
#         self.lat = torch.tensor(self.lat)
#         self.lon = torch.tensor(self.lon)
#         self.depth = torch.tensor(self.depth)

#         # 将lat和lon合并到input中
#         time = self.input.shape[0]
#         lat = self.input.shape[2]
#         lon = self.input.shape[3]
#         expand_lat = self.lat.unsqueeze(0).unsqueeze(-1).repeat(time, 1, 1, lon)
#         expand_lon = self.lon.unsqueeze(0).unsqueeze(0).repeat(time, 1, lat, 1)
#         self.input = torch.cat((self.input, expand_lat, expand_lon), dim=1)
#         # print('add lat: ', self.input.shape, expand_lat.shape)

#         # 将时间合并到input中
#         time_series = np.array(range(time))
#         jd1 = torch.cos( torch.tensor(2*np.pi*(time_series/12)+1) )
#         jd2 = torch.sin( torch.tensor(2*np.pi*(time_series/12)+1) )
#         jd1 = jd1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, lat, lon)
#         jd2 = jd2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, lat, lon)
#         # print('add time: ', self.input.shape, jd1.shape)
#         # print('jd1, jd2', jd1.shape, jd2.shape)
#         self.input = torch.cat((self.input, jd1), dim=1)
#         self.input = torch.cat((self.input, jd2), dim=1)

#         # Create mask: if any input variable or label at a grid point is NaN, set mask to False
#         input_nan_mask = torch.isnan(self.input)
#         label_nan_mask = torch.isnan(self.label)
#         input_nan_mask = input_nan_mask.any(dim=(0, 1))  # [time, var, lat, lon]
#         label_nan_mask = label_nan_mask.any(dim=(0, 1))
#         self.mask = (~label_nan_mask) & (~input_nan_mask)  # [lat, lon], NaN

#         # Time-series data: Add seq_len dimension, create sliding window sequences
#         if seq_len > 0:
#             num = self.input.shape[0] - seq_len
#             self.input = torch.stack([self.input[i:i + seq_len] for i in range(num)], dim=0)
#             self.label = torch.stack([self.label[i + seq_len - 1:i + seq_len] for i in range(num)], dim=0)

#         # Split data into train: 199401-201712, test: 199301-199312, 201801-201901
#         test_len = 12
#         train_len = self.input.shape[0] - test_len
#         if if_train:
#             self.input = self.input[test_len + 1:train_len, ...]
#             self.label = self.label[test_len + 1:train_len, ...]
#         else:
#             input1 = self.input[:test_len + 1, ...]  # First 12 months
#             label1 = self.label[:test_len + 1, ...]
#             input2 = self.input[train_len:, ...]  # Last 12 months
#             label2 = self.label[train_len:, ...]
#             self.input = torch.cat((input1, input2), dim=0)
#             self.label = torch.cat((label1, label2), dim=0)

#         print('Shape of variables:', self.input.shape, self.label.shape, self.lat.shape, self.lon.shape, self.depth.shape)
        
#         self.mask_x = self.mask.unsqueeze(0).unsqueeze(-1).repeat(self.input.shape[0],1,1, self.input.shape[1]).reshape(-1, self.input.shape[1])
#         self.input = self.input.permute(0,2,3,1).reshape(-1, self.input.shape[1])
#         self.mask_y = self.mask.unsqueeze(0).unsqueeze(-1).repeat(self.label.shape[0],1,1, self.label.shape[1]).reshape(-1, self.label.shape[1])
#         self.label = self.label.permute(0,2,3,1).reshape(-1, self.label.shape[1])
        

#         print('shape of variable: ', self.input.shape, self.label.shape, self.mask_x.shape, self.mask_y.shape, self.depth.shape)


#     def get_data(self, folder_path, key):
#         """
#         Extract input data and concatenate
#         folder_path: data folder path

#         return: (var, time, lat, lon)
#         """
#         input_path = os.path.join(folder_path, 'input.nc')
#         label_path = os.path.join(folder_path, 'label.nc')
#         input = xr.open_dataset(input_path)
#         label = xr.open_dataset(label_path)
        
#         # 归一化input
#         input_data = torch.tensor(input['data'].values)
        
#         masked_input = input_data.clone()
#         masked_input[torch.isnan(masked_input)] = -float('inf')  # 将 NaN 替换为无穷小以忽略它们
#         max_input, _ = torch.max(masked_input.view(masked_input.shape[0], -1), dim=1)
#         masked_input[torch.isnan(input_data)] = float('inf')  # 将 NaN 替换为无穷大以忽略它们
#         min_input, _ = torch.min(masked_input.view(masked_input.shape[0], -1), dim=1)
#         max_input[0], min_input[0] = 1, 0  # 第一个为mdt数据，不归一化

#         max_input, min_input = max_input.view(max_input.shape[0], 1, 1, 1).expand_as(input_data), min_input.view(max_input.shape[0], 1, 1, 1).expand_as(input_data)
#         input_minmax = (input_data - min_input) / (max_input - min_input) 

#         # 归一化label
#         label_data = torch.tensor(label[key].values).permute(1, 0, 2, 3)

#         masked_label = label_data.clone()
#         masked_label[torch.isnan(masked_label)] = -float('inf')
#         max_label, _ = torch.max(masked_label.reshape(masked_label.shape[0], -1), dim=1)
#         masked_label[torch.isnan(label_data)] = float('inf')
#         min_label, _ = torch.min(masked_label.reshape(masked_label.shape[0], -1), dim=1)

#         max_label, min_label = max_label.view(label_data.shape[0], 1, 1, 1).expand_as(label_data), min_label.view(label_data.shape[0], 1, 1, 1).expand_as(label_data)
#         label_minmax = (label_data - min_label) / (max_label - min_label) 

#         # 存储最值
#         minmax = [min_input, max_input, min_label, max_label]
        

#         # get sub
        
#         input_sub, lat_sub, lon_sub = self.get_sub(input_minmax, input['lat'].values, input['lon'].values)
#         label_sub, _, _ = self.get_sub(label_minmax, input['lat'].values, input['lon'].values)

#         return input_sub, label_sub, lat_sub, lon_sub, label['depth'].values, minmax
    
#     def get_sub(self, data, latitude, longitude):
#         """
#         提取子区域的数据

#         input:
#         lat_min, lat_max, lon_min, lon_max: 子区域范围
#         data: 原始数据
#         latitude, longitude: 经纬度数据

#         return: subset_data, subset_lat, subset_lon
#         """
#         # 找到对应的索引
#         # print('in sub: ', data.shape, latitude.shape, longitude.shape)
#         lat_indices = np.where((latitude >= self.lat_min) & (latitude <= self.lat_max))[0]
#         lon_indices = np.where((longitude >= self.lon_min) & (longitude <= self.lon_max))[0]
#         # 提取子集数据
#         subset_data = data[:, :, lat_indices, :][:, :, :, lon_indices]
#         # print(subset_data.shape)
#         # 提取相应的经纬度数组
#         subset_lat = latitude[lat_indices]
#         subset_lon = longitude[lon_indices]

#         return subset_data, subset_lat, subset_lon

#     def __len__(self):
#         return len(self.input)

#     def __getitem__(self, idx):
#         inputs = self.input[idx]   # (var, lat, lon) or (seq, var, lat, lon)
#         label = self.label[idx]    # (dept, lat, lon) or (seq, dept, lat, lon)
#         lat = self.lat
#         lon = self.lon
#         depth = self.depth
#         mask_x = self.mask_x[idx] 
#         mask_y = self.mask_y[idx]
        
#         return inputs.float(), label.float(), mask_x, mask_y, lon

