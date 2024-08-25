import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset, IterableDataset
import xarray as xr

class STDataset(Dataset):
    def __init__(self,
                 region_name = 'Gulf',
                 folder_path='../data/',
                 reference_file = '../', 
                 label_path = '../',
                 lat_min = 23,
                 lat_max = 50, 
                 lon_min = -80,
                 lon_max = -30,
                 key = 'so'
                 ):
        '''
        Args:
            region_name(str) : 提取的数据范围(Gulf )
            folder_path(str) : 存放所有数据的文件夹 , "/home/data2/pengguohang/My_Ocean/challenge"
            reference_file(str): 数据处理时的参考文件(参考mask 分辨率等)
            lat_min, lat_max(int) : 纬度范围
            lon_min, lon_max(int) : 经度范围
            key(str) : SS(so), ST(st)
        Returns:
            input, label, lat, lon, depth
        shape:
            (var, month, lat, lon), (depth, month, lat, lon), (x, y, p), (x, y, 2), (36)

            lat_min, lat_max, lon_min, lon_max, data, latitude, longitude
        '''
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max

        self.input = self.get_input_data(folder_path, reference_file)
        self.label, self.lat, self.lon, self.depth= self.get_armor(label_path, key)
        
        self.input = torch.from_numpy(self.input.values).permute(1,0,2,3)
        self.label = torch.from_numpy(self.label.values).permute(1,0,2,3)
        self.lat = torch.from_numpy(self.lat.values)
        self.lon = torch.from_numpy(self.lon.values)
        self.depth = torch.from_numpy(self.depth.values)

    def get_sub(self, lat_min, lat_max, lon_min, lon_max, data, latitude, longitude):
        """
        提取子区域的数据

        input:
        lat_min, lat_max, lon_min, lon_max: 子区域范围
        data: 原始数据
        latitude, longitude: 经纬度数据

        return: subset_data, subset_lat, subset_lon
        """
        # 找到对应的索引
        lat_indices = np.where((latitude >= lat_min) & (latitude <= lat_max))[0]
        lon_indices = np.where((longitude >= lon_min) & (longitude <= lon_max))[0]
        # 提取子集数据
        subset_data = data[:, lat_indices, :][:, :, lon_indices]
        # 提取相应的经纬度数组
        subset_lat = latitude[lat_indices]
        subset_lon = longitude[lon_indices]

        return subset_data, subset_lat, subset_lon

    def compute_climatological_mean_and_anomalies(self, data):
        """
        计算每个变量的气候学平均值, 从而计算其异常值
        input: data (xarray.Dataset or xarray.DataArray): 包含多个变量的时间序列数据，维度为 (time, lat, lon)。
        return: xarray.Dataset or xarray.DataArray: 包含异常值的数据集，维度为 (time, lat, lon)。
        """
        # 时间维度名为 'time'
        # print("Dimensions of data:", data.dims)
        
        # 计算气候学平均值（沿着time维度求平均）
        clim_mean = data.mean(dim='time')
        
        # 扩展气候学平均值，使其具有与原始数据相同的 time 维度
        clim_mean_expanded = clim_mean.broadcast_like(data)
        
        # 从原始数据中减去气候学平均值得到异常值
        anomalies = data - clim_mean_expanded
        
        return anomalies
    def min_max(self, data):
        """
        对输入数据按变量进行归一化

        input:(var, time, lat, lon)
        output: (var, time, lat, lon)
        """
        minmax = []
        for i in range(data.shape[0]):
            var_data = data[i]
            var_min = var_data.min(dim='time')
            var_max = var_data.max(dim='time')
            normalized_var_data = (var_data - var_min) / (var_max - var_min)
            minmax.append(normalized_var_data)
            # normalized_data.loc[dict(var=var)] = normalized_var_data

        minmax = xr.concat(minmax, dim='file')
        return minmax
    def get_input_data(self, folder_path, reference_file):
        """
        提取输入数据并裁剪
        folder_path, reference_file: 数据文件夹地址 及 参考数据文件地址
        
        return:  (var, time, lat, lon)
        """
        # 1、提取文件名
        nc_files = [file for file in os.listdir(folder_path) if file.endswith('.nc')]
        # 存储数据
        data_all = []
        # 先加载reference data, 作为网格插值的基准
        ref_ds = xr.open_dataset(reference_file)
        ref_lat = ref_ds['lat']
        ref_lon = ref_ds['lon']
        ref_data = ref_ds['data'][0:109, ...]

        # 2、提取子区域
        ref_subset_data, ref_subset_lat, ref_subset_lon = self.get_sub(self.lat_min, self.lat_max, self.lon_min, self.lon_max, ref_data, ref_lat, ref_lon)
        # 将 -999.0 的值转换为 np.nan
        mask = np.where(ref_subset_data == -999.0, np.nan, ref_subset_data)
        ref_subset_data = xr.DataArray(mask, dims=["time", "lat", "lon"], coords={"lat": ref_subset_lat, "lon": ref_subset_lon})
        data_all.append(ref_subset_data)

        # 3、逐个加载.nc文件并进行插值
        for file in nc_files:
            file_path = os.path.join(folder_path, file)
            print(f"Processing file: {file_path}")

            ds = xr.open_dataset(file_path)
            lat = ds['lat'][:]
            lon = ds['lon'][:]
            data = ds['data'][:109, ...]  # 提取前109个时间步的数据
            
            # 提取子区域
            subset_data, subset_lat, subset_lon = self.get_sub(self.lat_min, self.lat_max, self.lon_min, self.lon_max, data, lat, lon)
            # 将 'data' 插值到目标经纬度网格
            interpolated_var = xr.DataArray(subset_data, dims=["time", "lat", "lon"], coords={"lat": subset_lat, "lon": subset_lon}).interp(lat=ref_subset_lat, lon=ref_subset_lon)
            
            # 掩码处理：通过reference的nan值将所有数据相同位置的数字换为nan
            nan_mask = np.isnan(ref_subset_data)
            # print(nan_mask)
            masked_data = np.where(nan_mask, np.nan, interpolated_var)
            masked_data = xr.DataArray(masked_data, dims=["time", "lat", "lon"], coords={"lat": ref_subset_lat, "lon": ref_subset_lon})
            
            data_all.append(masked_data)


        # 将所有插值后的数据堆叠在一起
        data_all = xr.concat(data_all, dim='file')

        # 将数据中绝对值大于100的数值替换为NaN
        data_all = data_all.where(np.abs(data_all) <= 100, np.nan)

        # 计算数据异常值 - 减去 climatological mean
        data_all = self.compute_climatological_mean_and_anomalies(data_all)

        # 最大最小归一化
        data_all = self.min_max(data_all)

        print('shape of region:', data_all.shape)
        return data_all

    
    def get_armor(self, path, key):
        '''
        提取label

        armor数据如下:
        depth (36,)
        latitude (688,)
        longitude (1439,)
        time (313,)
        mlotst (313, 688, 1439)
        so (313, 36, 688, 1439)
        to (313, 36, 688, 1439)
        
        return: (depth, time, lat, lon)
        '''
        f = xr.open_dataset(path, chunks={'time': 1})
        data = f[key][204:313, ...]
        depth = f['depth']
        lat = f['latitude']
        lon = f['longitude']
        # print('begin:', data.shape)
        # 找到对应的索引
        lat_indices = np.where((lat >= self.lat_min) & (lat <= self.lat_max))[0]
        lon_indices = np.where((lon >= self.lon_min) & (lon <= self.lon_max))[0]
        # print('lat,lon:', lat_indices.shape, lon_indices.shape)

        # 提取子集数据
        subset_data = data[:, :, lat_indices, lon_indices].transpose('depth', 'time',  'latitude', 'longitude')
        # print('end:', subset_data.shape)

        # 提取相应的经纬度数组
        subset_lat = lat[lat_indices]
        subset_lon = lon[lon_indices]

        # 计算数据异常值 - 减去 climatological mean
        # print(subset_data.dims)
        subset_data = self.compute_climatological_mean_and_anomalies(subset_data)

        # minmax归一化
        subset_data = self.min_max(subset_data)

        # print('return:', subset_data.shape)

        return subset_data, subset_lat, subset_lon, depth
    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        inputs = self.input[idx]   # (var, lat, lon)
        label = self.label[idx]    # (dept, lat, lon)
        lat = self.lat
        lon = self.lon
        depth = self.depth
        # print(inputs.shape, label.shape, mask.shape, case_params.shape, grid.shape, case_id)
        return inputs, label, lat, lon, depth