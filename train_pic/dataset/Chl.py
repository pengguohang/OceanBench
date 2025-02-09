import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset, IterableDataset
import xarray as xr


class Interpolated_Img_Dataset(Dataset):
    
    def __init__(self, root_folder, input_file, output_file , transform=None, normalize=False):
        """
        Args:
            root_folder (String): path to input and output files
            input_file (String): npy file of the input data to be used by the NN
            output_file (String): npy file of the output data to be use by the NN
            transform (callable, Optional): Optional transform to be applied on
            a sample
        """
        self.root_folder = root_folder
        self.input_arr = np.load(os.path.join(self.root_folder, input_file ))
        self.output_arr = np.load(os.path.join(self.root_folder, output_file))
        self.transform = transform
        self.normalize = normalize
        self.mean_input = np.mean(self.input_arr, axis=(0,1,2))
        self.std_input = np.std(self.input_arr, axis=(0,1,2))
        self.mean_output = np.mean(self.output_arr)
        self.std_output = np.std(self.output_arr)        
        
    def __len__(self):

        return self.input_arr.shape[0]
    
    def __getitem__(self, idx):

        X = self.input_arr[idx,...]
        Y = self.output_arr[idx,...]
        
        if self.normalize:
            for ch in range(13):
                X[:,ch] = (X[:,ch] - self.mean_input[ch])/ self.std_input[ch]   ################### ch ??? 13 ??
            Y = (Y - self.mean_output)/ self.std_output
            
        if self.transform:
            X =  self.transform(X)
            Y =  self.transform(Y)
        
        return X,Y