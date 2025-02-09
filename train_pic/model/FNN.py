from torch import nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np

class FFNN(nn.Module):
    def __init__(self, input_dim, output_dim, n_units1, n_units2, dropout_fraction, activ):
        super(FFNN, self).__init__()
        self.dropout_fraction = dropout_fraction
        self.activ = getattr(nn, activ)()
        
        self.fc1 = nn.Linear(input_dim, n_units1)
        self.fc2 = nn.Linear(n_units1, n_units2) if n_units2 > 0 else None
        
        self.fc3 = nn.Linear(n_units2 if n_units2 > 0 else n_units1, output_dim)

        # self.fc1 = nn.Linear(input_dim, 5000)
        # self.fc2 = nn.Linear(5000, 1000)
        # self.fc3 = nn.Linear(1000, 8000)
        # self.fc4 = nn.Linear(8000, output_dim)



    def forward(self, x):
        x = nn.functional.dropout(x, p=self.dropout_fraction, training=True)
        x = self.fc1(x)
        x = self.activ(x)
        x = nn.functional.dropout(x, p=self.dropout_fraction, training=True)
        

        if self.fc2 is not None:
            x = self.fc2(x)
            x = self.activ(x)
            x = nn.functional.dropout(x, p=self.dropout_fraction, training=True)

        x = self.fc3(x)

        return x
        
    def train_one_step(self, x, y, mask, minmax, criterion, metric_names={}):
        '''
        x:      [bs, var, lat, lon]
        y:      [bs, depth, lat, lon]
        mask:   [bs, lat, lon]
        pred:   [bs, -1]

        return: loss, pred
        '''
        # process data
        info = {}
        bs, depth, lat, lon = y.shape
        mask_x = mask.unsqueeze(1).repeat(1 ,x.shape[1], 1, 1)  # (bs, lat, lon) --> (bs, var, lat, lon)
        mask_y = mask.unsqueeze(1).repeat(1 , depth, 1, 1).reshape(bs, depth, -1)  # (bs, lat, lon) --> (bs, depth, -1)
        
        # Apply mask to x 
        x = torch.where(mask_x, x, 0.0)
        x = x.reshape(bs, -1)  # (bs, n)

        # pred
        pred = self(x).reshape(bs, depth, -1)  # (bs, -1) -> (bs, depth, -1)
        y = y.reshape(bs, depth, -1)

        # Denormalization
        min_label, max_label = minmax
        min_label = min_label.view(1, y.shape[1], 1).expand_as(y).to(y.device)
        max_label = max_label.view(1, y.shape[1], 1).expand_as(y).to(y.device)

        pred = pred * (max_label-min_label) + min_label
        y = y * (max_label-min_label) + min_label

        # # 绘制y的图像
        # # 绘制温度图
        # y_plot = y.reshape(bs, depth, lat, lon).cpu().detach().numpy()[0, 0, ...]
        # latitudes = np.linspace(-90, 90, y_plot.shape[1])
        # longitudes = np.linspace(0, 360, y_plot.shape[0])
        # plt.figure(figsize=(10, 6))
        # # plt.pcolormesh(longitudes, latitudes, y_plot, shading='auto', cmap='viridis')
        # plt.pcolormesh(longitudes, latitudes, y_plot.T, shading='auto', cmap='viridis')
        # plt.colorbar(label='')
        # plt.xlabel('Longitude')
        # plt.ylabel('Latitude')
        # plt.title(f'')
        # plt.savefig('./FNN_test.jpg')
        # raise ValueError("11111111111")


        if metric_names:
            # add sequence dimension
            mask_y = mask_y.unsqueeze(1)
            # Count of valid elements for each batch
            valid_count = mask_y.sum(dim=-1)  # [bs, 1, depth]
            # Mask the predicted and true values
            pred_masked = torch.where(mask_y, pred.unsqueeze(1), 0.0)  # [bs, 1, depth, -1]
            y_masked = torch.where(mask_y, y.unsqueeze(1), 0.0)

            # Compute loss
            loss = criterion(pred_masked, y_masked, metric_names, valid_count)

            # Reapply the mask to pred for returning
            pred = torch.where(mask_y.reshape(bs, -1), pred.reshape(bs, -1), torch.tensor(float('nan')))
            pred = pred.reshape(bs, depth, lat, lon)
        else:
            pred_masked = torch.masked_select(pred.reshape(bs, -1), mask_y.reshape(bs, -1))  # mask：删去了mask为false的值，得到一个一维数组
            y_masked = torch.masked_select(y.reshape(bs, -1), mask_y.reshape(bs, -1))
            
            loss = criterion(pred_masked, y_masked)

        return loss, pred, info
