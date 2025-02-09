import torch
import torch.nn as nn


# Gaussian Radial Basis Function Layer
class GaussianRBF(nn.Module):
    def __init__(self, input_dim):
        super(GaussianRBF, self).__init__()
        self.b = nn.Parameter(torch.randn(input_dim))  # Learnable bias

    def forward(self, x):
        pi = torch.pi
        diff = x - self.b
        return torch.exp(-pi * (diff ** 2))

# MLP Model with RBF Layers
class MLPE(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPE, self).__init__()
        self.embedding_lat = nn.Embedding(180, 16)
        self.embedding_lon = nn.Embedding(360, 16)
        self.embedding_data = nn.Embedding(323, 16)

        in_dim = 399600

        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, 128),
            GaussianRBF(128),
            nn.Linear(128, 256),
            nn.Dropout(0.1),
            GaussianRBF(256)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(256 + in_dim, 128),  # Concatenate original input with MLP1 output
            nn.Dropout(0.1),
            GaussianRBF(128),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        # x: [bs, var]
        sea_surface_data = x[:, :6]
        lat = torch.cat((self.embedding_lat(torch.trunc(x[:, 6]).long()), torch.frac(x[:, 6]).unsqueeze(-1)), -1)
        lon = torch.cat((self.embedding_lon(torch.trunc(x[:, 7]).long()), torch.frac(x[:, 7]).unsqueeze(-1)), -1)
        t1 = torch.cat((self.embedding_data(torch.trunc(x[:, 8]).long()), torch.frac(x[:, 8]).unsqueeze(-1)), -1)
        t2 = torch.cat((self.embedding_data(torch.trunc(x[:, 9]).long()), torch.frac(x[:, 9]).unsqueeze(-1)), -1)

        embedded_x = torch.cat((sea_surface_data, lat, lon, t1, t2), -1)
        
        # print('error: ', embedded_x.shape, x.shape)

        mlp1_output = self.mlp1(embedded_x.float())
        # print(embedded_x.shape, mlp1_output.shape)
        combined_x = torch.cat((embedded_x, mlp1_output), 1).float()
        # print(combined_x.shape)
        return self.mlp2(combined_x)
    
    def train_one_step(self, x, y, mask_x, mask_y, criterion, k, metric_names={}):
        '''
        x:      [bs, var]
        y:      [bs, depth]
        mask:   [bs, var]
        mask_y: [bs, depth]
        pred:   [bs, -1]

        return: loss, pred
        '''
        # process data
        info = {}
        bs, depth = y.shape
        # mask_x = mask.unsqueeeze(1).repat(1 ,x.shape[1], 1, 1)  # (bs, lat, lon) --> (bs, var, lat, lon)
        # mask_y = mask.unsqueeze(1).repeat(1 , depth, 1, 1).reshape(bs, depth, -1)  # (bs, lat, lon) --> (bs, depth, -1)
        
        # Apply mask to x 
        x = torch.where(mask_x, x, 0.0)
        x = x.reshape(bs, x.shape[1], -1)  # (bs, n)

        # pred
        pred = self(x)

        if metric_names:
            mask_y = mask_y.unsqueeze(1)

            # Count of valid elements for each batch
            valid_count = mask_y.sum(dim=-1)  # [bs, 1, depth]
            
            # Mask the predicted and true values
            masked_pred = torch.where(mask_y, pred.unsqueeze(1), 0.0)  # [bs, 1, depth, -1]
            masked_y = torch.where(mask_y, y.unsqueeze(1), 0.0)

            # Compute loss
            loss = criterion(masked_pred, masked_y, metric_names, valid_count)

            # Reapply the mask to pred for returning
            pred = torch.where(mask_y.reshape(bs, -1), pred.reshape(bs, -1), torch.tensor(float('nan')))
            pred = pred.reshape(bs, depth, lat, lon)
        else:
            masked_pred = torch.masked_select(pred, mask_y)  # mask：删去了mask为false的值，得到一个一维数组
            masked_y = torch.masked_select(y, mask_y)
            
            loss = criterion(masked_pred, masked_y)

        return loss, pred, info



# # MLP Model with RBF Layers
# class MLPE(nn.Module):
#     def __init__(self, input_size, out_dim):
#         super(MLPE, self).__init__()
#         self.embedding_lat = nn.Embedding(360, 8)
#         self.embedding_lon = nn.Embedding(720, 8) 
#         self.embedding_sst = nn.Embedding(5400, 8)
#         self.embedding_sss = nn.Embedding(5400, 8)

#         # 2、MLP 层的前半部分包括两个线性层和两个高斯 RBF 激活函数，用于组合输入的初步处理。MLP 层的后半部分处理前一个 MLP 层的输出和嵌入层的输出的串联，并进一步通过线性层和高斯 RBF 激活函数
#         # 虑到叶绿素的单峰分布，我们用高斯径向基函数

#         self.mlp1 = nn.Sequential(
#             nn.Linear(input_size, 640),
#             GaussianRBF(640),
#             nn.Linear(640, 1280),
#             nn.Dropout(0.1),
#             GaussianRBF(1280)
#         )

#         self.mlp2 = nn.Sequential(
#             nn.Linear(1280 + input_size, 640),  # Concatenate original input with MLP1 output
#             nn.Dropout(0.1),
#             GaussianRBF(640),
#             nn.Linear(640, out_dim)
#         )

#     def forward(self, x):
#         # # x [bs, var, n]
#         # print('----------------------------------------------------')
#         # print(x.shape)

#         # 1、为每一个特征进行编码，每个特征的整数部分都是嵌入和编码的，而小数部分保留为连续值，并与要传递给模型的嵌入向量连接。捕获了空间和时间特征的复杂关系
#         # # x2 = torch.cat((self.embedding_sst(torch.trunc(x[:, 2, :]).long()), torch.frac(x[:, 2, :]).unsqueeze(-1)), -1)
#         # # x4 = torch.cat((self.embedding_sss(torch.trunc(x[:, 4, :]).long()), torch.frac(x[:, 4, :]).unsqueeze(-1)), -1)
#         # x6 = self.embedding_lat(x[:, 6, :].long())
#         # x7 = self.embedding_lon(x[:, 7, :].long())
#         # # print(x2.shape, x6.shape)
#         # embedded_x = torch.cat(( x[:, 0, :].unsqueeze(-1), x[:, 1, :].unsqueeze(-1),x[:, 3, :].unsqueeze(-1), x[:, 5, :].unsqueeze(-1), x6, x7, x[:, -1, :].unsqueeze(-1)), -1)
#         # print(embedded_x.shape)
#         # embedded_x = embedded_x.reshape(bs, -1)
#         # print(embedded_x.shape)
        
#         # print('error: ', embedded_x.shape, x.shape)

#         mlp1_output = self.mlp1(x)
#         # print(embedded_x.shape, mlp1_output.shape)
#         # print(x.shape, mlp1_output.shape)
#         combined_x = torch.cat((x, mlp1_output), -1).float()
#         # print(combined_x.shape)
#         return self.mlp2(combined_x)
    
#     def train_one_step(self, x, y, mask, mask_y, criterion, k, metric_names={}):
#         '''
#         x:      [bs, var, lat, lon]
#         y:      [bs, depth, lat, lon]
#         mask:   [1, lat, lon]
#         pred:   [bs, -1]

#         return: loss
#         '''
#         # process data
#         info = {}

#         bs, depth, lat, lon = y.shape

#         mask_x = mask.unsqueeze(1).repeat(1 ,x.shape[1], 1, 1)  # (1, lat, lon) --> (bs, var, lat, lon)
#         mask_y = mask.unsqueeze(1).repeat(1 ,k, 1, 1).reshape(bs, -1)  # (1, lat, lon) --> (bs, depth, lat, lon)
        
#         # Apply mask to x 
#         # print('x: ', x.shape)
#         x = torch.where(mask_x, x, 0.0)
#         x = x.reshape(bs, -1)  # (bs, var, n)

        
#         pred = self(x)  # pred shape: (bs, var, lat, lon)
        
#         y = y[:, 0:k, ...].reshape(bs, -1)
#         # print(pred.shape, mask_y.shape, y.shape)
        
#         if metric_names:
#             masked_pred = torch.where(mask_y, pred, 0.0)
#             masked_y = torch.where(mask_y, y, 0.0)
#             masked_pred = masked_pred.reshape(bs, k, -1).unsqueeze(1)
#             masked_y = masked_y.reshape(bs, k, -1).unsqueeze(1)
#             # print(y.shape, pred.shape)  [bs, seq, depth, n]
#             loss = criterion(masked_pred, masked_y, metric_names)

#             pred = torch.where(mask_y, pred, torch.tensor(float('nan')))
#         else:
#             # 掩码处理：删去了mask为false的值，得到一个一维数组
#             masked_pred = torch.masked_select(pred, mask_y)
#             masked_y = torch.masked_select(y, mask_y)
 
#             loss = criterion(masked_pred, masked_y)

#         return loss, pred, info


