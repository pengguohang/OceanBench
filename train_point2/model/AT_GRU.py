# 离散为点，单词epoch时间太长
# Time: 2146.86s


import torch.nn as nn
import torch

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, depth):
        super(Attention, self).__init__()
        self.query_linear = nn.Linear(input_size, hidden_size)
        self.key_linear = nn.Linear(input_size, hidden_size)
        self.value_linear = nn.Linear(input_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.depth_linear = nn.Linear(hidden_size, depth)
    def forward(self, query, keys, values):
        query = self.query_linear(query)
        keys = self.key_linear(keys)
        values = self.value_linear(values)
        scores = torch.matmul(query, keys.transpose(-2, -1))
        scores = self.softmax(scores)
        attended_values = torch.matmul(scores, values)
        output = self.depth_linear(attended_values)
        return output


class AT_GRU(nn.Module):

    def __init__(self, num_layers, rnn_hidden_size, encoder_input_size, encoder_hidden_size, out_dim):
        super(AT_GRU, self).__init__()
        in_dim = 399600
        encoder_input_size = 399600
        
        self.depth = out_dim
        self.num_layers = num_layers
        self.rnn_hidden_size = rnn_hidden_size

        self.embedding_lat = nn.Embedding(180, 16)
        self.embedding_lon = nn.Embedding(360, 16)
        self.embedding_data = nn.Embedding(323, 16)

        self.attention = Attention(encoder_input_size, encoder_hidden_size, self.depth)  
        self.line = nn.Linear(in_dim, self.depth) 
        self.dropout = nn.Dropout(p=0.1 )
        self.bi_gru = nn.GRU(input_size=1, hidden_size=rnn_hidden_size,num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(rnn_hidden_size * 2 * self.depth, out_dim)
        self.flatten = nn.Flatten(1, -1)

    def forward(self, x):

        bs = x.shape[0]
        sea_surface_data = x[:, :6, :].permute(0, 2, 1) 
        lat = torch.cat((self.embedding_lat(torch.trunc(x[:, 6, :]).long()), torch.frac(x[:, 6, :]).unsqueeze(-1)), -1)
        lon = torch.cat((self.embedding_lon(torch.trunc(x[:, 7, :]).long()), torch.frac(x[:, 7, :]).unsqueeze(-1)), -1)
        t1 = torch.cat((self.embedding_data(torch.trunc(x[:, 8, :]).long()), torch.frac(x[:, 8, :]).unsqueeze(-1)), -1)
        t2 = torch.cat((self.embedding_data(torch.trunc(x[:, 9, :]).long()), torch.frac(x[:, 9, :]).unsqueeze(-1)), -1)

        x = torch.cat((sea_surface_data, lat, lon, t1, t2), -1).permute(0, 2, 1).reshape(bs, -1)

        attention_output = self.attention(x.float(), x.float(), x.float())

        x = x.to(torch.float32)
        x = self.line(x)
        x = x + attention_output
        x = self.dropout(x)
        # gru
        output, h = self.bi_gru(x.unsqueeze(-1))
        # output
        output = self.flatten(output)
        output = self.fc(output)
        return output
    

    def train_one_step(self, x, y, mask, mask_y, criterion, k, metric_names={}):
        '''
        x:      [bs, var, lat, lon]
        y:      [bs, depth, lat, lon]
        mask:   [1, lat, lon]
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
        x = x.reshape(bs, x.shape[1], -1)  # (bs, n)

        # pred
        pred = self(x).reshape(bs, depth, -1)  # pred shape: (bs, -1)
        y = y.reshape(bs, depth, -1)

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
            masked_pred = torch.masked_select(pred.reshape(bs, -1), mask_y.reshape(bs, -1))  # mask：删去了mask为false的值，得到一个一维数组
            masked_y = torch.masked_select(y.reshape(bs, -1), mask_y.reshape(bs, -1))
            
            loss = criterion(masked_pred, masked_y)

        return loss, pred, info
    
    # def train_one_step(self, x, y, mask, mask_y, criterion, k, metric_names={}):
    #     '''
    #     x:      [bs, var, lat, lon]
    #     y:      [bs, depth, lat, lon]
    #     mask:   [1, lat, lon]
    #     pred:   [bs, -1]

    #     return: loss
    #     '''
    #     # process data
    #     info = {}

    #     bs, depth, lat, lon = y.shape

    #     mask_x = mask.unsqueeze(1).repeat(1 ,x.shape[1], 1, 1)  # (1, lat, lon) --> (bs, var, lat, lon)
    #     mask_y = mask.unsqueeze(1).repeat(1 ,k, 1, 1).reshape(bs, -1)  # (1, lat, lon) --> (bs, depth, lat, lon)
        
    #     # Apply mask to x 
    #     # print('x: ', x.shape)
    #     x = torch.where(mask_x, x, 0.0)
    #     x = x.reshape(bs, -1)  # (bs, var, n)

        
    #     pred = self(x)  # pred shape: (bs, var, lat, lon)
        
    #     y = y[:, 0:k, ...].reshape(bs, -1)
    #     # print(pred.shape, mask_y.shape, y.shape)
        
    #     if metric_names:
    #         masked_pred = torch.where(mask_y, pred, 0.0)
    #         masked_y = torch.where(mask_y, y, 0.0)
    #         masked_pred = masked_pred.reshape(bs, k, -1).unsqueeze(1)
    #         masked_y = masked_y.reshape(bs, k, -1).unsqueeze(1)
    #         # print(y.shape, pred.shape)  [bs, seq, depth, n]
    #         loss = criterion(masked_pred, masked_y, metric_names)

    #         pred = torch.where(mask_y, pred, torch.tensor(float('nan')))
    #     else:
    #         # 掩码处理：删去了mask为false的值，得到一个一维数组
    #         masked_pred = torch.masked_select(pred, mask_y)
    #         masked_y = torch.masked_select(y, mask_y)
 
    #         loss = criterion(masked_pred, masked_y)

    #     return loss, pred, info

    


