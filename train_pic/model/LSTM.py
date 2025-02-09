from torch import nn
import torch


class PermaDropout(nn.Module):
    def __init__(self, p):
        super(PermaDropout, self).__init__()
        self.p = p
    def forward(self, x):
        return nn.functional.dropout(x, p=self.p, training=True)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, n_units1, n_units2, dropout_fraction, activ):
        super(LSTMModel, self).__init__()
        self.dropout1 = PermaDropout(dropout_fraction)
        self.lstm1 = nn.LSTM(input_dim, n_units1, batch_first=True)
        self.dropout2 = PermaDropout(dropout_fraction)
        self.activation = getattr(nn, activ)()
        
        if n_units2 > 0:
            self.lstm2 = nn.LSTM(n_units1, n_units2, batch_first=True)
            self.dropout3 = PermaDropout(dropout_fraction)
            self.time_distributed = nn.Linear(n_units2, output_dim)
        else:
            self.time_distributed = nn.Linear(n_units1, output_dim)

    def forward(self, x):
        # print(111, x.shape)
        x = self.dropout1(x)

        x, _ = self.lstm1(x)
        # print(222,x.shape)
        x = self.activation(x)

        x = self.dropout2(x)

        # print(333, x.shape)
        
        if hasattr(self, 'lstm2'):
            x, _ = self.lstm2(x)
            x = self.activation(x)
            x = self.dropout3(x)
        # print(444, x.shape)
        x = self.time_distributed(x)
        # print(555, x.shape)
        return x
    
    def train_one_step(self, x, y, mask, minmax, criterion, k, metric_names={}):
        '''
        x:      [bs, seq, var, lat, lon]
        y:      [bs, seq, depth, lat, lon]
        mask:   [1, lat, lon]
        pred:   [bs, seq, n]

        return: loss
        '''
        # process data
        info = {}
        seq_x, var = x.shape[1], x.shape[2]
        bs, seq, depth, lat, lon = y.shape

        mask_x = mask.unsqueeze(1).unsqueeze(1).tile([1, seq_x, var, 1, 1]) 
        mask_y = mask.unsqueeze(1).unsqueeze(1).tile([1, seq, depth, 1, 1]).reshape(bs, seq, depth, -1)  # (1, lat, lon) --> (bs, seq, lat, lon, depth)
        # print(mask.shape, mask_x.shape, mask_y.shape, x.shape)
        x = torch.where(mask_x, x, 0.0).reshape(bs, seq_x, -1)  # (bs, seq, n)

        # pred
        pred = self(x)[:, -1, ...].unsqueeze(1).reshape(bs, seq, depth, -1)
        # print(pred.shape, mask_y.shape, y.shape)
        y = y.reshape(bs, seq, depth, -1)
        
        if metric_names:
            # Mask the predicted and true values
            masked_pred = torch.where(mask_y, pred, 0.0)
            masked_y = torch.where(mask_y, y, 0.0)
            
            # Count of valid elements for each batch
            valid_count = mask_y.sum(dim=-1)  # [bs, 1, k]

            # Compute loss
            loss = criterion(masked_pred, masked_y, metric_names, valid_count)
            
            # Reapply the mask to pred for returning
            pred = torch.where(mask_y, pred, torch.tensor(float('nan')))
            pred = pred.reshape(bs, seq, depth, lat, lon)[0, ...]
        else:        
            masked_pred = torch.masked_select(pred , mask_y )  # 掩码处理：直接删去了mask为false的值，得到一个一维数组
            masked_y = torch.masked_select(y , mask_y)
            # print(pred.shape, masked_pred.shape)
            loss = criterion(masked_pred, masked_y)

        return loss, pred, info
