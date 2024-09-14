from torch import nn


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
        x = self.dropout1(x)
        x, _ = self.lstm1(x)
        x = self.activation(x)
        x = self.dropout2(x)
        
        if hasattr(self, 'lstm2'):
            x, _ = self.lstm2(x)
            x = self.activation(x)
            x = self.dropout3(x)
        
        x = self.time_distributed(x)
        return x
    
    def train_one_step(self, x, y, mask, criterion, k, metric_names={}):
        '''
        x:      [bs, seq, depth, lat, lon]
        y:      [bs, seq, depth, lat, lon]
        mask:   [1, lat, lon]
        pred:   [bs, seq, n]

        return: loss
        '''
        # process data
        info = {}
        # print('x, y: ', x.shape, y.shape)
        bs_1 = x.shape[0]
        bs_2, seq, depth, lat, lon = y.shape
        x = x.reshape(bs_1, seq, -1)  # (bs, seq, n)
        y = y[:, :, 0:k, ...]
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(bs_2 ,seq, k, 1, 1)  # (1, lat, lon) --> (bs, seq, depth, lat, lon)
        y = y*mask
        
        pred = self(x)  # (bs, seq, n)

        if metric_names:
            pred = pred.reshape(bs_2, seq, k, lat, lon)
            pred = pred * mask
            pred = pred.reshape(bs_2, seq, k, -1)
            y = y.reshape(bs_2, seq, k, -1)
            loss = criterion(pred, y, metric_names)
        else:        
            pred = pred.reshape(bs_2, seq, k, lat, lon)
            pred = pred * mask
            loss = criterion(pred, y)

        return loss, pred, info


# LSTM输入数据：[batch_size, seq_len, input_size]
    
# LSTMModel(
#   (dropout1): PermaDropout()
#   (lstm1): LSTM(7, 35, batch_first=True)
#   (dropout2): PermaDropout()
#   (activation): Tanh()
#   (lstm2): LSTM(35, 35, batch_first=True)
#   (dropout3): PermaDropout()
#   (time_distributed): Linear(in_features=35, out_features=3, bias=True)
# )