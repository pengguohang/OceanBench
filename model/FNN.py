from torch import nn
import torch.nn.functional as F
import torch

class FFNN(nn.Module):
    def __init__(self, input_dim, output_dim, n_units1, n_units2, dropout_fraction, activ):
        super(FFNN, self).__init__()
        self.dropout_fraction = dropout_fraction
        self.activ = getattr(nn, activ)()
        
        self.fc1 = nn.Linear(input_dim, n_units1)
        self.fc2 = nn.Linear(n_units1, n_units2) if n_units2 > 0 else None
        self.fc3 = nn.Linear(n_units2 if n_units2 > 0 else n_units1, output_dim)

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
        
    def train_one_step(self, x, y, mask, criterion, k):
        '''
        x:      [bs, var, lat, lon]
        y:      [bs, depth, lat, lon]
        mask:   [1, lat, lon]
        pred:   [bs, -1]

        return: loss
        '''
        # process data
        info = {}
        bs_1 = x.shape[0]
        bs_2, depth, lat, lon = y.shape
        x = x.reshape(bs_1, -1)  # (bs, n)
        y = y[:, 0:k, ...]
        mask = mask.unsqueeze(0).repeat(bs_2 ,k, 1, 1)  # (1, lat, lon) --> (bs, depth, lat, lon)
        y = y*mask
        
        pred = self(x)  # (bs, n)
        
        pred = pred.reshape(bs_2, k, lat, lon)
        pred = pred * mask
        loss = criterion(pred, y)

        return loss, pred, info
