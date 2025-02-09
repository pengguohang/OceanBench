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
        
    def train_one_step(self, x, y, mask, _, criterion, k, metric_names={}):
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
        x = x.reshape(bs, -1)  # (bs, n)

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
