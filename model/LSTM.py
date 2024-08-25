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
