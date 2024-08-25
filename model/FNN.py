from torch import nn


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