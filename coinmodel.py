from torch.nn import Module, GRU, Linear, Dropout, Tanh

class coinmodel(Module):
    def __init__(self):
        super(coinmodel, self).__init__()

        self.linear1 = Linear(6, 12)
        self.GRU1 = GRU(12, 40, num_layers=1, bias=True, batch_first=True, bidirectional=True)
        self.GRU2 = GRU(80, 70, num_layers=1, bias=True, batch_first=True, bidirectional=True)
        self.GRU3 = GRU(140, 70, num_layers=1, bias=True, batch_first=True, bidirectional=True)
        self.GRU4 = GRU(140, 40, num_layers=1, bias=True, batch_first=True, bidirectional=True)
        self.linear2 = Linear(80, 1)

    def forward(self, x):
        x = self.linear1(x)
        x, (hidden, cell) = self.GRU1(x)
        x = Dropout(0.4)(x)
        x, (hidden, cell) = self.GRU2(x)
        x = Dropout(0.3)(x)
        x, (hidden, cell) = self.GRU3(x)
        x = Dropout(0.4)(x)
        x, (hidden, cell) = self.GRU4(x)
        x = Dropout(0.4)(x)
        x = self.linear2(x)
        x = Tanh()(x)
        return x

class coinmodel_test(Module):
    def __init__(self):
        super(coinmodel_test, self).__init__()

        self.linear1 = Linear(6, 12)
        self.GRU1 = GRU(12, 40, num_layers=1, bias=True, batch_first=True, bidirectional=True)
        self.GRU2 = GRU(80, 70, num_layers=1, bias=True, batch_first=True, bidirectional=True)
        self.GRU3 = GRU(140, 70, num_layers=1, bias=True, batch_first=True, bidirectional=True)
        self.GRU4 = GRU(140, 40, num_layers=1, bias=True, batch_first=True, bidirectional=True)
        self.linear2 = Linear(80, 1)

    def forward(self, x):
        x = self.linear1(x)
        x, (hidden, cell) = self.GRU1(x)
        x, (hidden, cell) = self.GRU2(x)
        x, (hidden, cell) = self.GRU3(x)
        x, (hidden, cell) = self.GRU4(x)
        x = self.linear2(x)
        x = Tanh()(x)
        return x