from torch.nn import Module, GRU, Linear, Dropout, Tanh, MSELoss
import torch
import numpy as np

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

class lossfuncs:
    def __init__(self, init_value=100):
        self.init_value = init_value
        self.mseloss = MSELoss()
        return

    def stage_1(self, out, y):
        loss = self.mseloss(out*1000, y*1000)
        return loss

    def stage_2(self, out, y):
        loss = self.mseloss(out, y)
        labelprice = [self.init_value]
        outputprice = [self.init_value]
        for i, j in zip(y.flatten(), out.flatten()):
            labelprice.append((1 + j) * labelprice[-1])
            outputprice.append((1 + i) * outputprice[-1])
        labelprice = torch.Tensor(labelprice)
        outputprice = torch.Tensor(outputprice)
        loss.data = self.mseloss(outputprice, labelprice)
        return loss

    def stage_3(self, out, y):
        loss = self.mseloss(out, y)
        labelprice = [self.init_value]
        outputprice = [self.init_value]
        for i, j in zip(y.flatten(), out.flatten()):
            labelprice.append((1 + j) * labelprice[-1])
            outputprice.append((1 + i) * outputprice[-1])
        loss.data = torch.tensor(max(np.abs(np.array(labelprice) - np.array(outputprice))))
        return loss