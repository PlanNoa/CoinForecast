import torch.utils.data as data
import os
import torch
import json

class rippleDataset(data.Dataset):
    def __init__(self, data_dir):
        super(rippleDataset, self).__init__()
        self.data_dir = os.path.expanduser(data_dir)
        with open(data_dir, "r") as json_file:
            rawdata = json.load(json_file)

        self.rawdata = [[1, 1, 1, 1, 1, 1]]
        self.rawlabel = [[1]]
        for log in range(1, len(rawdata)):
            self.rawdata.append([(rawdata[log]['openingPrice']-rawdata[log-1]['openingPrice'])/rawdata[log-1]['openingPrice'],
                                 (rawdata[log]['highPrice']-rawdata[log-1]['highPrice'])/rawdata[log-1]['highPrice'],
                                 (rawdata[log]['lowPrice']-rawdata[log-1]['lowPrice'])/rawdata[log-1]['lowPrice'],
                                 (rawdata[log]['tradePrice']-rawdata[log-1]['tradePrice'])/rawdata[log-1]['tradePrice'],
                                 (rawdata[log]['candleAccTradeVolume']-rawdata[log-1]['candleAccTradeVolume'])/rawdata[log-1]['candleAccTradeVolume'],
                                 (rawdata[log]['candleAccTradePrice']-rawdata[log-1]['candleAccTradePrice'])/rawdata[log-1]['candleAccTradePrice']])
            self.rawlabel.append([(rawdata[log]['tradePrice']-rawdata[log-1]['tradePrice'])/rawdata[log-1]['tradePrice']])
        self.rawdata = list(reversed(self.rawdata))
        self.rawlabel = list(reversed(self.rawlabel))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index,):
        return torch.Tensor(self.data[index]), torch.Tensor(self.label[index])

    def parsedata(self):
        self.data = []
        self.label = []
        for n in range(200, 250):
            for i in range(n, len(self.rawdata)-1):
                self.data.append(self.rawdata[i-n:i])
                self.label.append(self.rawlabel[i-n+1:i+1])

if __name__ == '__main__':
    dataset = rippleDataset('data.json')
    dataset.parsedata()