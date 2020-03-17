from coinmodel import coinmodel_test
import torch
import json
import argparse
import sys

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Test ripplemodel')
    parser.add_argument('--log', dest='logpath',
                        type=str)
    parser.add_argument('--model', dest='modelpath',
                        type=str)
    parser.add_argument('--vis', dest='vis',
                        type=bool, default=False)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    model = torch.load(args.modelpath)

    ripplemodel = coinmodel_test()
    ripplemodel.load_state_dict(model['model_state_dict'])

    with open(args.logpath, "r") as json_file:
        rawdata = json.load(json_file)
    data = [[(rawdata[log]['openingPrice'] - rawdata[log + 1]['openingPrice']) / rawdata[log + 1]['openingPrice'],
             (rawdata[log]['highPrice'] - rawdata[log + 1]['highPrice']) / rawdata[log + 1]['highPrice'],
             (rawdata[log]['lowPrice'] - rawdata[log + 1]['lowPrice']) / rawdata[log + 1]['lowPrice'],
             (rawdata[log]['tradePrice'] - rawdata[log + 1]['tradePrice']) / rawdata[log + 1]['tradePrice'],
             (rawdata[log]['candleAccTradeVolume'] - rawdata[log + 1]['candleAccTradeVolume']) / rawdata[log + 1][
                 'candleAccTradeVolume'],
             (rawdata[log]['candleAccTradePrice'] - rawdata[log + 1]['candleAccTradePrice']) / rawdata[log + 1][
                 'candleAccTradePrice']] for log in range(len(rawdata) - 2, -1, -1)]

    data = torch.Tensor([data])
    output = ripplemodel(data)
    label = [319]
    pred = [319, 318]
    for i, j in zip(data[0], output[0]):
        label.append(label[-1] * (1 + i[2].item()))
        # pred.append(label[-2] * (1 + j.item()))
        pred.append(pred[-1][-2] * (1 + j.item()))
    label.append(pred[-1])

    if args.vis:
        import matplotlib.pyplot as plt
        plt.plot(label)
        plt.plot(pred)
        plt.savefig(args.modelpath.split('/')[-1] + '.png')
        plt.show()

if __name__ == '__main__':
    main()
