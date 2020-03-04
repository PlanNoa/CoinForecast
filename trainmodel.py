import torch
from torch.nn import MSELoss
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
import os
from datasets import rippleDataset
from coinmodel import coinmodel
import matplotlib.pyplot as plt
import argparse
import sys

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Test ripplemodel')
    parser.add_argument('--log', dest='logpath',
                        type=str)
    parser.add_argument('--resume_train', dest='modelpath',
                        type=str, default=False)
    parser.add_argument('--vis', dest='vis',
                        type=bool, default=False)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    dataset = rippleDataset(args.logpath)
    dataset.parsedata()
    traindata = DataLoader(dataset, shuffle=True)

    model = coinmodel()

    lossfunc = MSELoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    if args.modelpath:
        model.load_state_dict(torch.load(args.modelpath)['model_state_dict'])
        c = True
    else: c = False

    epoch = 0
    while epoch < 10:
        print('epoch: {}'.format(epoch))
        pbar = tqdm(total=len(traindata), leave=False)
        if c:
            pbar.n = torch.load(args.modelpath)['iter']
            epoch = torch.load(args.modelpath)['epoch']
            c = False
        for x, y in traindata:

            out = model(x)
            labelprice = [100]
            outputprice = [100]

            for i, j in zip(y.flatten(), out.flatten()):
                labelprice.append((1+j)*labelprice[-1])
                outputprice.append((1 + i) * outputprice[-1])
            labelprice = torch.Tensor(labelprice)
            outputprice = torch.Tensor(outputprice)

            optimizer.zero_grad()
            loss = lossfunc(out, y)
            ldata = (lossfunc(outputprice, labelprice)+lossfunc(out, y))/2
            loss.data = ldata
            loss.backward()
            optimizer.step()
            pbar.set_description('iter {} | train loss: %.5f, len: {}'.format(pbar.n, x.shape) % (loss))
            pbar.update()

            if pbar.n % 500 == 0:
                modelpath = os.path.join('ripplemodel_epoch{}_iter{}'.format(epoch, pbar.n))
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'iter':pbar.n},
                           modelpath)

                if args.vis:
                    plt.plot(outputprice.flatten())
                    plt.plot(labelprice.flatten())
                    plt.savefig('{}-{}.png'.format(epoch, pbar.n))
                    plt.show()

        epoch + 1

        pbar.close()
        modelpath = os.path.join('ripplemodel_epoch%d' % epoch)
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'iter': 0},
                    modelpath)