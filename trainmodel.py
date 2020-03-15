from coinmodel import coinmodel, lossfuncs as lfuncs
from torch.utils.data import DataLoader
from datasets import rippleDataset
from torch.optim import Adam
from tqdm import tqdm
import argparse
import torch
import sys
import os

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Test ripplemodel')
    parser.add_argument('--log', dest='logpath',
                        type=str)
    parser.add_argument('--resume_train', dest='modelpath',
                        type=str, default=False)

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
    lossfuncs = lfuncs()
    stage = 1

    '''Resume training'''
    # if args.modelpath:
    #     model.load_state_dict(torch.load(args.modelpath)['model_state_dict'])
    #     c = True
    # else: c = False

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Stage 1: Training Percentage Yield')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    lossfunc = lossfuncs.stage_1
    optimizer = Adam(model.parameters(), lr=0.01)

    pbar = tqdm(total=len(traindata), leave=False)
    losses = []
    for epoch in range(5):
        for x, y in traindata:
            out = model(x)
            optimizer.zero_grad()
            loss = lossfunc(out, y)
            losses.append(loss)
            if len(losses) == 5:
                loss.data = torch.mean(torch.tensor(losses))
                loss.backward()
                optimizer.step()
                pbar.set_description('stage {} epoch {} iter {} | train loss: %.5f, len: {}'.format(stage, epoch, pbar.n, x.shape) % (loss))
                pbar.update()

                if pbar.n % 1000 == 0:
                    modelpath = os.path.join('model/ripplemodel_stage{}_epoch{}_iter{}'.format(stage, epoch, pbar.n))
                    torch.save({'stage': stage,
                                'model_state_dict': model.state_dict(),
                                'iter':pbar.n},
                               modelpath)

                losses = []

    stage += 1

    pbar.close()
    modelpath = os.path.join('model/ripplemodel_stage{}'.format(1))
    torch.save({'stage': stage,
                'model_state_dict': model.state_dict(),
                'iter': 0},
                modelpath)

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Stage 2: Training Real Value')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    lossfunc = lossfuncs.stage_2
    optimizer = Adam(model.parameters(), lr=0.01)

    pbar = tqdm(total=len(traindata), leave=False)
    losses = []
    for epoch in range(5):
        for x, y in traindata:
            out = model(x)
            optimizer.zero_grad()
            loss = lossfunc(out, y)*100
            losses.append(loss)
            if len(losses) == 5:
                loss.data = torch.mean(torch.tensor(losses))
                loss.backward()
                optimizer.step()
                pbar.set_description('stage {} epoch {} iter {} | train loss: %.5f, len: {}'.format(stage, epoch, pbar.n, x.shape) % (loss))
                pbar.update()

                if pbar.n % 1000 == 0:
                    modelpath = os.path.join('model/ripplemodel_stage{}_epoch{}_iter{}'.format(stage, epoch, pbar.n))
                    torch.save({'stage': stage,
                                'model_state_dict': model.state_dict(),
                                'iter':pbar.n},
                               modelpath)

                losses = []

    stage += 1

    pbar.close()
    modelpath = os.path.join('model/ripplemodel_stage{}'.format(1))
    torch.save({'stage': stage,
                'model_state_dict': model.state_dict(),
                'iter': 0},
                modelpath)

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Stage 3: Training Hard Part')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    lossfunc = lossfuncs.stage_3
    optimizer = Adam(model.parameters(), lr=0.01)

    pbar = tqdm(total=len(traindata), leave=False)
    losses = []
    for epoch in range(5):
        for x, y in traindata:
            out = model(x)
            optimizer.zero_grad()
            loss = lossfunc(out, y)*100
            losses.append(loss)
            if len(losses) == 5:
                loss.data = torch.mean(torch.tensor(losses))
                loss.backward()
                optimizer.step()
                pbar.set_description('stage {} eoch {} iter {} | train loss: %.5f, len: {}'.format(stage, epoch, pbar.n, x.shape) % (loss))
                pbar.update()

                if pbar.n % 1000 == 0:
                    modelpath = os.path.join('model/ripplemodel_stage{}_epoch{}_iter{}'.format(stage, epoch, pbar.n))
                    torch.save({'stage': stage,
                                'model_state_dict': model.state_dict(),
                                'iter':pbar.n},
                               modelpath)

                losses = []

    pbar.close()
    modelpath = os.path.join('model/ripplemodel_final')
    torch.save({'stage': None,
                'model_state_dict': model.state_dict(),
                'iter': None},
                modelpath)

if __name__=='__main__':
    main()