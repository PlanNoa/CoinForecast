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

    msg = ["Stage 1: Training Percentage Yield",
           "Stage 2: Training Real Value",
           "Stage 3: Training Hard Part"]
    lossfunc = [lossfuncs.stage_1, lossfuncs.stage_2, lossfuncs.stage_3]

    for stage in range(3):
        pbar = tqdm(total=int(len(traindata)/4), leave=False)
        print(msg[stage])
        optimizer = Adam(model.parameters(), lr=0.01)
        lossdata = []
        for epoch in range(5):
            for x, y in traindata:
                out = model(x)
                optimizer.zero_grad()
                loss = lossfunc[stage](out, y)
                lossdata.append(loss)
                if len(lossdata) == 20:
                    loss.data = torch.mean(torch.tensor(lossdata))
                    loss.backward()
                    optimizer.step()
                    pbar.set_description(
                        'stage {} epoch {} iter {} | train loss: %.5f, len: {}'.format(stage+1, epoch+1, pbar.n, x.shape) % (loss))
                    pbar.update()

                    if pbar.n % 1000 == 0:
                        modelpath = os.path.join(
                            'model/ripplemodel_stage{}_epoch{}_iter{}'.format(stage+1, epoch+1, pbar.n))
                        torch.save({'stage': stage,
                                    'model_state_dict': model.state_dict(),
                                    'iter': pbar.n},
                                   modelpath)
                    lossdata = []

        pbar.close()
        modelpath = os.path.join('model/ripplemodel_stage{}'.format(stage+1))
        torch.save({'stage': stage+1,
                    'model_state_dict': model.state_dict(),
                    'iter': 0},
                   modelpath)

if __name__=='__main__':
    main()