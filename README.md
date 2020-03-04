# CoinForecast
Predict cryptocurrency price for next a hour, with GRU Network based on RNN Network.

### Requirements

- python3
- pytorch
- matplotlib

### Data

get data from upbit [api](https://crix-api.upbit.com/v1/crix/candles/minutes/60?code=CRIX.UPBIT.KRW-XRP&count=2000&ciqrandom=1509540252193)

### Instructions

**Training on new data**
`python trainmodel.py --log "yourcoinlog" --resume_train "pretrainedmodelpath"`
To see predict value difference while training, just add `--vis True` to the line above.

**Running**
`python testmodel.py --log "yourcoinlog" --model "yourmodelpath" `
To see predict value difference, just add `--vis True` to the line above.

**Working example with [pretrained model](https://github.com/PlanNoa/CoinForecast/blob/master/pretrained model/ripplemodel)**
(Blue:Real values,Orange:Predicted)

![ripple predict](https://github.com/PlanNoa/CoinForecast/blob/master/pretrained model/ripplemodel.png)

