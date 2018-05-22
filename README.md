# MC Tuning with PyTorch

## Data Preparation

[`numpythia`](https://github.com/scikit-hep/numpythia), a pythonic interface to [Pythia](http://home.thep.lu.se/Pythia/), is built from source at commit [4064d5d](https://github.com/scikit-hep/numpythia/commit/4064d5d85ccc95bf5a57b1cca6154c6248a10e70), replacing the included external Pythia version 8.226 with Pythia version 8.230. Pythia arguments are defined in the [`test.yaml`](test.yaml) file and converted to Pythia standard format by the methods provided in [`pythiaconf.py`](pythiaconf.py).

After the various preprocessing steps, the data is loaded into a torch `DataLoader`.

## Model Design
The neural networks used as discriminators are defined in [models.py](models.py). The models are built using [PyTorch](http://pytorch.org/).

The main model, called `DoubleLSTM`, is a two-stream, bidirectional LSTM that processes inputs in the form of sequences of tracks per jet. One stream is dedicated to the leading jet, the other to the subleding jet. All tracks are characterized by 9 kinematic variables ('E', 'Et', 'eta', 'm', 'phi', 'pt', 'px', 'py', 'pz').
Since the network classifies differently weighted batches of identical samples, weights require extra care. 

For benchmarking purposes, the `NTrackModel` only uses as input variables the number of tracks in the leading jet and the number of tracks in the subleading jet.

## Training
The main script, called [train.py](train.py), works as follows:
```
usage: train.py [-h] [--maxlen MAXLEN] [--ntrain NTRAIN] [--nval NVAL]
                [--ntest NTEST] [--min-lead-pt MIN_LEAD_PT]
                [--batchsize BATCHSIZE] [--nepochs NEPOCHS]
                [--patience PATIENCE] [--monitor MONITOR] [--lr LR]
                [--model MODEL] [--checkpoint CHECKPOINT]
                [--test-iter TEST_ITER] [--pretrain]
                config0 class0 config1 class1

positional arguments:
  config0               Path to yaml config file for class 0
  class0                Name of variation for class 0. If unweighted, write
                        Baseline
  config1               Path to yaml config file for class 1
  class1                Name of variation for class 1. If unweighted, write
                        Baseline

optional arguments:
  -h, --help            show this help message and exit
  --maxlen MAXLEN
  --ntrain NTRAIN
  --nval NVAL
  --ntest NTEST
  --min-lead-pt MIN_LEAD_PT
  --batchsize BATCHSIZE
  --nepochs NEPOCHS
  --patience PATIENCE
  --monitor MONITOR
  --lr LR
  --model MODEL         Either rnn or ntrack (default: rnn)
  --checkpoint CHECKPOINT
  --test-iter TEST_ITER
  --pretrain            Load pretrained weights (default: False)
  ```
Example: `CUDA_VISIBLE_DEVICES=1 python train.py test.yaml Baseline fsr13.yaml Baseline --ntrain 100000 --ntest 100000 --nval 100000 --model rnn --patience 100 --nepochs 1000 --test-iter 50 --lr 0.001`

## Task List and Ideas:
- [ ] do not mask zeros
- [ ] train rnn on all vars + dr
- [ ] train rnn on only pT + dr
 
