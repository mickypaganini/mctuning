#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File: train.py
Author: Michela Paganini (michela.paganini@yale.edu)
'''
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import logging
import sys
from sklearn.metrics import roc_auc_score

from dataprep import load_data
from models import DoubleLSTM, NTrackModel
from utils import configure_logging
import plotting

# logging
configure_logging()
logger = logging.getLogger("Main")

customFloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def train_on_batch(model, optimizer, epoch, batch_idx, data, name_weights):
    #weights
    weights = data[name_weights].type(customFloatTensor)
    batch_size = weights.shape[0]
    batch_weights = Variable(
            weights.resize_(1, batch_size),
            requires_grad=False) / torch.sum(weights) # normalized

    # loss 
    # loss_function = nn.BCEWithLogitsLoss(weight=batch_weights) 
    # if torch.cuda.is_available():
    #     loss_function = nn.BCEWithLogitsLoss(weight=batch_weights).cuda()

    if isinstance(model, NTrackModel):
        inputs = Variable(data['nparticles'].type(customFloatTensor), requires_grad=False) # B x 2
        optimizer.zero_grad()
        predictions = model(inputs, batch_weights, batch_size)

    elif isinstance(model, DoubleLSTM):  
        leading_input = Variable(data['leading_jet'].type(customFloatTensor), requires_grad=False)
        subleading_input = Variable(data['subleading_jet'].type(customFloatTensor), requires_grad=False)
        unsorted_lengths = data['unsorted_lengths'].numpy()
        optimizer.zero_grad()
        predictions = model(leading_input, subleading_input, unsorted_lengths, batch_weights, batch_size)
    else:
        raise TypeError

    if name_weights == 'weights_Baseline':
        targets = Variable(torch.zeros((batch_size, 1)), requires_grad=False)
    else:
        targets = Variable(torch.ones((batch_size, 1)), requires_grad=False)

    if torch.cuda.is_available():
        targets = targets.cuda()
        
    loss = F.binary_cross_entropy_with_logits(predictions, targets, weight=Variable(weights, requires_grad=False))
    loss.backward()
    optimizer.step()
    
    batch_weighted_loss = loss * batch_size # to compute per epoch loss

    return batch_weighted_loss.data[0]


def predict(model, data, variation):
    '''
    Raises:
    -------
        TypeError
    '''

    weights_baseline = data['weights_Baseline'].type(customFloatTensor)
    weights_variation = data['weights_' + variation].type(customFloatTensor)

    batch_size = weights_baseline.shape[0]

    batch_weights_baseline = Variable(
            weights_baseline.resize_(1, batch_size),
            requires_grad=False, volatile=True) / torch.sum(weights_baseline) # normalized

    batch_weights_variation = Variable(
            weights_variation.resize_(1, batch_size),
            requires_grad=False, volatile=True) / torch.sum(weights_variation) # normalized

    if isinstance(model, NTrackModel):
        inputs = Variable(data['nparticles'].type(customFloatTensor), requires_grad=False) # B x 2
        pred_baseline = model(inputs, batch_weights_baseline, batch_size)#)#.data.numpy()
        pred_variation = model(inputs, batch_weights_variation, batch_size)#)#.data.numpy()

    elif isinstance(model, DoubleLSTM):
        leading_input = Variable(data['leading_jet'].type(customFloatTensor), volatile=True)
        subleading_input = Variable(data['subleading_jet'].type(customFloatTensor), volatile=True)
        unsorted_lengths = data['unsorted_lengths'].numpy()

        pred_baseline = model(leading_input, subleading_input, unsorted_lengths, batch_weights_baseline, batch_size)#)#.data.numpy()
        pred_variation = model(leading_input, subleading_input, unsorted_lengths, batch_weights_variation, batch_size)#)#.data.numpy()

    else:
        raise TypeError

    return pred_baseline, pred_variation


def validate_on_batch(model, data_val, variation):
    '''
    '''
    batch_size = data_val['weights_Baseline'].shape[0]
    pred_baseline, pred_variation = predict(model, data_val, variation)
    if torch.cuda.is_available():
        targets_baseline = Variable(torch.zeros((batch_size, 1)), requires_grad=False, volatile=True).cuda()
        targets_variation = Variable(torch.ones((batch_size, 1)), requires_grad=False, volatile=True).cuda()
    else:
        targets_baseline = Variable(torch.zeros((batch_size, 1)), requires_grad=False, volatile=True)
        targets_variation = Variable(torch.ones((batch_size, 1)), requires_grad=False, volatile=True)

    # print float(batch_size * (
    #         F.binary_cross_entropy_with_logits(
    #             pred_baseline, 
    #             targets_baseline, 
    #             weight=Variable(data_val['weights_Baseline'].type(customFloatTensor), requires_grad=False)) + 
    #         F.binary_cross_entropy_with_logits(
    #             pred_variation, 
    #             targets_variation, 
    #             weight=Variable(data_val['weights_' + variation].type(customFloatTensor), requires_grad=False))
    #     ) / 2.), float(batch_size * (
    #         F.binary_cross_entropy_with_logits(
    #             pred_baseline, 
    #             targets_baseline) +
    #             F.binary_cross_entropy_with_logits(
    #             pred_variation, 
    #             targets_variation, 
    #             )
    #     ) / 2.)

    loss_increment = batch_size * (
            F.binary_cross_entropy_with_logits(
                pred_baseline, 
                targets_baseline, 
                weight=Variable(data_val['weights_Baseline'].type(customFloatTensor), requires_grad=False)) + 
            F.binary_cross_entropy_with_logits(
                pred_variation, 
                targets_variation, 
                weight=Variable(data_val['weights_' + variation].type(customFloatTensor), requires_grad=False))
        ) / 2.
    return loss_increment.data[0]


def test(model, dataloader, variation, times=2):
    '''
    '''
    model.eval()
    # if torch.cuda.is_available():
    #     loss_function = nn.BCEWithLogitsLoss().cuda()
    # else:
    #     loss_function = nn.BCEWithLogitsLoss()
    
    # will be lists of lists [times x n_events]
    pred_baseline = []
    pred_variation = []
    sample_weight = [] # event weights for the variation

    for t in range(times): # number of independent evaluations
        logger.debug('Test iteration number {}'.format(t))
        # lists
        pred_baseline_t = []
        pred_variation_t = []
        sample_weight_t = []

        for batch_idx, data in enumerate(dataloader): # loop thru batches
            if batch_idx % 10:
                logger.debug('Batch {}'.format(batch_idx))
            _pred_baseline, _pred_variation = predict(model, data, variation)
            if torch.cuda.is_available():
                pred_baseline_t.extend(F.sigmoid(_pred_baseline).data.cpu().numpy())
                pred_variation_t.extend(F.sigmoid(_pred_variation).data.cpu().numpy())
            else:
                pred_baseline_t.extend(F.sigmoid(_pred_baseline).data.numpy())
                pred_variation_t.extend(F.sigmoid(_pred_variation).data.numpy())
            sample_weight_t.extend(data['weights_' + variation])

        pred_baseline.append(pred_baseline_t)
        pred_variation.append(pred_variation_t)
        sample_weight.append(sample_weight_t)

    return pred_baseline, pred_variation, sample_weight


def train(model, optimizer, variation, train_data, validation_data, checkpoint_path, epochs=100, patience=5):
    '''
    train_data = dataloader
    validation_data = dataloader_val
    '''

    n_validation = validation_data.dataset.nevents
    n_training = train_data.dataset.nevents
    best_loss = np.inf
    wait = 0
        
    try:    
        for epoch in range(epochs):
            
            # set model state to training
            model.train()
            
            # initialize per-epoch losses
            batch_weighted_loss_baseline_epoch = 0
            batch_weighted_loss_variation_epoch = 0
            
            for batch_idx, data in enumerate(train_data):
                this_batch_size = len(data['weights_Baseline'])
                
                batch_weighted_loss_baseline_i = train_on_batch(
                    model, optimizer, epoch, batch_idx, data, 'weights_Baseline'
                )
                
                batch_weighted_loss_variation_i = train_on_batch(
                    model, optimizer, epoch, batch_idx, data, 'weights_' + variation
                )

                if batch_idx % 1 == 0:
                    logger.debug('Epoch: {} [{}/{} ({}%)]; Loss: Baseline = {:0.3f}, {} = {:0.3f}, Total = {:0.5f}'.format(
                        epoch,
                        train_data.batch_size * batch_idx + this_batch_size,
                        len(train_data.dataset),
                        100 * (train_data.batch_size * batch_idx + this_batch_size) / len(train_data.dataset),
                        batch_weighted_loss_baseline_i / this_batch_size,
                        variation,
                        batch_weighted_loss_variation_i / this_batch_size,
                        (batch_weighted_loss_baseline_i + batch_weighted_loss_variation_i) / (2 * this_batch_size)
                    ))
                
                # accumulate per-epoch loss
                batch_weighted_loss_baseline_epoch += batch_weighted_loss_baseline_i
                batch_weighted_loss_variation_epoch += batch_weighted_loss_variation_i
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            logger.info('Epoch {}: Loss Baseline = {:0.5f}; Loss {} = {:0.5f}; Total = {:0.5f}'.format(
                epoch,
                batch_weighted_loss_baseline_epoch / n_training,
                variation,
                batch_weighted_loss_variation_epoch / n_training,
                (batch_weighted_loss_baseline_epoch + batch_weighted_loss_variation_epoch) / (2 * n_training)
            ))
            
            # validate
            model.eval()
            loss_val = 0
            # if torch.cuda.is_available():
            #     loss_function = nn.BCEWithLogitsLoss().cuda()
            # else:
            #     loss_function = nn.BCEWithLogitsLoss()
            
            for batch_idx_val, data_val in enumerate(validation_data):
                loss_val += validate_on_batch(model, data_val, variation)

            loss_val /= n_validation
            loss_val = float(loss_val)
            logger.info('Epoch {}: Validation Loss = {:0.5f}'.format(epoch, loss_val))

            # early stopping
            if loss_val < best_loss:
                logger.info('Validation loss improved from {:0.5f} to {:0.5f}'.format(best_loss, loss_val))
                best_loss = loss_val
                wait = 0
                logger.info('Saving checkpoint at ' + checkpoint_path)
                torch.save(model.state_dict(), checkpoint_path)

            else:
                wait += 1
                if wait >= patience - 1:
                    logger.info('Stopping early.')
                    break
    except KeyboardInterrupt:
        logger.info('Training ended early.')
        
    logger.info('Restoring best weights from checkpoint at ' + checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path))


if __name__ == '__main__':

    import argparse
    import cPickle as pickle

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('variation', type=str)
    parser.add_argument('--maxlen', type=int, default=150)
    parser.add_argument('--ntrain', type=int, default=100000)
    parser.add_argument('--nval', type=int, default=100000)
    parser.add_argument('--ntest', type=int, default=100000)
    parser.add_argument('--min-lead-pt', type=int, default=500)
    parser.add_argument('--config', type=str, default='test.yaml')
    parser.add_argument('--batchsize', type=int, default=1024)
    parser.add_argument('--nepochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--model', type=str, default='rnn', help='Either rnn or ntrack')
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    parser.add_argument('--test-iter', type=int, default=2)

    args = parser.parse_args()

    # check arguments
    if args.model not in ['rnn', 'ntrack']:
        raise ValueError('--model can only be one of ["rnn", "ntrack"]')

    # load or make data
    logger.debug('Loading data')
    dataloader, dataloader_val, dataloader_test = load_data(
        args.config,
        args.variation,
        args.ntrain,
        args.nval,
        args.ntest,
        args.maxlen,
        args.min_lead_pt,
        args.batchsize
    )

    # data inspection
    logger.debug('Plotting distributions')
    plotting.plot_jetn_trkn(
        dataloader.dataset, dataloader_val.dataset, dataloader_test.dataset,
        args.variation, jetn=0, trkn=0)
    plotting.plot_jetn_trkn(
        dataloader.dataset, dataloader_val.dataset, dataloader_test.dataset,
        args.variation, jetn=1, trkn=0)
    plotting.plot_ntrack(
        dataloader.dataset, dataloader_val.dataset, dataloader_test.dataset, args.variation)

    # initialize model
    if args.model == 'rnn': 
        model = DoubleLSTM(input_size=9,
                        output_size=32,
                        num_layers=1,
                        dropout=0.1,
                        bidirectional=True,
                        batch_size=args.batchsize,
                        tagger_output_size=1
        )
    else:
        model = NTrackModel(input_size=2)

    if torch.cuda.is_available():
        model.cuda() # move model to GPU
        logger.info('Running on GPU')
        
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    logger.debug('Model: {}'.format(model))

    checkpoint_path = args.checkpoint + '_' + args.model + '_' + args.variation + '.pth'

    # train
    train(model,
        optimizer,
        args.variation,
        dataloader,
        dataloader_val,
        checkpoint_path=checkpoint_path,
        epochs=args.nepochs,
        patience=args.patience
    )

    # test
    logger.debug('Testing')
    pred_baseline, pred_variation, sample_weight = test(model, dataloader_test, args.variation, times=args.test_iter)
    # check performance
    for t in range(args.test_iter):
        y_true = np.concatenate((np.zeros(len(pred_baseline[0])), np.ones(len(pred_baseline[0]))))
        y_score = np.concatenate((pred_baseline[t], pred_variation[t]))
        logger.debug('ROC iteration {}'.format(t))
        logger.info(roc_auc_score(
            y_true,
            y_score,
            average='weighted',
            sample_weight=np.concatenate( (np.ones(len(pred_baseline[0])), sample_weight[t]) )
        ))
