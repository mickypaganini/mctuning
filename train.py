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
import os
from sklearn.metrics import roc_auc_score

from dataprep import load_data
from models import DoubleLSTM, NTrackModel
from utils import configure_logging
import plotting

# logging
configure_logging()
logger = logging.getLogger("Main")

customFloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def compute_loss(predictions, data, batch_size, classname):
    '''
    Computes the loss on a batch of examples given the model predictions
    '''
    if classname == 'Baseline':
        if torch.cuda.is_available():
            targets = Variable(torch.zeros((batch_size, 1)), requires_grad=False).cuda()
        else:
            targets = Variable(torch.zeros((batch_size, 1)), requires_grad=False)
    
    else: # variation
        if torch.cuda.is_available():
            targets = Variable(torch.ones((batch_size, 1)), requires_grad=False).cuda()
        else:
            targets = Variable(torch.ones((batch_size, 1)), requires_grad=False)

    loss = F.binary_cross_entropy_with_logits(predictions, targets,
            weight=Variable(data['weights_' + classname].type(customFloatTensor), requires_grad=False))

    return loss


def train_on_batch(model, optimizer, epoch, batch_idx, batch_size, data, variation):
    '''
    Notes:
        Calls predict() to get predictions
        Calls compute_loss() to get the losses
        Takes care of backprop and weight update
    Returns:
        batch weighted values of baseline and variation losses
    '''
    returns = []
    for classname in ['Baseline', variation]:
        optimizer.zero_grad()
        # get predictions
        predictions = predict(model, data, batch_size, classname, volatile=False)
        # get loss 
        loss = compute_loss(predictions, data, batch_size, classname)
        # update weights
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()
        returns.append((batch_size * loss).data[0])

    return tuple(returns) # batch weighted losses for baseline and variation


def predict(model, data, batch_size, classname, volatile):
    '''
    Wraps variable creation for inputs and weights + call of forward() for the model
    on a batch of examples
    Raises:
    -------
        TypeError: if the model type is not one of [NTrackModel, DoubleLSTM]
    Returns:
    --------
        model predictions
    '''

    weights = data['weights_' + classname].type(customFloatTensor)

    batch_weights = Variable(
            weights.resize_(1, batch_size),
            requires_grad=False, volatile=volatile) / torch.sum(weights) # normalized

    if isinstance(model, NTrackModel):
        inputs = Variable(data['nparticles'].type(customFloatTensor), volatile=volatile) # B x 2
        predictions = model(inputs, batch_weights, batch_size)#)#.data.numpy()

    elif isinstance(model, DoubleLSTM):
        leading_input = Variable(data['leading_jet'].type(customFloatTensor), volatile=volatile)
        subleading_input = Variable(data['subleading_jet'].type(customFloatTensor), volatile=volatile)
        unsorted_lengths = data['unsorted_lengths'].numpy()
        predictions = model(leading_input, subleading_input, unsorted_lengths, batch_weights, batch_size)#)#.data.numpy()

    else:
        raise TypeError

    return predictions


def multitest(model, dataloader, variation, times=2):
    '''
    Custom test function that calls predict() on each batch of
    `times` different datasets.
    '''
    model.eval()
    
    # will be lists of lists [times x n_events]
    pred_baseline, pred_variation, sample_weight = [], [], []

    for t in range(times): # number of independent evaluations
        logger.debug('Test iteration number {}'.format(t))

        pred_baseline_t, pred_variation_t, sample_weight_t = [], [], []

        for batch_idx, data in enumerate(dataloader): # loop thru batches
            if batch_idx % 10:
                logger.debug('Batch {}'.format(batch_idx))
            batch_size = len(data['weights_Baseline'])
            _pred_baseline = predict(model, data, batch_size, 'Baseline', volatile=True)
            _pred_variation = predict(model, data, batch_size, variation, volatile=True)
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


def train(model, optimizer, variation, train_data, validation_data, checkpoint_path, epochs=100, patience=5, monitor='roc_auc'):
    '''
    train_data = dataloader
    validation_data = dataloader_val
    '''

    n_validation = validation_data.dataset.nevents
    n_training = train_data.dataset.nevents
    best_loss = np.inf
    best_roc = 0.
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
                
                batch_weighted_loss_baseline_i, batch_weighted_loss_variation_i = train_on_batch(
                    model, optimizer, epoch, batch_idx, this_batch_size, data, variation
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
            scores_baseline, scores_variation, sample_weight = [], [], []

            for batch_idx_val, data_val in enumerate(validation_data):
                batch_size = len(data_val['weights_Baseline'])
                # get predictions
                pred_baseline = predict(model, data_val, batch_size, 'Baseline', volatile=True)
                pred_variation = predict(model, data_val, batch_size, variation, volatile=True)
                scores_baseline.extend(F.sigmoid(pred_baseline).data.cpu().numpy())
                scores_variation.extend(F.sigmoid(pred_variation).data.cpu().numpy())
                sample_weight.extend(data_val['weights_' + variation])
                # get loss 
                loss_baseline = compute_loss(pred_baseline, data_val, batch_size, 'Baseline')
                loss_variation = compute_loss(pred_variation, data_val, batch_size, variation)
                loss_val += ((loss_baseline + loss_variation) * batch_size / 2.).data[0] # important to get data, not Var

            loss_val = float(loss_val / n_validation)
            logger.info('Epoch {}: Validation Loss = {:0.5f}'.format(epoch, loss_val))

            # early stopping
            if monitor == 'val_loss':
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
            elif monitor == 'roc_auc':
                y_true = np.concatenate((np.zeros(n_validation), np.ones(n_validation)))
                y_score = np.concatenate((scores_baseline, scores_variation))
                roc_score = roc_auc_score(
                    y_true,
                    y_score,
                    sample_weight=np.concatenate( (np.ones(n_validation), sample_weight) )
                )
                if roc_score > best_roc:
                    logger.info('Validation ROC AUC improved from {:0.5f} to {:0.5f}'.format(best_roc, roc_score))
                    best_roc = roc_score
                    wait = 0
                    logger.info('Saving checkpoint at ' + checkpoint_path)
                    torch.save(model.state_dict(), checkpoint_path)
                else:
                    logger.info('Validation ROC AUC did not improve (Best: {:0.5f}; current: {:0.5f})'.format(best_roc, roc_score))
                    wait += 1
                    if wait >= patience - 1:
                        logger.info('Stopping early.')
                        break

            else:
                raise ValueError
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
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--nepochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--monitor', type=str, default='roc_auc')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--model', type=str, default='rnn', help='Either rnn or ntrack')
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    parser.add_argument('--test-iter', type=int, default=2)
    parser.add_argument('--pretrain', action='store_true', help='Load pretrained weights')

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
                        dropout=0.0,
                        bidirectional=False,
                        batch_size=args.batchsize,
                        tagger_output_size=1
        )
    else:
        model = NTrackModel(input_size=2)

    if torch.cuda.is_available():
        model.cuda() # move model to GPU
        logger.info('Running on GPU')
        
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.0)
    logger.debug('Model: {}'.format(model))

    checkpoint_path = args.checkpoint + '_' + args.model + '_' + args.variation + '.pth'
    if args.pretrain and os.path.exists(checkpoint_path):
        logger.info('Restoring best weights from checkpoint at ' + checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        logger.info('Training from scratch')

    # train
    train(model,
        optimizer,
        args.variation,
        dataloader,
        dataloader_val,
        checkpoint_path=checkpoint_path,
        epochs=args.nepochs,
        patience=args.patience,
        monitor=args.monitor
    )

    # test
    logger.debug('Testing')
    pred_baseline, pred_variation, sample_weight = multitest(model, dataloader_test, args.variation, times=args.test_iter)
    # check performance
    for t in range(args.test_iter):
        plotting.plot_output(
            np.array(pred_baseline[t]).ravel(), np.array(pred_variation[t]).ravel(),
            args.variation, np.array(sample_weight[t]).ravel(), args.model, t)
        y_true = np.concatenate((np.zeros(len(pred_baseline[0])), np.ones(len(pred_baseline[0]))))
        y_score = np.concatenate((pred_baseline[t], pred_variation[t]))
        logger.debug('ROC iteration {}'.format(t))
        logger.info(roc_auc_score(
            y_true,
            y_score,
            # average='weighted',
            sample_weight=np.concatenate( (np.ones(len(pred_baseline[0])), sample_weight[t]) )
        ))
