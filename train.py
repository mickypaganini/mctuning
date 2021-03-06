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
from time import time
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from itertools import izip
from joblib import Parallel, delayed

from dataprep import load_data, make_cv_dataloaders, DijetDataset
from models import DoubleLSTM, NTrackModel, BeefyConv1DModel
from utils import configure_logging, safe_mkdir
import plotting

TIME = str(int(time()))
safe_mkdir('history')

# logging
configure_logging()
logger = logging.getLogger("Main")

def reset_logger():
    global logger
    logger = logging.getLogger("Main")

customFloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
customLongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


def compute_loss(predictions, data, batch_size, classname, classn):
    '''
    Computes the loss on a batch of examples given the model predictions
    '''
    if classn == 0:
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


def train_on_batch(model, optimizer, epoch, batch_size, data, classname, classn):
    '''
    Notes:
        Calls predict() to get predictions
        Calls compute_loss() to get the loss
        Takes care of backprop and weight update
    Returns:
        batch weighted loss values
    '''
    # for n, classname in enumerate([class0, class1]):
    #optimizer.zero_grad()
    # get predictions
    predictions = predict(model, data, batch_size, classname, volatile=False)
    # get loss 
    loss = compute_loss(predictions, data, batch_size, classname, classn)
    # update weights
    loss.backward()
    # torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
    #optimizer.step()
    return (batch_size * loss).data[0] # batch weighted loss
    # return tuple(returns) # batch weighted losses for baseline and variation


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
        ntrk_inputs = Variable(data['nparticles'].type(customFloatTensor), volatile=volatile)
        leading_input = Variable(data['leading_jet'].type(customFloatTensor), volatile=volatile)
        subleading_input = Variable(data['subleading_jet'].type(customFloatTensor), volatile=volatile)
        # unsorted_lengths = Variable(data['unsorted_lengths'].type(customLongTensor), volatile=volatile)
        unsorted_lengths = data['unsorted_lengths'].type(customLongTensor)
        predictions = model(ntrk_inputs, leading_input, subleading_input, unsorted_lengths, batch_weights, batch_size)#)#.data.numpy()
    
    elif isinstance(model, BeefyConv1DModel):
        ntrk_inputs = Variable(data['nparticles'].type(customFloatTensor), volatile=volatile)
        leading_input = Variable(data['leading_jet'].type(customFloatTensor), volatile=volatile)
        subleading_input = Variable(data['subleading_jet'].type(customFloatTensor), volatile=volatile)
        predictions = model(ntrk_inputs, leading_input, subleading_input, batch_weights)

    else:
        raise TypeError

    return predictions


def test(model, dataloader_0, dataloader_1, class0, class1):
    model.eval()
    
    pred_baseline, pred_variation, weights_baseline, weights_variation = [], [], [], []

    for batch_idx, (data_0, data_1) in enumerate(izip(dataloader_0, dataloader_1)): # loop thru batches
        if batch_idx % 10:
            logger.debug('Batch {}'.format(batch_idx))
        batch_size_0 = len(data_0['weights_' + class0])
        batch_size_1 = len(data_1['weights_' + class1])

        _pred_baseline = predict(model, data_0, batch_size_0, class0, volatile=True)
        _pred_variation = predict(model, data_1, batch_size_1, class1, volatile=True)
        if torch.cuda.is_available():
            pred_baseline.extend(F.sigmoid(_pred_baseline).data.cpu().numpy())
            pred_variation.extend(F.sigmoid(_pred_variation).data.cpu().numpy())
        else:
            pred_baseline.extend(F.sigmoid(_pred_baseline).data.numpy())
            pred_variation.extend(F.sigmoid(_pred_variation).data.numpy())
        weights_baseline.extend(data_0['weights_' + class0])
        weights_variation.extend(data_1['weights_' + class1])

    return pred_baseline, pred_variation, weights_baseline, weights_variation


def multitest(model, dataloader_0, dataloader_1, class0, class1, times=2):
    '''
    Custom test function that calls predict() on each batch of
    `times` different datasets.
    '''
    model.eval()
    
    # will be lists of lists [times x n_events]
    pred_baseline, pred_variation, weights_baseline, weights_variation = [], [], [], []

    for t in tqdm(range(times)): # number of independent evaluations
        logger.debug('Test iteration number {}'.format(t))

        pred_baseline_t, pred_variation_t, weights_baseline_t, weights_variation_t = [], [], [], []
        features_baseline, features_variation = [], []

        for batch_idx, (data_0, data_1) in enumerate(izip(dataloader_0, dataloader_1)): # loop thru batches
            if batch_idx % 10:
                logger.debug('Batch {}'.format(batch_idx))
            batch_size_0 = len(data_0['weights_' + class0])
            batch_size_1 = len(data_1['weights_' + class1])

            _pred_baseline = predict(model, data_0, batch_size_0, class0, volatile=True)
            # accumulate hidden batch features for plotting
            if torch.cuda.is_available():
                features_baseline.append(model.batch_features.data.cpu().numpy())
            else:
                features_baseline.append(model.batch_features.data.numpy())

            _pred_variation = predict(model, data_1, batch_size_1, class1, volatile=True)
            # accumulate hidden batch features for plotting
            if torch.cuda.is_available():
                features_variation.append(model.batch_features.data.cpu().numpy())
            else:
                features_variation.append(model.batch_features.data.numpy())

            if torch.cuda.is_available():
                pred_baseline_t.extend(F.sigmoid(_pred_baseline).data.cpu().numpy())
                pred_variation_t.extend(F.sigmoid(_pred_variation).data.cpu().numpy())
            else:
                pred_baseline_t.extend(F.sigmoid(_pred_baseline).data.numpy())
                pred_variation_t.extend(F.sigmoid(_pred_variation).data.numpy())
            weights_baseline_t.extend(data_0['weights_' + class0])
            weights_variation_t.extend(data_1['weights_' + class1])

        pred_baseline.append(pred_baseline_t)
        pred_variation.append(pred_variation_t)
        weights_baseline.append(weights_baseline_t)
        weights_variation.append(weights_variation_t)

    return pred_baseline, pred_variation, weights_baseline, weights_variation, features_baseline, features_variation


def train(model,
        optimizer,
        class0, class1,
        varID_0, varID_1,
        train_data_0, train_data_1,
        validation_data_0, validation_data_1,
        checkpoint_path,
        epochs=100,
        patience=5,
        monitor='roc_auc', override_logger=None):
    '''
    train_data = dataloader
    validation_data = dataloader_val
    '''

    if override_logger:
        logger = override_logger

    # TODO: this is currently wrong but will be right once I change the Dataloade/dataset creation
    n_validation = min(len(validation_data_0.dataset), len(validation_data_1.dataset)) 
    n_training = min(len(train_data_0.dataset), len(train_data_1.dataset)) 
    
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
            
            # zipping cuts length
            
            it = izip(train_data_0, train_data_1)
            for batch_idx, (data_0, data_1) in enumerate(it):
                this_batch_size_0 = len(data_0['weights_' + class0])
                this_batch_size_1 = len(data_1['weights_' + class1])

                batch_weighted_loss_baseline_i = train_on_batch(
                    model, optimizer, epoch, this_batch_size_0, data_0, class0, classn=0)
                batch_weighted_loss_variation_i = train_on_batch(
                    model, optimizer, epoch, this_batch_size_1, data_1, class1, classn=1)

                if (batch_idx % 10 == 0) and (batch_idx > 0):
                # if batch_idx % 64 == 0:
                    # print 'grad:'
                    # print model.dense.weight.grad
                    # np.save('densegrad1.npy', model.dense.weight.grad.cpu().data.numpy())
                    # np.save('densegrad2.npy', model.dense2.weight.grad.cpu().data.numpy())
                    # print '---'
                    # print model.dense.weight.min(), model.dense.weight.mean(), model.dense.weight.max()
                    optimizer.step()
                    optimizer.zero_grad()

                if batch_idx % 1 == 0:
                    logger.debug('Epoch: {} [{}/{} ({}%)]; Loss: {} = {:0.3f}, {} = {:0.3f}, Total = {:0.5f}'.format(
                        epoch,
                        train_data_0.batch_size * batch_idx + this_batch_size_0, # approximate, only true for class0
                        n_training,
                        100 * (train_data_0.batch_size * batch_idx + this_batch_size_0) / n_training, # approximate, only true for class0
                        varID_0.replace('_', ' '),
                        batch_weighted_loss_baseline_i / this_batch_size_0,
                        varID_1.replace('_', ' '),
                        batch_weighted_loss_variation_i / this_batch_size_1,
                        (batch_weighted_loss_baseline_i + batch_weighted_loss_variation_i) / (this_batch_size_0 + this_batch_size_1)
                    ))
                
                # accumulate per-epoch loss
                batch_weighted_loss_baseline_epoch += batch_weighted_loss_baseline_i
                batch_weighted_loss_variation_epoch += batch_weighted_loss_variation_i
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            logger.info('Epoch {}: Loss {} ~ {:0.5f}; Loss {} ~ {:0.5f}; Total ~ {:0.5f}'.format(
                epoch,
                varID_0.replace('_', ' '),
                batch_weighted_loss_baseline_epoch / n_training, #approx
                varID_1.replace('_', ' '),
                batch_weighted_loss_variation_epoch / n_training, #approx
                (batch_weighted_loss_baseline_epoch + batch_weighted_loss_variation_epoch) / (2. * n_training) #approx
            ))
            
            # validate
            model.eval()
            loss_val = 0
            scores_baseline, scores_variation, weights_baseline, weights_variation = [], [], [], []

            for batch_idx_val, (data_val_0, data_val_1) in enumerate(izip(validation_data_0, validation_data_1)):
                batch_size_0 = len(data_val_0['weights_' + class0])
                batch_size_1 = len(data_val_1['weights_' + class1])
                # get predictions
                pred_baseline = predict(model, data_val_0, batch_size_0, class0, volatile=True)
                pred_variation = predict(model, data_val_1, batch_size_1, class1, volatile=True)
                scores_baseline.extend(F.sigmoid(pred_baseline).data.cpu().numpy())
                scores_variation.extend(F.sigmoid(pred_variation).data.cpu().numpy())
                weights_baseline.extend(data_val_0['weights_' + class0])
                weights_variation.extend(data_val_1['weights_' + class1])
                # get loss 
                loss_baseline = compute_loss(pred_baseline, data_val_0, batch_size_0, class0, classn=0)
                loss_variation = compute_loss(pred_variation, data_val_1, batch_size_1, class1, classn=1)
                # important to get data, not Var
                loss_val += ((loss_baseline * batch_size_0 + loss_variation * batch_size_1) / (batch_size_0 + batch_size_1)).data[0]

            loss_val = float(loss_val) / (batch_idx_val + 1.) # / n_validation)
            logger.info('Epoch {}: Validation Loss = {:0.5f}'.format(epoch, loss_val))

            # log history
            val_file = os.path.join('history', checkpoint_path.split('/')[-1].split('.')[0] + '_val.txt')
            train_file = os.path.join('history', checkpoint_path.split('/')[-1].split('.')[0] + '_train.txt')
            roc_file = os.path.join('history', checkpoint_path.split('/')[-1].split('.')[0] + '_ROCval.txt')

            y_true = np.concatenate((np.zeros(len(scores_baseline)), np.ones(len(scores_variation))))
            y_score = np.concatenate((scores_baseline, scores_variation))
            roc_score = roc_auc_score(
                y_true,
                y_score,
                sample_weight=np.concatenate( (weights_baseline, weights_variation) )
            )

            def _write_history(loss, filepath):
                if os.path.isfile(filepath):
                    f = open(filepath, "a+")
                    f.write(str(loss) + ', ')
                    f.close()
                else:
                    with open(filepath, "w") as f:
                        f.write(str(loss) + ', ')

            _write_history(loss_val, val_file)
            _write_history(
                (batch_weighted_loss_baseline_epoch + batch_weighted_loss_variation_epoch) / (2. * n_training),
                train_file
            )
            _write_history(roc_score, roc_file)
            
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


def run_single_experiment(args, varID_0, varID_1, dataloader_0, dataloader_1, 
                          dataloader_val_0, dataloader_val_1, dataloader_test_0,
                          dataloader_test_1, experiment_id=None):

    if experiment_id is not None:
        global logger
        logger = logger.getChild('experiment_id={}'.format(experiment_id))
    # initialize model
    if args.model == 'rnn': 
        model = DoubleLSTM(ntrk_input_size=2,
                           input_size=2, #10
                           output_size=10,
                           num_layers=1,
                           dropout=0.3,
                           bidirectional=True,
                           batch_size=args.batchsize,
                           tagger_output_size=1
        )
    elif args.model == 'ntrack':
        model = NTrackModel(input_size=2)
    else:
        model = BeefyConv1DModel(ntrk_input_size=2,
                                 input_size=10,
                                 hidden_size=4,
                                 rnn_output_size=4,
                                 kernel_size=2,
                                 dropout=0.2,
                                 bidirectional=True
        )
    if torch.cuda.is_available():
        model.cuda() # move model to GPU
        logger.info('Running on GPU')
        
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.0)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)#, momentum=0.95, nesterov=True)
    logger.debug('Model: {}'.format(model))

    safe_mkdir('checkpoints')
    checkpoint_path = os.path.join(
        'checkpoints',
        args.checkpoint + '_exp' + str(experiment_id) + '_' + args.model + '_' + varID_0 + '_' + varID_1 + '.pth'
    )
    if args.pretrain and os.path.exists(checkpoint_path):
        logger.info('Restoring best weights from checkpoint at ' + checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        logger.info('Training from scratch')

    # train
    train(model,
        optimizer,
        args.class0, args.class1,
        varID_0, varID_1,
        dataloader_0, dataloader_1, 
        dataloader_val_0, dataloader_val_1,
        checkpoint_path=checkpoint_path,
        epochs=args.nepochs,
        patience=args.patience,
        monitor=args.monitor, 
        override_logger=logger
    )

    # test
    logger.info('Testing')
    
    # bootstrap the AUC and its uncertainty due to the random batches
    rocs = []
    for i_bs in range(10): 
        logger.debug('Test iteration {}'.format(i_bs))
        pred_baseline, pred_variation, weights_baseline, weights_variation = test(model,
            dataloader_test_0, dataloader_test_1, args.class0, args.class1
        )

        # check performance
        y_true = np.concatenate((np.zeros(len(pred_baseline)), np.ones(len(pred_variation))))
        y_score = np.concatenate((pred_baseline, pred_variation))

        # only plot once
        if i_bs == 0:
            plotting.plot_output(
                np.array(pred_baseline).ravel(),
                np.array(pred_variation).ravel(),
                varID_0,
                varID_1,
                np.array(weights_baseline).ravel(),
                np.array(weights_variation).ravel(),
                args.model,
            )
        rocs.append(roc_auc_score(
            y_true,
            y_score,
            sample_weight=np.concatenate( (weights_baseline, weights_variation) )
        ))

    roc_auc_est = np.mean(rocs)
    roc_auc_err = np.std(rocs)/np.sqrt(len(rocs)) # systematic uncertainty due to batching
    # get bootstrap results
    logger.info('ROC {} = {} +- {}'.format(
        varID_1,
        roc_auc_est,
        roc_auc_err
    ))

    reset_logger()
    
    return {
        'variation': varID_1, 
        'ROC_AUC': roc_auc_est, 
        'ROC_AUC_Err': roc_auc_err, 
        'experiment_id': experiment_id
    }

if __name__ == '__main__':

    import argparse
    import cPickle as pickle

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('config0', type=str, help='Path to yaml config file for class 0')
    parser.add_argument('class0', type=str, help='Name of variation for class 0. If unweighted, write Baseline')
    parser.add_argument('config1', type=str, help='Path to yaml config file for class 1')
    parser.add_argument('class1', type=str, help='Name of variation for class 1. If unweighted, write Baseline')
    parser.add_argument('--maxlen', type=int, default=100)
    parser.add_argument('--ntrain', type=int, default=100000)
    parser.add_argument('--nval', type=int, default=100000)
    parser.add_argument('--ntest', type=int, default=100000)
    parser.add_argument('--nfolds', type=int, default=10)
    parser.add_argument('--dataloader-workers', '-j', type=int, default=6, help='Number of DataLoader workers')
    parser.add_argument('--min-lead-pt', type=int, default=500)
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--nepochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--monitor', type=str, default='roc_auc')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--model', type=str, default='rnn', help='One of rnn, ntrack, 1dconv')
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    # parser.add_argument('--test-iter', type=int, default=2)
    parser.add_argument('--pretrain', action='store_true', help='Load pretrained weights')

    args = parser.parse_args()

    # check arguments
    if args.model not in ['rnn', 'ntrack', '1dconv']:
        raise ValueError('--model can only be one of ["rnn", "ntrack", "1dconv"]')

    # load or make data
    logger.debug('Loading data')
    dataloader_0, dataloader_val_0, dataloader_test_0, varID_0 = load_data(
        args.config0, args.class0, args.ntrain, args.nval, args.ntest, args.maxlen, args.min_lead_pt, args.batchsize
    )
    # variation could be None here
    dataloader_1, dataloader_val_1, dataloader_test_1, varID_1 = load_data(
        args.config1, args.class1, args.ntrain, args.nval, args.ntest, args.maxlen, args.min_lead_pt, args.batchsize
    )

    if varID_0 == varID_1:
        logger.warning('Requested classification of datasets with identical parameters!')

    '''
    logger.debug('Plotting distributions')
    plotting.plot_weights(
        dataloader_0.dataset, dataloader_val_0.dataset, dataloader_test_0.dataset,
        dataloader_1.dataset, dataloader_val_1.dataset, dataloader_test_1.dataset,
        varID_0, varID_1, args.class0, args.class1)
    plotting.plot_jetn_trkn(
        dataloader_0.dataset, dataloader_val_0.dataset, dataloader_test_0.dataset,
        dataloader_1.dataset, dataloader_val_1.dataset, dataloader_test_1.dataset,
        varID_0, varID_1, args.class0, args.class1, jetn=0, trkn=0)
    plotting.plot_jetn_trkn(
        dataloader_0.dataset, dataloader_val_0.dataset, dataloader_test_0.dataset,
        dataloader_1.dataset, dataloader_val_1.dataset, dataloader_test_1.dataset,
        varID_0, varID_1, args.class0, args.class1, jetn=1, trkn=0)
    plotting.plot_ntrack(
        dataloader_0.dataset, dataloader_val_0.dataset, dataloader_test_0.dataset,
        dataloader_1.dataset, dataloader_val_1.dataset, dataloader_test_1.dataset,
        varID_0, varID_1, args.class0, args.class1)
    '''

    # We don't change the dataloading / regeneration semantics, we'll just
    # apply some clever crossvaliation to each variation . We'll throw away the
    # dataloaders built here and reconstruct them later.
    from torch.utils.data import ConcatDataset
    from itertools import izip
    TRAIN_PCT = 0.66

    datasets_0 = [dataloader_0.dataset, dataloader_val_0.dataset, 
                  dataloader_test_0.dataset]
    datasets_1 = [dataloader_1.dataset, dataloader_val_1.dataset, 
                  dataloader_test_1.dataset]

    #combined_dataset_0 = torch.utils.data.ConcatDataset(datasets_0)
    #combined_dataset_1 = torch.utils.data.ConcatDataset(datasets_1)
    combined_dataset_0 = DijetDataset.concat(datasets_0)
    combined_dataset_1 = DijetDataset.concat(datasets_1)
    
    pin_memory = True if torch.cuda.is_available() else False

    logger.info('creating experiment setup for {}-fold cross-validation'.format(args.nfolds))
    
    dataloaders_0 = make_cv_dataloaders(
        dataset=combined_dataset_0, 
        batchsize=args.batchsize, 
        train_size=TRAIN_PCT, 
        nfolds=args.nfolds, 
        pin_memory=pin_memory, 
        num_workers=args.dataloader_workers
    )

    dataloaders_1 = make_cv_dataloaders(
        dataset=combined_dataset_1, 
        batchsize=args.batchsize, 
        train_size=TRAIN_PCT, 
        nfolds=args.nfolds, 
        pin_memory=pin_memory, 
        num_workers=args.dataloader_workers
    )

    
    experiment_runs = []
    for i, ((d_0, d_val_0, d_test_0), (d_1, d_val_1, d_test_1)) in enumerate(izip(dataloaders_0, dataloaders_1)):
        plotting.plot_weights(
            d_0.dataset, d_val_0.dataset, d_test_0.dataset,
            d_1.dataset, d_val_1.dataset, d_test_1.dataset,
            varID_0, varID_1, args.class0, args.class1, exp=i)
        plotting.plot_jetn_trkn(
            d_0.dataset, d_val_0.dataset, d_test_0.dataset,
            d_1.dataset, d_val_1.dataset, d_test_1.dataset,
            varID_0, varID_1, args.class0, args.class1, jetn=0, trkn=0, exp=i)
        plotting.plot_jetn_trkn(
            d_0.dataset, d_val_0.dataset, d_test_0.dataset,
            d_1.dataset, d_val_1.dataset, d_test_1.dataset,
            varID_0, varID_1, args.class0, args.class1, jetn=1, trkn=0, exp=i)
        plotting.plot_ntrack(
            d_0.dataset, d_val_0.dataset, d_test_0.dataset,
            d_1.dataset, d_val_1.dataset, d_test_1.dataset,
            varID_0, varID_1, args.class0, args.class1, exp=i)
 
        experiment_runs.append(
            run_single_experiment(
                args, varID_0, varID_1, 
                dataloader_0=d_0, 
                dataloader_1=d_1, 
                dataloader_val_0=d_val_0, 
                dataloader_val_1=d_val_1, 
                dataloader_test_0=d_test_0,
                dataloader_test_1=d_test_1, 
                experiment_id=i
            )
        )
     
   #for i, ((d_0, d_val_0, d_test_0), (d_1, d_val_1, d_test_1)) in enumerate(izip(dataloaders_0, dataloaders_1))
    #]

    logger.info('{} experiments finished. Results:'.format(len(experiment_runs)))

    logger.info('DICT:{}'.format(experiment_runs))
    logger.info('Formatted results:')

    for run in experiment_runs:
        logger.info('Experiment {experiment_id}: ROC({variation}) = {ROC_AUC}+-{ROC_AUC_Err}'.format(**run))

    
    
