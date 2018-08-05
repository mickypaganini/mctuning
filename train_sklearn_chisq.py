#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File: train_sklearn.py
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
from utils import configure_logging, safe_mkdir
import plotting
from likelihood import Likelihood2D
from chisq import pseudo_chi_squared

TIME = str(int(time()))
safe_mkdir('history')

# logging
configure_logging()
logger = logging.getLogger("Main")


def reset_logger():
    global logger
    logger = logging.getLogger("Main")


def run_single_experiment(args, varID_0, varID_1, dataloader_0, dataloader_1,
                          dataloader_val_0, dataloader_val_1, dataloader_test_0,
                          dataloader_test_1, experiment_id=None):

    if experiment_id is not None:
        global logger
        logger = logger.getChild('experiment_id={}'.format(experiment_id))

    # initialize model
    bins = np.linspace(0, 60, 61).tolist() + \
        np.linspace(62, 100, 11).tolist() + [130, 10000]
    model = Likelihood2D((10, 10))

    def get_data(dataloader_0, dataloader_1):
        weights_sig = dataloader_1.dataset.to_dict()['weights'][args.class1]
        weights_bkg = dataloader_0.dataset.to_dict()['weights'][args.class0]

        # val dataloaders unused
        ntrk_sig = dataloader_1.dataset.to_dict()['nparticles']
        ntrk_bkg = dataloader_0.dataset.to_dict()['nparticles']

        X_train = np.concatenate((ntrk_sig, ntrk_bkg), axis=0)
        weights_train = np.concatenate((weights_sig.ravel(), weights_bkg.ravel()),
                                       axis=0)
        y_train = np.array([1] * ntrk_sig.shape[0] + [0] * ntrk_bkg.shape[0])
        return X_train, y_train, weights_train

    model.fit(*get_data(dataloader_0, dataloader_1))

    # test
    logger.debug('Testing')

    X_test, y_true, weights_test = get_data(dataloader_test_0,
                                            dataloader_test_1)
    y_score = model.predict(X_test)

    pred_baseline = y_score[y_true == 0]
    pred_variation = y_score[y_true == 1]
    weights_baseline = weights_test[y_true == 0]
    weights_variation = weights_test[y_true == 1]

    plotting.plot_output(
        np.array(pred_baseline).ravel(),
        np.array(pred_variation).ravel(),
        varID_0,
        varID_1,
        np.array(weights_baseline).ravel(),
        np.array(weights_variation).ravel(),
        'likelihood',
    )

    roc_auc_est = roc_auc_score(
        y_true,
        y_score,
        sample_weight=weights_test
    )

    chi_sq_test = pseudo_chi_squared(X=X_test, y=y_true, weights=weights_test,
                                     bins=[bins, bins])

    logger.debug('ROC {} = {}'.format(varID_1, roc_auc_est))
    logger.debug('Chi^2({}) = {}'.format(varID_1, chi_sq_test))

    reset_logger()

    return {
        'variation': varID_1,
        'ROC_AUC': roc_auc_est,
        'experiment_id': experiment_id,
        'chi_sq': chi_sq_test
    }

if __name__ == '__main__':

    import argparse
    import cPickle as pickle

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('config0', type=str,
                        help='Path to yaml config file for class 0')
    parser.add_argument(
        'class0', type=str, help='Name of variation for class 0. If unweighted, write Baseline')
    parser.add_argument('config1', type=str,
                        help='Path to yaml config file for class 1')
    parser.add_argument(
        'class1', type=str, help='Name of variation for class 1. If unweighted, write Baseline')
    parser.add_argument('--maxlen', type=int, default=100)
    parser.add_argument('--ntrain', type=int, default=100000)
    parser.add_argument('--nval', type=int, default=100000)
    parser.add_argument('--ntest', type=int, default=100000)
    parser.add_argument('--nfolds', type=int, default=10)
    parser.add_argument('--dataloader-workers', '-j', type=int,
                        default=6, help='Number of DataLoader workers')
    parser.add_argument('--min-lead-pt', type=int, default=500)
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--nepochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--monitor', type=str, default='roc_auc')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--model', type=str, default='rnn',
                        help='One of rnn, ntrack, 1dconv')
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    # parser.add_argument('--test-iter', type=int, default=2)
    parser.add_argument('--pretrain', action='store_true',
                        help='Load pretrained weights')

    args = parser.parse_args()

    # check arguments
    if args.model not in ['rnn', 'ntrack', '1dconv']:
        raise ValueError(
            '--model can only be one of ["rnn", "ntrack", "1dconv"]')

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
        logger.warning(
            'Requested classification of datasets with identical parameters!')

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

    logger.info(
        'creating experiment setup for {}-fold cross-validation'.format(args.nfolds))

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

   # for i, ((d_0, d_val_0, d_test_0), (d_1, d_val_1, d_test_1)) in enumerate(izip(dataloaders_0, dataloaders_1))
    #]

    logger.debug('{} experiments finished. Results:'.format(
        len(experiment_runs)))

    logger.debug('DICT:{}'.format(experiment_runs))
    logger.debug('Formatted results:')

    #for run in experiment_runs:
        # logger.info('Experiment {experiment_id}: ROC({variation}) = {ROC_AUC}+-{ROC_AUC_Err}'.format(**run))
        #logger.info(
        #    'Experiment {experiment_id}: ROC({variation}) = {ROC_AUC}, Chi^2({variation}) = {chi_sq}'
        #    .format(**run)
        #)
    logger.info('{} : {}'.format(experiment_runs[0]['variation'], [run['ROC_AUC'] for run in experiment_runs] ))    
    logger.info('{} : {}'.format(experiment_runs[0]['variation'], [run['chi_sq'] for run in experiment_runs] ))

