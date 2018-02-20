import os
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
from utils import safe_mkdir
import torch

matplotlib.rcParams.update({'font.size' : 20})

FEATURES = ['E', 'Et', 'eta', 'm', 'phi', 'pt', 'px', 'py', 'pz']

def plot_weights(d_0, d_val_0, d_test_0, d_1, d_val_1, d_test_1, varID_0, varID_1, class0, class1):
    '''
    '''
    safe_mkdir('plots')
    plt.figure(figsize=(8, 8))
    bins=np.logspace(-7, 6, 30)
    plt.hist(d_0[:]['weights_' + class0],
                histtype='step', bins=bins, color='teal',
                label=r'train - {}'.format(varID_0.replace('_', ' ')))
        
    plt.hist(d_1[:]['weights_' + class1],
            histtype='step', bins=bins, color='orange',
            label=r'train - {}'.format(varID_1.replace('_', ' ')))
    
    plt.hist(d_val_0[:]['weights_' + class0], alpha=0.1,
            histtype='stepfilled', bins=bins, color='teal',
            label=r'val - {}'.format(varID_0.replace('_', ' ')))
    
    plt.hist(d_val_1[:]['weights_' + class1], alpha=0.1,
            histtype='stepfilled', bins=bins, color='orange',
            label=r'val - {}'.format(varID_1.replace('_', ' ')))

    plt.hist(d_test_0[:]['weights_' + class0],
            histtype='step', bins=bins, color='teal', linestyle='dashed',
            label=r'test - {}'.format(varID_0.replace('_', ' ')))
    
    plt.hist(d_test_1[:]['weights_' + class1],
            histtype='step', bins=bins, color='orange', linestyle='dashed',
            label=r'test - {}'.format(varID_1.replace('_', ' ')))
        
    plt.legend(loc='upper left')
    plt.xlabel('Event weight')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(os.path.join('plots','{}_{}_weights.pdf'.format(varID_0, varID_1)))
    plt.close()

def plot_jetn_trkn(d_0, d_val_0, d_test_0, d_1, d_val_1, d_test_1, varID_0, varID_1, class0, class1, jetn, trkn):
    '''
    '''
    safe_mkdir('plots')

    if jetn == 0:
        jet = 'leading_jet'
    elif jetn == 1:
        jet = 'subleading_jet'
    else:
        raise ValueError('jetn: select 0 for leading jet, 1 for subleading jet.')

    for i in range(len(FEATURES)):
        plt.figure(figsize=(8, 8)) 
        bins = np.linspace(d_0[:][jet][:, trkn, i].min(), d_0[:][jet][:, trkn, i].max(), 10)

        plt.hist(d_0[:][jet][:, trkn, i], weights=d_0[:]['weights_' + class0],
                histtype='step', normed=True, bins=bins, color='teal',
                label=r'train - {}'.format(varID_0.replace('_', ' ')))
        
        plt.hist(d_1[:][jet][:, trkn, i], weights=d_1[:]['weights_' + class1],
                histtype='step', normed=True, bins=bins, color='orange',
                label=r'train - {}'.format(varID_1.replace('_', ' ')))
        
        plt.hist(d_val_0[:][jet][:, trkn, i], weights=d_val_0[:]['weights_' + class0], alpha=0.1,
                histtype='stepfilled', normed=True, bins=bins, color='teal',
                label=r'val - {}'.format(varID_0.replace('_', ' ')))
        
        plt.hist(d_val_1[:][jet][:, trkn, i], weights=d_val_1[:]['weights_' + class1], alpha=0.1,
                histtype='stepfilled', normed=True, bins=bins, color='orange',
                label=r'val - {}'.format(varID_1.replace('_', ' ')))

        plt.hist(d_test_0[:][jet][:, trkn, i], weights=d_test_0[:]['weights_' + class0],
                histtype='step', normed=True, bins=bins, color='teal', linestyle='dashed',
                label=r'test - {}'.format(varID_0.replace('_', ' ')))
        
        plt.hist(d_test_1[:][jet][:, trkn, i], weights=d_test_1[:]['weights_' + class1],
                histtype='step', normed=True, bins=bins, color='orange', linestyle='dashed',
                label=r'test - {}'.format(varID_1.replace('_', ' ')))
        
        label = FEATURES[i]
        if FEATURES[i] not in ['eta', 'm', 'phi']:
            label += ' / 1000'
        plt.xlabel(label)
        plt.legend(fontsize=15)

        plt.savefig(os.path.join(
            'plots',
            'jet{}_trk{}_{}_{}_{}.pdf'.format(jetn, trkn, FEATURES[i], varID_0, varID_1, class0, class1)
        ))
        plt.close()

def plot_ntrack(d_0, d_val_0, d_test_0, d_1, d_val_1, d_test_1, varID_0, varID_1, class0, class1):
    safe_mkdir('plots')

    for jet in range(2):
        plt.figure(figsize=(8, 8))
        bins = np.linspace(0, d_0[:]['nparticles'][:, jet].max(), 50)
        _ = plt.hist(d_0[:]['nparticles'][:, jet], weights=d_0[:]['weights_' + class0],
            histtype='step', bins=bins, normed=True, color='teal',
            label=r'train - {}'.format(varID_0.replace('_', ' ')))

        _ = plt.hist(d_1[:]['nparticles'][:, jet], weights=d_1[:]['weights_' + class1],
            histtype='step', bins=bins, normed=True, color='orange',
            label=r'train - {}'.format(varID_1.replace('_', ' ')))

        _ = plt.hist(d_test_0[:]['nparticles'][:, jet], weights=d_test_0[:]['weights_' + class0],
            histtype='step', bins=bins, normed=True, color='teal', linestyle='dashed',
            label=r'test - {}'.format(varID_0.replace('_', ' ')))

        _ = plt.hist(d_test_1[:]['nparticles'][:, jet], weights=d_test_1[:]['weights_' + class1],
            histtype='step', bins=bins, normed=True, color='orange', linestyle='dashed',
            label=r'train - {}'.format(varID_1.replace('_', ' ')))

        _ = plt.hist(d_val_0[:]['nparticles'][:, jet], weights=d_val_0[:]['weights_' + class0],
            histtype='stepfilled', bins=bins, normed=True, color='teal', alpha=0.1,
            label=r'val - {}'.format(varID_0.replace('_', ' ')))

        _ = plt.hist(d_val_1[:]['nparticles'][:, jet], weights=d_val_1[:]['weights_' + class1],
            histtype='stepfilled', bins=bins, normed=True, color='orange', alpha=0.1,
            label=r'train - {}'.format(varID_1.replace('_', ' ')))

        plt.legend(fontsize=15)
        plt.xlabel('Number of tracks')
        plt.savefig(os.path.join('plots','jet{}_ntracks_{}_{}.pdf'.format(jet, varID_0, varID_1)))

def plot_batch_features(features_baseline, features_variation, varID_0, varID_1, weights_baseline, weights_variation, model_name):
    safe_mkdir('plots')
    for fn, (f0, f1) in enumerate(zip(features_baseline.T, features_variation.T)):
        plt.figure(figsize=(8, 8))
        bins=np.linspace(min(f0.min(), f1.min()), max(f0.max(), f1.max()), 30)
        _ = plt.hist(f0, 
                 bins=bins, histtype='step', label=r'test - {}'.format(varID_0.replace('_', ' ')),
                 weights=weights_baseline)
        _ = plt.hist(f1,
                 bins=bins, histtype='step', label=r'test - {}'.format(varID_1.replace('_', ' ')),
                 weights=weights_variation)
        plt.legend(loc='upper left')
        plt.savefig(os.path.join('plots','{}_{}_{}_{}.pdf'.format(model_name, varID_0, varID_1, fn)))


def plot_output(pred_baseline, pred_variation, varID_0, varID_1, weights_baseline, weights_variation, model_name, t=''):
    '''
    '''
    safe_mkdir('plots')
    plt.figure(figsize=(8, 8))
    min_ = min(pred_baseline.min(), pred_variation.min())
    max_ = max(pred_baseline.max(), pred_variation.max())
    bins=np.linspace(min_, max_, 30)
    _ = plt.hist(pred_baseline, 
             bins=bins, histtype='step', label=r'test - {}'.format(varID_0.replace('_', ' ')),
             weights=weights_baseline)
    _ = plt.hist(pred_variation,
             bins=bins, histtype='step', label=r'test - {}'.format(varID_1.replace('_', ' ')),
             weights=weights_variation)
    plt.legend(loc='upper left')
    plt.xlabel('Weighted NN Output')
    plt.savefig(os.path.join('plots','{}_{}_{}_output_{}.pdf'.format(model_name, varID_0, varID_1, t)))