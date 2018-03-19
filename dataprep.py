#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File: 
Description: Generate Pythia events from input specification,
             cluster jets with pyjet, save constituents of 2 leading jets,
             zero pad, save as DijetDataset derived from pytorch Dataset
Author: Michela Paganini (michela.paganini@yale.edu)
'''

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import operator
import os
import cPickle as pickle
import logging
from joblib import Parallel, delayed
import h5py

from pyjet import cluster
from numpythia import Pythia, STATUS, HAS_END_VERTEX, ABS_PDG_ID

from utils import configure_logging
from pythiaconf import read_pythia_from_yaml, create_dataset_hash

from multiprocessing import Lock
import multiprocessing 
# LOCK = Lock()

# logging
configure_logging()
logger = logging.getLogger("Data Preparation")

def _properties(constituents):
    '''
    Arguments:
    ----------
        constituents: iterable of objects of type PseudoJet
        (see http://fastjet.fr/repo/doxygen-3.0.0/classfastjet_1_1PseudoJet.html)

    Returns:
    --------
        recarray of constituents' properties: e, et, eta, mass, phi, pt, px, py, pz
    '''
#   return np.array(
#       [np.array([c.e, c.et, c.eta, c.mass, c.phi, c.pt, c.px, c.py, c.pz])
#           for c in constituents]).view(dtype=[(l, 'float64')
#               for l in ('E', 'Et', 'eta', 'm', 'phi', 'pt', 'px', 'py', 'pz')])
    return np.array([np.array([c.e / 1000., c.et / 1000., c.eta, c.mass, c.phi, c.pt / 1000., c.px / 1000., c.py / 1000., c.pz / 1000.]) 
            for c in constituents])

def _padded(properties, max_len=50, n_properties=9):
    '''
    Arguments:
    ----------
        
    Returns:
    --------
    '''
    data = np.zeros((2, max_len, n_properties))
    lengths = []
    ntracks = []
    
    for i, jet in enumerate(properties):
    
        # take all particles unless there are more than max_len 
        nparticles = jet.shape[0] 
        ntracks.append(nparticles)
        data[i, :(min(nparticles, max_len)), :] = jet[:(min(nparticles, max_len)), :] 
        lengths.append(min(nparticles, max_len))
        
    return data, lengths, ntracks

def get_leadingjets_constituents(events_particles, jet_ptmin, max_len=50):
    '''
    Arguments:
    ----------
        events_particles:
        jet_ptmin:
        max_len:

    Returns:
    --------
        numpy array of shape (n_events, 2) where every jet entry is a list of
        constituents' kinematic properties
    '''
    # placeholder
    events_constituents = []
    events_lengths = []
    events_nparticles = []
    events_jetpts = []

    # loop through events
    for particles in events_particles:
        # cluster jets using anti_kt(R=0.4)
        sequence = cluster(particles[['E', 'px', 'py', 'pz']],
            algo='antikt',
            ep=True,
            R=0.4)
        # remove jets with pt < 10 GeV
        jets = sequence.inclusive_jets(ptmin=jet_ptmin)
        data, lengths, nparticles = _padded(
            [_properties(
                sorted(
                    j.constituents(), key=operator.attrgetter('pt'), reverse=True)
                ) for j in jets[0:2]
            ], max_len=max_len
        )
        events_constituents.append(data)
        events_lengths.append(lengths)
        events_nparticles.append(nparticles)
        events_jetpts.append([j.pt for j in jets[0:2]])

    return np.array(events_constituents), np.array(events_lengths), np.array(events_nparticles), np.array(events_jetpts)



def generate_events(config_path, nevents):
    '''
    Arguments:
    ----------
        config_path:
        nevents:

    Returns:
    --------
        events_particles:
        events_weights:
    '''
    pythia = Pythia(config_path, verbosity=1)
    variations = pythia.weight_labels
    if variations == ['']:
        variations = ['Baseline'] # to match hard-coded decision in Pythia

    # only consider final state particles that are not neutrinos
    selection = ((STATUS == 1) & ~HAS_END_VERTEX & # final state
        (ABS_PDG_ID != 12) & (ABS_PDG_ID != 14) & (ABS_PDG_ID != 16)) # not a neutrino

    # placeholders for particles and weights
    events_particles = []
    events_weights = []
    # generate events from pythia
    for event in pythia(events=nevents):
        events_weights.append(event.weights.astype('float32'))
        events_particles.append(event.all(selection))
    # convert to numpy arrays for usability
    events_weights = np.array(events_weights)
    # defaults to 'f0' if no variations are specified!!
    events_weights = events_weights.view(
                        dtype=[(n, 'float32') for n in variations])
    events_particles = np.array(events_particles)

    return events_particles, events_weights



class DijetDataset(Dataset):
    '''
    Dataset of jet components for leading and subleading jet in QCD events.
    '''
    def __init__(self, config_path=None, nevents=100, max_len=50, min_lead_pt=None):
        if config_path is not None:
            self.nevents = nevents
            particles, self.weights = generate_events(config_path, nevents)
            constituents, lengths, self.nparticles, self.jetpts = get_leadingjets_constituents(
                particles, jet_ptmin=10.0, max_len=max_len
            )
            self.leading_jet = np.squeeze(constituents[:, 0, :, :])
            self.subleading_jet = np.squeeze(constituents[:, 1, :, :])
            
            self.lengths = lengths
            
            if min_lead_pt is not None:
                self.nevents = sum(self.jetpts[:, 0] > min_lead_pt)
                self.leading_jet = self.leading_jet[self.jetpts[:, 0] > min_lead_pt]
                self.subleading_jet = self.subleading_jet[self.jetpts[:, 0] > min_lead_pt]
                self.lengths = self.lengths[self.jetpts[:, 0] > min_lead_pt]
                self.nparticles = self.nparticles[self.jetpts[:, 0] > min_lead_pt]
                self.weights = self.weights[self.jetpts[:, 0] > min_lead_pt]
                self.jetpts = self.jetpts[self.jetpts[:, 0] > min_lead_pt]
    
    def to_dict(self):
        """
        data = DijetDataset(...)

        d = data.to_dict()
        pickle.dump(d, open(...))
        """
        d = dict(self.__dict__)
        for k in d:
            if k not in ['nevents', 'weights']:
                d[k] = d[k].astype('float32')
        return d

    @classmethod
    def from_dict(cls, d):
        """
        d = pickle.load(open(...))
        data = DijetDataset.from_dict(d)
        """
        obj = cls()
        obj.__dict__.update(d)
        return obj

    @staticmethod
    def concat(sequence):
        dicts = [d.to_dict() for d in sequence]
        output_dict = {
            field : np.concatenate(tuple(dic[field] for dic in dicts), axis=0)
                for field in dicts[0].keys() if dicts[0][field].shape
        }
        output_dict.update({
            field : sum(dic[field] for dic in dicts) for field in dicts[0].keys() if not dicts[0][field].shape
        })
        return DijetDataset.from_dict(output_dict)
        
    def __len__(self):
        return self.nevents

    def __getitem__(self, idx):
        # with LOCK:
        sample = {
            'leading_jet': self.leading_jet[idx],
            'subleading_jet': self.subleading_jet[idx],
            'unsorted_lengths': self.lengths[idx],
            'nparticles': self.nparticles[idx],
            'jet_pts': self.jetpts[idx]
        }
        sample.update({
            'weights_' + name: self.weights[name][idx] for name in self.weights.dtype.names})
        return sample


def load_data(config, variation, ntrain, nval, ntest, maxlen, min_lead_pt, batch_size):
    # load datasets
    temp_filepath, cfg = read_pythia_from_yaml(config)

    if variation == 'Baseline':
        # TODO: allow more parameter variations
        if 'TimeShower:renormMultFac' in cfg:
            varID = 'unweighted_fsr' + str(cfg['TimeShower:renormMultFac']).replace('.', '')
        elif 'UncertaintyBands:List' in cfg:
            varID = 'weighted_Baseline'
    else:
        if 'UncertaintyBands:List' not in cfg or variation not in cfg['UncertaintyBands:List']:
            raise ValueError('Requested variation {} not in {}'.format(variation, config))
        varID = 'weighted_' + variation 

    def _load_data(sample, nevents):
        logger.debug('Loading {} set'.format(sample))
        dataset_string = os.path.join('data', 'dataset_' + sample + '_' + create_dataset_hash(cfg,
        extra_args={
            'nevents': nevents,
            'max_len': maxlen,
            'min_lead_pt': min_lead_pt
        }) + '.h5') #.pkl
        print dataset_string
        if os.path.isfile(dataset_string):
            #dic = pickle.load(open(dataset_string, 'r'))
            # f = h5py.File(dataset_string, 'r')
            with h5py.File(dataset_string) as f:
                def convert(x):
                    if len(x.shape) > 0:
                        return np.array(x[:])
                    return x
                dic = {key : convert(f[key]) for key in f.keys()}
            # if 'nevents' in dic:
                dic['nevents'] = int(dic['nevents'].value)
            d = DijetDataset.from_dict(dic)
        else:
            ncpu = multiprocessing.cpu_count() - 1 
            dataset_list = Parallel(n_jobs=ncpu, verbose=True)(delayed(DijetDataset)(
                temp_filepath, nevents=nevents/ncpu, max_len=maxlen, min_lead_pt=min_lead_pt) for _ in range(ncpu))
            d = DijetDataset.concat(dataset_list)
            #pickle.dump(d.to_dict(), open(dataset_string, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            with h5py.File(dataset_string) as f:
                for key, value in d.to_dict().iteritems():
                    f[key] = value
        pin_memory = True if torch.cuda.is_available() else False
        return DataLoader(d,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=6,
                    # num_workers=0,
                    pin_memory=pin_memory) # for GPU

    return _load_data('train', ntrain), _load_data('val', nval), _load_data('test', ntest), varID
