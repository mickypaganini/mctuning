#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File: 
Description: Generate Pythia events from input specification,
             cluster jets with pyjet, save constituents of 2 leading jets,
             zero pad, save as DijetDataset derived from pytorch Dataset
Author: Michela Paganini (michela.paganini@yale.edu)
'''

from torch.utils.data import Dataset, DataLoader
import numpy as np
import operator
import os
import cPickle as pickle
import logging

from pyjet import cluster
from numpythia import Pythia, STATUS, HAS_END_VERTEX, ABS_PDG_ID

from utils import configure_logging
from pythiaconf import read_pythia_from_yaml, create_dataset_hash

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

    # only consider final state particles that are not neutrinos
    selection = ((STATUS == 1) & ~HAS_END_VERTEX & # final state
        (ABS_PDG_ID != 12) & (ABS_PDG_ID != 14) & (ABS_PDG_ID != 16)) # not a neutrino

    # placeholders for particles and weights
    events_particles = []
    events_weights = []
    # generate events from pythia
    for event in pythia(events=nevents):
        events_weights.append(event.weights)
        events_particles.append(event.all(selection))
    # convert to numpy arrays for usability
    events_weights = np.array(events_weights)
    events_weights = events_weights.view(
                        dtype=[(n, 'float64') for n in variations])
    events_particles = np.array(events_particles)

    return events_particles, events_weights



class DijetDataset(Dataset):
    '''
    Dataset of jet components for leading and subleading jet in QCD events.
    '''
    def __init__(self, config_path, nevents=100, max_len=50, min_lead_pt=None):
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
            
        
    def __len__(self):
        return self.nevents

    def __getitem__(self, idx):
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

    # check that requested variation is present in the config
    if not variation in cfg['UncertaintyBands:List']:
        raise ValueError('Requested variation not in ' + config)

    def _load_data(sample, nevents):
        logger.debug('Loading {} set'.format(sample))
        dataset_string = 'dataset_' + sample + '_' + create_dataset_hash(cfg,
        extra_args={
            'nevents': nevents,
            'max_len': maxlen,
            'min_lead_pt': min_lead_pt
        }) + '.pkl'

        if os.path.isfile(dataset_string):
            d = pickle.load(open(dataset_string, 'r'))
        else:
            d = DijetDataset(temp_filepath, nevents=nevents, max_len=maxlen, min_lead_pt=min_lead_pt)
            pickle.dump(d, open(dataset_string, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        shuffle = True if sample =='train' else False
        return DataLoader(d,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=4,
                    pin_memory=True) # for GPU

    return _load_data('train', ntrain), _load_data('val', nval), _load_data('test', ntest)
