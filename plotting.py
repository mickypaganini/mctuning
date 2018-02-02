import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from utils import safe_mkdir

matplotlib.rcParams.update({'font.size' : 20})

FEATURES = ['E', 'Et', 'eta', 'm', 'phi', 'pt', 'px', 'py', 'pz']

def plot_jetn_trkn(d, d_val, d_test, variation, jetn, trkn):

	if jetn == 0:
		jet = 'leading_jet'
	elif jetn == 1:
		jet = 'subleading_jet'
	else:
		raise ValueError('jetn: select 0 for leading jet, 1 for subleading jet.')

	for i in range(len(FEATURES)):
	    plt.figure(figsize=(8, 8)) 
	    bins = np.linspace(d[:][jet][:, trkn, i].min(), d[:][jet][:, trkn, i].max(), 10)

	    plt.hist(d[:][jet][:, trkn, i], weights=d[:]['weights_Baseline'],
	            histtype='step', normed=True, bins=bins, color='teal',
	            label=r'train - $\mu_\mathrm{FSR} = 1.0$')
	    
	    plt.hist(d[:][jet][:, trkn, i], weights=d[:]['weights_' + variation],
	            histtype='step', normed=True, bins=bins, color='orange',
	            label=r'train - $\mu_\mathrm{FSR} = $' + '.'.join(list(variation.split('fsr')[-1])))
	    
	    plt.hist(d_val[:][jet][:, trkn, i], weights=d_val[:]['weights_Baseline'], alpha=0.1,
	            histtype='stepfilled', normed=True, bins=bins, color='teal',
	            label=r'val - $\mu_\mathrm{FSR} = 1.0$')
	    
	    plt.hist(d_val[:][jet][:, trkn, i], weights=d_val[:]['weights_' + variation], alpha=0.1,
	            histtype='stepfilled', normed=True, bins=bins, color='orange',
	            label=r'val - $\mu_\mathrm{FSR} = $' + '.'.join(list(variation.split('fsr')[-1])))

	    plt.hist(d_test[:][jet][:, trkn, i], weights=d_test[:]['weights_Baseline'],
	            histtype='step', normed=True, bins=bins, color='teal', linestyle='dashed',
	            label=r'test - $\mu_\mathrm{FSR} = 1.0$')
	    
	    plt.hist(d_test[:][jet][:, trkn, i], weights=d_test[:]['weights_' + variation],
	            histtype='step', normed=True, bins=bins, color='orange', linestyle='dashed',
	            label=r'test - $\mu_\mathrm{FSR} = $' + '.'.join(list(variation.split('fsr')[-1])))
	    
	    label = FEATURES[i]
	    if FEATURES[i] not in ['eta', 'm', 'phi']:
	    	label += ' / 1000'
	    plt.xlabel(label)
	    plt.legend(fontsize=15)

	    safe_mkdir('plots')
	    plt.savefig(os.path.join(
	    	'plots',
	    	'{}_jet{}_trk{}_{}.pdf'.format(variation, jetn, trkn, FEATURES[i])
	    ))