import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import numpy as np

from dataprep import load_data
from models import DoubleLSTM, NTrackModel

def train_on_batch(model, optimizer, epoch, batch_idx, data, name_weights):
	
	weights = data[name_weights].type(torch.FloatTensor)
	batch_size = weights.shape[0]
	batch_weights = Variable(
			weights.resize_(1, batch_size),
			requires_grad=False) / torch.sum(weights) # normalized

	if isinstance(model, NTrackModel):
		inputs = Variable(data['nparticles'].type(torch.FloatTensor), requires_grad=False) # B x 2

		optimizer.zero_grad()
		predictions = model(inputs, batch_weights, batch_size)

	elif isinstance(model, DoubleLSTM):
		leading_input = Variable(data['leading_jet'].type(torch.FloatTensor), requires_grad=False)
		subleading_input = Variable(data['subleading_jet'].type(torch.FloatTensor), requires_grad=False)
		unsorted_lengths = data['unsorted_lengths'].numpy()

		optimizer.zero_grad()
		predictions = model(leading_input, subleading_input, unsorted_lengths, batch_weights, batch_size)
	else:
		raise TypeError

	if name_weights == 'weights_Baseline':
		targets = Variable(torch.zeros((batch_size, 1)), requires_grad=False)
	else:
		targets = Variable(torch.ones((batch_size, 1)), requires_grad=False)
		
	loss_function = nn.BCEWithLogitsLoss() #weight=batch_weights)
	loss = loss_function(predictions, targets)
	loss.backward()
	optimizer.step()
	
	batch_weighted_loss = loss * batch_size # to compute per epoch loss

	return batch_weighted_loss


def validate_on_batch(model, data_val, variation, loss_function):

	weights_baseline = data_val['weights_Baseline'].type(torch.FloatTensor)
	weights_variation = data_val['weights_' + variation].type(torch.FloatTensor)
	batch_size = weights_baseline.shape[0]

	batch_weights_baseline = Variable(
			weights_baseline.resize_(1, batch_size),
			requires_grad=False, volatile=True) / torch.sum(weights_baseline) # normalized

	batch_weights_variation = Variable(
			weights_variation.resize_(1, batch_size),
			requires_grad=False, volatile=True) / torch.sum(weights_variation) # normalized

	if isinstance(model, NTrackModel):
		inputs = Variable(data_val['nparticles'].type(torch.FloatTensor), requires_grad=False) # B x 2
		pred_baseline_val = model(inputs, batch_weights_baseline, batch_size)#)#.data.numpy()
		pred_variation_val = model(inputs, batch_weights_variation, batch_size)#)#.data.numpy()

	elif isinstance(model, DoubleLSTM):
		leading_input = Variable(data_val['leading_jet'].type(torch.FloatTensor), volatile=True)
		subleading_input = Variable(data_val['subleading_jet'].type(torch.FloatTensor), volatile=True)
		unsorted_lengths = data_val['unsorted_lengths'].numpy()

		pred_baseline_val = model(leading_input, subleading_input, unsorted_lengths, batch_weights_baseline, batch_size)#)#.data.numpy()
		pred_variation_val = model(leading_input, subleading_input, unsorted_lengths, batch_weights_variation, batch_size)#)#.data.numpy()

	else:
		raise TypeError
	
	targets_baseline = Variable(torch.zeros((batch_size, 1)), requires_grad=False, volatile=True)
	targets_variation = Variable(torch.ones((batch_size, 1)), requires_grad=False, volatile=True)

	loss_val_increment = batch_size * (
		loss_function(pred_baseline_val, targets_baseline) + loss_function(pred_variation_val, targets_variation)
		) / 2.
	return loss_val_increment


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
				
				batch_weighted_loss_baseline_i = train_on_batch(
					model, optimizer, epoch, batch_idx, data, 'weights_Baseline'
				)
				
				batch_weighted_loss_variation_i = train_on_batch(
					model, optimizer, epoch, batch_idx, data, 'weights_' + variation
				)
				
				# accumulate per-epoch loss
				batch_weighted_loss_baseline_epoch += batch_weighted_loss_baseline_i
				batch_weighted_loss_variation_epoch += batch_weighted_loss_variation_i
				

			print 'Epoch {}: Loss Baseline = {:0.5f}; Loss {} = {:0.5f}; Total = {:0.5f}'.format(
				epoch,
				batch_weighted_loss_baseline_epoch.data[0] / n_training,
				variation,
				batch_weighted_loss_variation_epoch.data[0] / n_training,
				(batch_weighted_loss_baseline_epoch.data[0] + batch_weighted_loss_variation_epoch.data[0]) / (2 * n_training)
			)
			
			# validate
			model.eval()
			loss_val = 0
			loss_function = nn.BCEWithLogitsLoss()
			
			for batch_idx_val, data_val in enumerate(validation_data):
				loss_val += validate_on_batch(model, data_val, variation, loss_function)

			loss_val /= n_validation
			loss_val = float(loss_val)
			print 'Epoch {}: Validation Loss = {:0.5f}'.format(epoch, loss_val)

			# early stopping
			if loss_val < best_loss:
				print 'Validation loss improved from {:0.5f} to {:0.5f}'.format(best_loss, loss_val)
				best_loss = loss_val
				wait = 0
				print 'Saving checkpoint at ' + checkpoint_path
				torch.save(model.state_dict(), checkpoint_path)

			else:
				wait += 1
				if wait >= patience - 1:
					print 'Stopping early.'
					break
	except KeyboardInterrupt:
		print 'Training ended early.'
		
	print 'Restoring best weights from checkpoint at ' + checkpoint_path
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
	parser.add_argument('--lr', type=int, default=0.01)
	parser.add_argument('--model', type=str, default='rnn', help='Either rnn or ntrack')
	parser.add_argument('--checkpoint', type=str, default='checkpoint')
	args = parser.parse_args()

	# check arguments
	if args.model not in ['rnn', 'ntrack']:
		raise ValueError('--model can only be one of ["rnn", "ntrack"]')

	# load or make data
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

		optimizer = optim.SGD(model.parameters(), lr=args.lr)
	else:
		model = NTrackModel(input_size=2)
		optimizer = optim.SGD(model.parameters(), lr=args.lr)

	train(model,
		optimizer,
		args.variation,
		dataloader,
		dataloader_val,
		checkpoint_path=args.checkpoint + '_' + args.model + '_' + args.variation + '.pth',
		epochs=args.nepochs,
		patience=args.patience
	)