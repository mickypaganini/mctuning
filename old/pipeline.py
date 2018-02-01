
import numpy as np
def pipeline(theta, sigma, batch_size=1000, method='avo'):
	# repeat until delta uncertainty is below epsilon:
	#while delta > epsilon:
	for _ in range(10):
		# -- set initial thetas and priors from arguments
		x_gen_train = sample(theta, sigma, batch_size)
		x_gen_test = sample(theta, sigma, batch_size)
		# -- sample target data distribution N times
		x_data = sample(np.array([0.55, 2.0]), np.arra([1.5, 0.5]), batch_size)
		# -- train discriminant (to convergence?) to distinguish samples
		d = train_discriminator(x_gen, x_data)
		yhat = d.predict(x_gen_test)
		# -- update parameters
		theta, sigma = update(theta, sigma, yhat, method)
	print theta, sigma

def train_discriminator(x_gen, x_data):
	#TODO
	return d

def update(theta, sigma, yhat, method='avo'):
	if method == 'avo':
		return avo(theta, sigma, yhat)
	else:
		raise NotImplementedError('Only the AVO method is currently available.')

def avo(theta, sigma, yhat):
	line_13 = np.mean(-yhat * grad_log_normal, axis=0)
	line_14 = np.mean(grad_normal * log_normal, axis=0)
	# -- update guess theta and its uncertainty
	theta, sigma = Adam(theta, sigma, line_13 + gamma * line_14)

def sample(theta, sigma, batch_size):
	theta_m = np.random.normal(loc=theta, scale=sigma, size=(batch_size, len(theta))) # samples from proposals
	# -- evaluate black box N times to produce generated sample
	z_0 = np.random.normal(theta_m[:, 0], 1)
	z_1 = np.random.normal(theta_m[:, 1], 3)
	# z_3 = ...
	z = np.concatenate((z_0, z_1), axis=0).T # 1000x2
	R = [[0.5, 1.2], [1.2, 3.1]]
	x = np.dot(R, z) #1000x2 = batch_size x dimensionality of observation space
	return x

def _parse_input(filepath):
	'''
	Args:
	-----
		filepath: string, path to input YAML file of format provided in 'Examples'
	Returns:
	--------
		theta: 1d numpy array of initial parameter values
		sigma: 1d numpy array of initial gaussian prior width values
	Examples:
	---------
	an example file would have the following line entries of space-separated values:
		theta0_name:
			value: 0.5
			sigma: 1.0
		theta1_name:
			value: 2.4
			sigma: 0.1
	'''
	import yaml
	theta = []
	sigma = []
	with open(filepath, 'r') as stream:
		try:
			s = yaml.load(stream)
		except yaml.YAMLError as exc:
			#logger.error(exc)
			raise exc
	for name, properties in s.iteritems():
		theta.append(properties['value'])
		sigma.append(properties['sigma'])
	#logger.debug('Identified {} parameters'.format(len(theta)))
	print 'Identified {} parameters'.format(len(theta))
	return np.array(theta), np.array(sigma)


if __name__ == '__main__':
	import sys
	import argparse
	# -- read in command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'theta_init',
		help='Path to YAML file with initial theta guesses and errors')
	parser.add_argument(
		'--batch_size', type=int, default=1000,
		help='Number of data samples per batch.')
	parser.add_argument(
		'--method', default='avo',
		help='Method used to update the parameters. Only option = avo')
	args = parser.parse_args()
	# -- parse input text file
	theta, sigma = _parse_input(args.theta_init)
	# -- call main
	sys.exit(pipeline(theta, sigma, args.batch_size, args.method))