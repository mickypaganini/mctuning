import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from keras import backend as K
K.set_session(sess)

import h5py
import numpy as np
from sklearn.utils import shuffle
import sys
from sklearn.metrics import roc_auc_score

def load_data(filepath):
    with h5py.File(filepath) as f:
        def convert(x):
            if len(x.shape) > 0:
                return np.array(x[:])
            return x
        d = {key : convert(f[key]) for key in f.keys()}
        d['nevents'] = int(d['nevents'].value)
    return d

if len(sys.argv) == 1:
	print 'Usage: {} [ntrack|rnn] SHA'.format(sys.argv[0])
	sys.exit(2)

MODEL = sys.argv[1]
SHA = sys.argv[2]

if MODEL not in ['ntrack', 'rnn']:
	raise RuntimeError()

print 'Loading data'
# mu = 1.0
d_train_0 = load_data('./data/dataset_train_e1152e3.h5')
d_test_0 = load_data('./data/dataset_test_e1152e3.h5')
d_val_0 = load_data('./data/dataset_val_e1152e3.h5')

d_train_1 = load_data('./data/dataset_train_' + SHA + '.h5')
d_test_1 = load_data('./data/dataset_test_' + SHA + '.h5')
d_val_1 = load_data('./data/dataset_val_' + SHA + '.h5')



def make_wxy(d0, d1):
    X_ntrack = np.concatenate((d0['nparticles'], d1['nparticles']))
    X_lead = np.concatenate((d0['leading_jet'], d1['leading_jet']))
    X_sublead = np.concatenate((d0['subleading_jet'], d1['subleading_jet']))
    y = np.concatenate(
        (np.zeros(d0['leading_jet'].shape[0]), np.ones(d1['leading_jet'].shape[0]))
    )
    w = np.concatenate((d0['weights']['Baseline'], d1['weights']['Baseline']))
    return X_ntrack, X_lead, X_sublead, y, w

print 'Preprocessing data'
X_ntrack_train, X_lead_train, X_sublead_train, y_train, w_train = make_wxy(d_train_0, d_train_1)
X_ntrack_val, X_lead_val, X_sublead_val, y_val, w_val = make_wxy(d_val_0, d_val_1)
X_ntrack_test, X_lead_test, X_sublead_test, y_test, w_test = make_wxy(d_test_0, d_test_1)

X_ntrack_train, X_lead_train, X_sublead_train, y_train, w_train = shuffle(
    X_ntrack_train, X_lead_train, X_sublead_train, y_train, w_train)
X_ntrack_val, X_lead_val, X_sublead_val, y_val, w_val = shuffle(
    X_ntrack_val, X_lead_val, X_sublead_val, y_val, w_val)
X_ntrack_test, X_lead_test, X_sublead_test, y_test, w_test = shuffle(
    X_ntrack_test, X_lead_test, X_sublead_test, y_test, w_test)

X_ntrack_train = (X_ntrack_train - 50) / 100.
X_ntrack_val = (X_ntrack_val - 50) / 100.
X_ntrack_test = (X_ntrack_test - 50) / 100.


from keras.layers import Input, LSTM, Dropout, Bidirectional, concatenate, Masking, Dense, GRU
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
print 'Training'
if MODEL == 'rnn':
	x_lead = Input(X_lead_train.shape[1:])
	x_sublead = Input(X_sublead_train.shape[1:])

	masked_lead = x_lead #Masking()(x_lead)
	masked_sublead = x_sublead #Masking()(x_sublead)


	lstm_lead = Bidirectional(GRU(32, recurrent_dropout=0.2))(masked_lead)
	lstm_sublead = Bidirectional(GRU(32, recurrent_dropout=0.2))(masked_sublead)
	h = concatenate([lstm_lead, lstm_sublead])
	h = Dropout(0.6)(h)
	h = Dense(128, activation='relu')(h)
	h = Dropout(0.6)(h)
	# h = Dense(64, activation='relu')(h)
	# h = Dropout(0.6)(h)
	out = Dense(1, activation='sigmoid')(h)


	model = Model(inputs=[x_lead, x_sublead], outputs=out)


	model.compile('RMSprop', 'binary_crossentropy')

	fp = '{}_{}_best.h5'.format(SHA, MODEL)
	try:
		model.fit([X_lead_train, X_sublead_train], y_train, sample_weight=w_train.ravel(), verbose=2, batch_size=128,
	          validation_data=([X_lead_val, X_sublead_val], y_val, w_val.ravel()),
	          callbacks=[
	                        ModelCheckpoint(fp, monitor='val_loss', verbose=1, save_best_only=True),
	                        EarlyStopping(monitor='val_loss', patience=20, verbose=1)

	                    ],
	          epochs=100)
	except KeyboardInterrupt:
		print 'ending'
	model.load_weights(fp)
	# yhat_val = model.predict([X_lead_val, X_sublead_val], verbose=1, batch_size=1024)
	# yhat_train = model.predict([X_lead_train, X_sublead_train], verbose=1, batch_size=1024)
	yhat_test = model.predict([X_lead_test, X_sublead_test], verbose=1, batch_size=1024)
elif MODEL == 'ntrack':
	x = Input(X_ntrack_train.shape[1:])
	h = Dense(20, activation='relu')(x)
	h = Dropout(0.2)(h)
	h = Dense(20, activation='relu')(h)
	h = Dropout(0.2)(h)
	h = Dense(128, activation='relu')(h)
	h = Dropout(0.2)(h)
	h = Dense(128, activation='relu')(h)
	h = Dropout(0.2)(h)
	h = Dense(20, activation='relu')(h)
	h = Dropout(0.2)(h)
	out = Dense(1, activation='sigmoid')(h)


	ntrack_model = Model(x, out)


	ntrack_model.compile('RMSProp', 'binary_crossentropy')


	fp = '{}_{}_best.h5'.format(SHA, MODEL)
	try:
		ntrack_model.fit(X_ntrack_train, y_train, sample_weight=w_train.ravel(), verbose=2, batch_size=100,
		                    validation_data=(X_ntrack_val, y_val, w_val.ravel()),
		                    callbacks=[
		                        ModelCheckpoint(fp, monitor='val_loss', verbose=1, save_best_only=True),
		                        EarlyStopping(monitor='val_loss', patience=20, verbose=1)

		                    ],
		                    epochs=200)
	except KeyboardInterrupt:
		print 'ending'
	ntrack_model.load_weights(fp)
	# yhat_val = ntrack_model.predict(X_ntrack_val, batch_size=1024, verbose=1)
	# yhat_train = ntrack_model.predict(X_ntrack_train, batch_size=1024, verbose=1)
	yhat_test = ntrack_model.predict(X_ntrack_test, batch_size=1024, verbose=1)


print '=' * 40
print 'RESULTS DATASET: {}, MODEL: {}'.format(SHA, MODEL)
for batch_size in [10, 25, 50, 100]:
	batched_0 = []
	batched_1 = []
	for n in range(sum(y_val == 0)/batch_size):
	    batched_0.append(np.prod(yhat_test[y_test == 0][n*batch_size: (n+1)*batch_size]))
	for n in range(sum(y_val == 1)/batch_size): 
	    batched_1.append(np.prod(yhat_test[y_test == 1][n*batch_size: (n+1)*batch_size]))

	auc = roc_auc_score(
	    np.concatenate((np.zeros_like(batched_0), np.ones_like(batched_1))),
	    np.concatenate((batched_0, batched_1))
	)

	print 'BATCH SIZE: {}, ROC AUC: {}'.format(batch_size, auc)
print '=' * 40