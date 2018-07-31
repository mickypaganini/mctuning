import numpy as np


class Likelihood1D(object):
    """docstring for Likelihood1D"""

    def __init__(self, bins):
        super(Likelihood1D, self).__init__()
        self._bins = bins

    def fit(self, X, y, weights=None):
        X = np.array(X)
        y = np.array(y)

        if len(X.shape) == 2 and not X.shape[-1] == 1:
            raise ValueError('Need 1d array for X or column vector')

        if len(y.shape) == 2 and not y.shape[-1] == 1:
            raise ValueError('Need 1d array for y or column vector')

        X = X.ravel()
        y = y.ravel()

        labelset = {l for l in y}
        assert len(labelset) == 2, 'Need only two labels for fitting'
        assert (1 in labelset) and (
            0 in labelset), 'Need both 0 and 1 in labels'

        if weights:
            # Get the probability per bin in signal and background
            self._sig_p, _ = np.histogram(X[y == 1], bins=self._bins,
                                          normed=True, weights=weights[y == 1])
            self._bkg_p, _ = np.histogram(X[y == 0], bins=self._bins,
                                          normed=True, weights=weights[y == 0])
        else:
            self._sig_p, _ = np.histogram(X[y == 1], bins=self._bins,
                                          normed=True)
            self._bkg_p, _ = np.histogram(X[y == 0], bins=self._bins,
                                          normed=True)

        # The ratio of these will be the likelihood ratio
        self._ratio = np.divide(self._sig_p, self._bkg_p)
        self._ratio[np.isinf(self._ratio)] = np.isfinite(self._ratio).max()
        self._ratio[np.isnan(self._ratio)] = min(np.isfinite(self._ratio).min(),
                                                 1e-10)
        return self

    def predict(self, X):
        try:
            return self._ratio[self._bins.searchsorted(X) - 1]
        except IndexError:
            if not hasattr(X, '__iter__'):
                return 1.0
            return np.array([1.0 for _ in X])


class Likelihood2D(object):
    """docstring for Likelihood2D"""

    def __init__(self, bins=(10, 10)):
        super(Likelihood2D, self).__init__()
        if not isinstance(bins, tuple):
            raise TypeError('Bins must be a length 2 tuple of ints or arrays')
        self._bins = bins

    def fit(self, X, y, weights=None):
        X = np.array(X)
        y = np.array(y)

        if not len(X.shape) == 2:
            raise ValueError('Need 2d array for X')

        if len(y.shape) == 2 and not y.shape[-1] == 1:
            raise ValueError('Need 1d array for y or column vector')

        y = y.ravel()

        labelset = {l for l in y}
        assert len(labelset) == 2, 'Need only two labels for fitting'
        assert (1 in labelset) and (
            0 in labelset), 'Need both 0 and 1 in labels'

        # Fit the bins on all data...
        _, binsx, binsy = np.histogram2d(*X.T, bins=self._bins, normed=True,
                                         weights=weights)
        self._bins = (binsx, binsy)

        if weights is not None:
            self._sig_p, _, _ = np.histogram2d(*X[y == 1].T,
                                               bins=self._bins,
                                               normed=True,
                                               weights=weights[y == 1])
            self._bkg_p, _, _ = np.histogram2d(*X[y == 0].T, bins=self._bins,
                                               normed=True,
                                               weights=weights[y == 0])
        else:
            self._sig_p, _, _ = np.histogram2d(*X[y == 1].T,
                                               bins=self._bins,
                                               normed=True)
            self._bkg_p, _, _ = np.histogram2d(*X[y == 0].T, bins=self._bins,
                                               normed=True)

        self._ratio = np.divide(self._sig_p, self._bkg_p)
        self._ratio[np.isinf(self._ratio)] = np.isfinite(self._ratio).max()
        self._ratio[np.isnan(self._ratio)] = min(np.isfinite(self._ratio).min(),
                                                 1e-10)
        return self

    def predict(self, X):
        ix_x = self._bins[0].searchsorted(X[:, 0]) - 1
        oob_lower_x = (ix_x == -1)
        oob_upper_x = (ix_x == self._ratio.shape[0])
        ix_x[oob_upper_x] = 0
        ix_x[oob_lower_x] = 0

        ix_y = self._bins[1].searchsorted(X[:, 1]) - 1
        oob_lower_y = (ix_y == -1)
        oob_upper_y = (ix_y == self._ratio.shape[1])
        ix_y[oob_upper_y] = 0
        ix_y[oob_lower_y] = 0
        llh = self._ratio[ix_x, ix_y]

        for fixes in [oob_upper_x, oob_lower_x, oob_upper_y, oob_lower_y]:
            if fixes.sum():
                llh[fixes] = 1.0

        return llh

