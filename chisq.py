import numpy as np


def pseudo_chi_squared(X, y, weights=None, bins=10):
    """
    Calculate the \chi^2 statistic between two empirical distributions via a
    naive summation of the individual, marginal 1D chi^2 distributions
    """
    X = np.array(X)

    # Make sure we always have a (nb_samples, nb_features) array
    if (1 in X.shape) or len(X.shape) == 1:
        X = X.reshape(-1, 1)

    _, nb_features = X.shape

    if isinstance(bins, (list, tuple)):
        if len(bins) != nb_features:
            raise ValueError('Expected number of bin definitions, {}, to be '
                             'equal to the number of features, {}.'
                             .format(len(bins), nb_features))
    else:
        bins = [bins] * nb_features

    chi_sq = 0.0
    for bin_def, col in zip(bins, X.T):
        chi_sq += pseudo_chi_squared_nd(col, y, weights=weights,
                                        bins=bin_def, nd=1)

    return chi_sq


def pseudo_chi_squared_nd(X, y, weights=None, bins=10, nd='D'):
    """
    Calculate the \chi^2 statistic between two empirical distributions
    """
    if nd == 1:
        hist_fn = np.histogram
    elif nd == 2:
        hist_fn = np.histogram2d
    elif nd == 'D':
        hist_fn = np.histogramdd
    else:
        raise ValueError('nd can be one of 1, 2, or D')

    sample_a, sample_b = (X[y == 0], X[y == 1])
    if weights is not None:
        weights_a, weights_b = (weights[y == 0], weights[y == 1])
        weights_sq_a, weights_sq_b = (weights_a ** 2, weights_b ** 2)
    else:
        weights_a, weights_b = (None, None)
        weights_sq_a, weights_sq_b = (None, None)

    SIGMA_THRESH = 0.01

    # We cycle the bins determined from distribution a due to the fact that we
    # want identical bins. Ideally, one will specify bins manually, so this
    # won't have any influence :)
    counts_a, bins = hist_fn(sample_a, weights=weights_a, bins=bins)
    sigma_sq_a, _ = hist_fn(sample_a, weights=weights_sq_a, bins=bins)

    counts_b, _ = hist_fn(sample_b, weights=weights_b, bins=bins)
    sigma_sq_b, _ = hist_fn(sample_b, weights=weights_sq_b, bins=bins)

    below_thresh_a = (np.sqrt(sigma_sq_a) /
                      np.clip(counts_a, 1e-10, np.inf)) < SIGMA_THRESH
    sigma_sq_a[below_thresh_a] = SIGMA_THRESH * counts_a[below_thresh_a]
    below_thresh_b = (np.sqrt(sigma_sq_b) /
                      np.clip(counts_b, 1e-10, np.inf)) < SIGMA_THRESH
    sigma_sq_b[below_thresh_b] = SIGMA_THRESH * counts_b[below_thresh_b]

    chi_squared = np.nansum(np.square(counts_a - counts_b) /
                            (sigma_sq_a + sigma_sq_b))
    return chi_squared

