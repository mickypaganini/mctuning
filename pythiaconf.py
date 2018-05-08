def read_pythia_from_yaml(yaml_input_file, output_pythia_file=None,
                          output_yaml_file=None, indent=False, seed=None):
    """Reads a YAML config file and converts it to a format usable by Pythia.

    Args:
        yaml_input_file (str | file): Either a file path or a file object. This 
            can either be a template (which defines Proposals, NumSamples, 
            and Mode parameters) or a vanilla YAML config
        output_pythia_file (None | str): If passed, will write the Pythia 
            formatted config file to the passed file path.
        output_yaml_file (None | str): If passed, will write the final YAML
            formatted config file to the passed file path (useful for templated 
            use case).
        indent (bool): Whether or not to indent *:List blocks.

    Returns:
        A tuple (filepath, cfg)
        filepath: filepath to the written Pythia-ready config file. 
            If output_pythia_file is None, will be a temporary file, else, 
            will be the filepath specified in output_pythia_file.
        cfg: a configuration dictionary of parameters that can be used by
            downstream functionality to obtain a hash
    """
    import yaml
    import tempfile

    ROOT_LEVEL = 0
    LINEITEM_LEVEL = 1
    INDIVIDUAL_LEVEL = 2

    indent = '' if not indent else 4 * ' '

    def convert_payload(param_name, param_value, recurse=ROOT_LEVEL):

        if isinstance(param_value, dict):
            if recurse == ROOT_LEVEL:
                inside = (',\n' + indent).join(
                    convert_payload(*item, recurse=1)
                    for item in param_value.items()
                )
                return param_name + ' = {\n' + indent + inside + '\n}'
            return param_name + ' ' + ' '.join(
                convert_payload(*item, recurse=LINEITEM_LEVEL)
                for item in param_value.items()
            )
        if isinstance(param_value, bool):
            param_value = 'on' if param_value else 'off'
        space = '' if recurse == INDIVIDUAL_LEVEL else ' '
        return '{n}{s}={s}{v}'.format(n=param_name, v=param_value, s=space)

    if isinstance(yaml_input_file, str):
        infp = open(yaml_input_file, 'r')
    else:
        infp = yaml_input_file

    cfg = yaml.load(infp)

    if 'Proposals' in cfg:
        cfg = generate_parameterized_cfg(cfg, seed=seed)

    if output_yaml_file is not None:
        if isinstance(output_yaml_file, str):
            ofp = open(output_yaml_file, 'w')
        else:
            ofp = output_yaml_file
        yaml.dump(cfg, ofp, default_flow_style=False)

    output_pythia_cfg = '\n'.join(convert_payload(*it) for it in cfg.items())
    if output_pythia_file is None:
        _, tempfp = tempfile.mkstemp()
        fp = open(tempfp, 'w')
    else:
        fp = open(output_pythia_file, 'w')

    fp.write(output_pythia_cfg)
    fp.close()

    return fp.name, cfg


def normalize_cfg(cfg, extra_args=None):
    cfg = dict(cfg)

    def format_value(v):
        if isinstance(v, dict):
            return normalize_cfg(v)
        return v

    if extra_args is not None:
        cfg.update(extra_args)
    return sorted([
        '{} = {}'.format(k, format_value(v)) for k, v in cfg.items()
    ])


def generate_parameterized_cfg(templated_yaml_cfg,
                               under='UncertaintyBands:List', seed=None):
    import pydoc
    from collections import defaultdict
    import numpy as np

    cfg = dict(templated_yaml_cfg)
    proposals = cfg.pop('Proposals')
    nb_samples = cfg.pop('NumSamples', 10)
    mode = cfg.pop('Mode', 'sample')

    class clipped_dist(object):

        def __init__(self, dist, min=None, max=None):
            self.dist = dist
            self.min = min or -np.inf
            self.max = max or np.inf

        def rvs(self, size=None, *args, **kwargs):
            return np.clip(self.dist.rvs(size, *args, **kwargs), self.min,
                           self.max)

    param_grid = {}
    grid_search_possible = True
    for parameter_name, distribution in proposals.items():
        if not (bool(distribution.get('values')) ^ bool(distribution.get('dist'))):
            raise RuntimeError('Need either values or dist')

        if distribution.get('dist'):
            grid_search_possible = False
            dist_fn = pydoc.locate(distribution.get('dist'))
            dist_fn = dist_fn(
                *distribution.get('args', []),
                **distribution.get('kwargs', {})
            )
            candidate = clipped_dist(
                dist=dist_fn,
                min=distribution.get('min'),
                max=distribution.get('max')
            )
        else:
            candidate = distribution.get('values')

        param_grid[parameter_name] = candidate

    if mode == 'grid':
        if not grid_search_possible:
            raise ValueError('Cant do grid search, found continuous dist')
        from sklearn.model_selection import ParameterGrid
        sampler = ParameterGrid(param_grid)
        print('Ignoring number of samples in mode == grid')
    else:
        from sklearn.model_selection import ParameterSampler
        sampler = ParameterSampler(param_grid, n_iter=nb_samples,
                                   random_state=seed)

    def safe_round(number, places=6):
        if isinstance(number, float):
            return float(round(number, places))
        if not isinstance(number, basestring):
            return int(number)
        return number

    try:
        param_grid = [
            {k: safe_round(v, 6) for (k, v) in d.items()}
            for d in list(sampler)
        ]
    except ValueError:
        raise ValueError(
            'You likely are asking more samples than there are parameter '
            'variations. Please lower NumSamples or turn Mode to grid'
        )

    names = []
    name_tracker = set()
    for elem in param_grid:
        key = sorted(list(elem.items()))
        key = ';'.join('{}={}'.format(param, value) for param, value in key)
        if key in name_tracker:
            key = 'SKIP'
        else:
            name_tracker.add(key)
        names.append(key)

    cfg[under] = dict((k, v) for k, v in zip(names, param_grid) if k != 'SKIP')
    return cfg


def create_dataset_hash(cfg, extra_args=None):
    """Get a hash for a set of parameters for data generation

    Args:
        cfg (dict): dictionary output from read_pythia_from_yaml
        extra_args (dict, optional): Additional parameters to encode, passed 
            in as a dictionary

    Returns:
        str: truncated hash of the full config
    """
    import hashlib
    normalized = normalize_cfg(cfg, extra_args)
    encoding = ''.join(normalized).replace(' ', '').replace('\n', '')
    md5 = hashlib.md5()
    md5.update(encoding)
    return md5.hexdigest()[:7]

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        sys.stdout.write('Usage: {} FILE.yaml [FILE2.yaml ...]\n'
                         .format(sys.argv[0]))
        sys.exit(2)
    for fname in sys.argv[1:]:
        hashname = create_dataset_hash(
            read_pythia_from_yaml(fname)[1],
            extra_args={'nevents': 100000, 'max_len': 100, 'min_lead_pt': 500}
        )
        sys.stdout.write('{}: {}\n'.format(fname, hashname))
