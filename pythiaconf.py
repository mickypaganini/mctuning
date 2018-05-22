def read_pythia_from_yaml(yaml_input_file, output_pythia_file=None):
    """Reads a YAML config file and converts it to a format usable by Pythia.

    Args:
        yaml_input_file (str | file): Either a file path or a file objects
        output_pythia_file (None | str): If passed, will write the Pythia 
            formatted config file to the passed file path.

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

    def _convert_payload(param_name, param_value, recurse=0):

        if isinstance(param_value, dict):
            if recurse == 0:
                inside = ',\n'.join(_convert_payload(*item, recurse=1)
                                    for item in param_value.items())
                return param_name + ' = {\n' + inside + '\n}'
            return param_name + ' ' + ' '.join(
                _convert_payload(*item, recurse=2)
                for item in param_value.items()
            )
        if isinstance(param_value, bool):
            param_value = 'on' if param_value else 'off'
        space = '' if recurse == 2 else ' '
        return '{n}{s}={s}{v}'.format(n=param_name, v=param_value, s=space)

    if isinstance(yaml_input_file, str):
        infp = open(yaml_input_file, 'r')
    else:
        infp = yaml_input_file

    cfg = yaml.load(infp)
    output_pythia_cfg = '\n'.join(_convert_payload(*it) for it in cfg.items())
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
        print 'Usage: {} FILE.yaml [FILE2.yaml ...]'.format(sys.argv[0])
        sys.exit(2)
    for fname in sys.argv[1:]:
        hashname = create_dataset_hash(
            read_pythia_from_yaml(fname)[1], 
            extra_args={'nevents': 100000, 'max_len': 100, 'min_lead_pt': 500}
        )
        print '{}: {}'.format(fname, hashname)
    
    
