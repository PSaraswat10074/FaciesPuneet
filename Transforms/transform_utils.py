import numpy as np


def t_j(averaging_size, filter_options):
    if not 'Q' in filter_options:
        filter_options['Q'] = 1
    if not 'B' in filter_options:
        filter_options['B'] = filter_options['Q']
    if not 'phi_bw_multiplier' in filter_options:
        filter_options['phi_bw_multiplier'] = 1 + filter_options['Q'] == 1
    j = 1.0 + np.round(
        np.log2(averaging_size / (4.0 * filter_options['B'] / filter_options['phi_bw_multiplier'])) * filter_options[
            'Q'])
    return j


def default_filter_options(filter_type, averaging_size):
    filter_options = {}
    if filter_type == 'image':
        filter_options['J'] = np.ceil(np.log2(averaging_size))
    elif filter_type == 'audio':
        filter_options['Q'] = [8, 1]
        filter_options['J'] = t_j(averaging_size, filter_options)
    elif filter_type == 'dyadic':
        filter_options['Q'] = 1
        filter_options['J'] = t_j(averaging_size, filter_options)
    return filter_options
