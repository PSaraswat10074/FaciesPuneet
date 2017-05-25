from cumulative_filter_banks_1d import *
from wavelet_transform_1d import *
from facies_scattering_transform import *
from transform_utils import *
import numpy as np
from matplotlib import *

__all__ = ['factory1d']


def factory_generator(n, filter_options, scattering_options):
    filters = cumulative_filter_banks_1d(n, filter_options)
    if not 'M' in scattering_options:
        scattering_options['M'] = 2
    wavelet_operators = {}
    counter = 0
    for wave_num_idx in range(0, scattering_options['M'] + 1):
        wave_num_idx = min(wave_num_idx, len(filters) - 1)
        wavelet_operators[counter] = wavelet_lambda(filters[wave_num_idx], scattering_options)
        counter += 1
    return wavelet_operators


def wavelet_lambda(filter_opt, scatter_opt):
    return lambda x: wavelet_transform_1d(x, filter_opt, scatter_opt)


if __name__ == "__main__":
    averaging_size = 4096
    filt_opt = default_filter_options('audio', averaging_size)
    scat_opt = {'M': 2}
    filters = factory_generator(65536, filt_opt, scat_opt)
    scattered_signal, cov_mod_list = facies_scattering_transform(y, filters)
    print 'done'
