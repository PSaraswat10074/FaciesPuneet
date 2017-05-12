from cumulative_filter_banks_1d import *
from wavelet_transform_1d import*
from matplotlib import *
__all__ = ['factory1d']


def factory_generator(n, filter_options, scattering_options):
    filters = cumulative_filter_banks_1d(n, filter_options)
    if not 'M' in scattering_options:
        scattering_options['M'] = 2
    wavelet_operators = {}
    for wave_num in range(0, scattering_options['M']+1):
        wavelet_operators[wave_num]= lambda xbox: wavelet_transform_1d(xbox, filters[wave_num], scattering_options)
    return wavelet_operators

if __name__ == "__main__":
    filt_opt = {}
    scat_opt = {}
    scat_opt['M'] =2
    filt_opt['filter_type'] = 'morlet_1d'
    filt_opt['Q'] = [8, 1]
    filt_opt['B'] = []
    filt_opt['xi_psi'] = []
    filt_opt['sigma_psi'] = []
    filt_opt['phi_bw_multiplier'] = []
    filt_opt['sigma_phi'] = []
    filt_opt['J'] = [57, 12]
    filt_opt['P'] = []

    filters = factory_generator(65536, filt_opt, scat_opt)
    print 'done'
