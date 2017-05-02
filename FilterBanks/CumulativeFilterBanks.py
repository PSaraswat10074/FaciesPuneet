__all__ = ['filters_bank']
import numpy as np
import scipy.fftpack as fft
from math import *

def cumulative_filter_banks(n, filt_options):
    parameter_field ={'filter_type', 'Q', 'B', 'xi_psi', 'sigma_psi', 'phi_bw_multiplier', 'sigma_phi', 'J', 'P',
                      'spline_order', 'filter_format'}
    filters = {}
    " default parameters"
    "filt_options['filter_type']='morlet_1d'"
    "filt_options['Q'] = []"
    "filt_options['B'] = []"
    "filt_options['xi_psi'] = []"
    "filt_options['sigma_psi'] = []"
    "filt_options['phi_bw_multiplier'] = []"
    "filt_options['sigma_phi'] = []"
    "filt_options['J'] = []"
    "filt_options['P'] = []"
    "filt_options['spline_order'] = []"
    "filt_options['filter_format'] = 'fourier_truncated'"
    bank_count = len(filt_options)
    for k in range(0, bank_count, 1):
        current_option= filt_options[k]
        if filt_options['filter_type']== 'morlet_1d' or filt_options['filter_type']== 'garbor_1d':
            filters[k]= morlet_filter_bank_1d(n, current_option)
        else:
            filters={}

    return filters
def morlet_filter_bank_1d(n, filter_options):
    filters={}
    filters['meta']= filter_options
    psi_ampl=1.0
    sigN= 2*n
    N= pow(2, ceil(log(sigN,2)))
    filters['meta']['size_filter']= N
    filters['psi']['filter']= {}
    filters['phi']=[]
    psi_center, psi_bw, phi_bw= morlet_freq_1d(filters['meta'])
    return filters

def morlet_freq_1d(filt_opt):
    sigma0= 2.0/sqrt(3.0)
    xi_psi= filt_opt['xi_psi']* pow(2.0,(0:-1:1- filt_opt['J'])/filt_opt['Q'])