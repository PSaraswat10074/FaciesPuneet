import numpy as np
from math import *

__all__ = ['facies_scattering_transform']


def facies_scattering_transform(signal, wavelet_operators):
    cov_mod = {'signal': signal, 'meta': {}}
    cov_mod['meta']['J'] = np.zeros([1])
    cov_mod['meta']['Q'] = np.zeros([1])
    cov_mod['meta']['resolution'] = 0
    cov_mod_list = list()
    cov_mod_list.append(cov_mod)
    scatter_signal = list()
    for scat_number in range(0, len(wavelet_operators)):
        if scat_number < len(wavelet_operators) - 1:
            layer_phi, layer_psi = wavelet_operators[scat_number](cov_mod_list[scat_number])
            scatter_signal.append(layer_phi)
            cov_mod_list.append(layer_modulus(layer_psi))
        else:
            layer_phi, layer_psi = wavelet_operators[scat_number](cov_mod_list[scat_number])
            scatter_signal.append(layer_phi)
    return scatter_signal, cov_mod_list


def layer_modulus(layer_psi):
    modulated_layer = {'signal': layer_psi['signal']}
    for signal_counter in range(0, len(modulated_layer['signal'])):
        modulated_layer['signal'][signal_counter]= np.abs(modulated_layer['signal'][signal_counter])
    modulated_layer['meta']= layer_psi['meta']
    return modulated_layer
