import numpy as np
from math import *

__all__ = ['wavelet_transform_1d']


def wavelet_transform_1d(layer, filters, scattering_option):
    layer_phi = {}
    layer_psi = {}
    if not 'path_margin' in scattering_option:
        scattering_option['path_margin'] = 0
    psi_xi, psi_bw, phi_bw = morlet_freq_1d(filters['meta'])
    if not 'bandwidth' in layer['meta']:
        layer['meta']['bandwidth'] = 2 * np.pi
    if not 'resolution' in layer['meta']:
        layer['meta']['resolution'] = 0
    layer_phi['signal'] = {}
    layer_phi['meta'] = {}
    layer_phi['meta']['bandwidth'] = []
    layer_phi['meta']['resolution'] = []
    layer_phi['meta']['J'] = np.zeros(len(layer['meta']['J']))
    layer_psi['signal'] = {}
    layer_psi['meta'] = {}
    layer_psi['meta']['bandwidth'] = []
    layer_psi['meta']['resolution'] = []
    layer_psi['meta']['J'] = np.zeros(len(layer['meta']['J']))
    counter = 0
    for param in range(0, len(layer['signal'])):
        current_bandwidth = layer['meta']['bandwidth'][param] * np.power(2.0, scattering_option['path_margin'])
        psi_mask_req = True & (current_bandwidth > psi_xi)
        scattering_option['x']['resolution'] = layer['meta']['resolution'][param]
        scattering_option['psi_mask'] = psi_mask_req
        x_phi, x_psi, meta_phi, meta_psi = wavelet1d(layer['signal'][param], filters, scattering_option)
        layer_phi['signal'][param] = x_phi
        layer_phi['meta'][param] = layer['meta'][param]
        layer_phi['meta']['bandwidth'][param] = []
        layer_phi['meta']['bandwidth'][param] = meta_phi['bandwidth']
        layer_phi['meta']['resolution'][param] = []
        layer_phi['meta']['resolution'][param] = meta_phi['resolution']
        indices = np.arange(counter, counter + np.sum(psi_mask_req))
        layer_psi['signal'][param] = x_psi[psi_mask_req]
        layer_psi['meta'][param] = layer['meta'][param]
        layer_psi['meta']['bandwidth'][param] = []
        layer_psi['meta']['bandwidth'][param] = meta_psi['bandwidth']
        layer_psi['meta']['resolution'][param] = []
        layer_psi['meta']['resolution'][param] = meta_psi['resolution']
        layer_psi['meta']['J'][:, indices] = np.append(np.array([layer['meta']['J'][:, param]]) * np.ones(len(indices)),
                                                       meta_psi['J'])
        counter += len(indices)

    return layer_phi, layer_psi


def wavelet1d(X, filters, scatteropt):
    localscatteropt= scatteropt
    if not 'oversampling' in scatteropt:
        localscatteropt['oversampling'] = 1
    if not 'x_resolution' in scatteropt:
        localscatteropt['x_resolution'] = 0
    if scatteropt['psi_mask']:
        localscatteropt['psi_mask']=[]
        localscatteropt['psi_mask'] = np.ones(len(filters['psi']['filter']), dtype=bool)
    temp, psi_bw ,phi_bw = morlet_freq_1d(filters['meta'])
    j0 = localscatteropt['resolution']
    n_padded = filters['meta']['size_filter']/np.power(2.0, j0)
    x = np.reshape(X,[])

    return


def morlet_freq_1d(filt_opt):
    sigma0 = 2.0 / sqrt(3.0)
    xi_psi = list(filt_opt['xi_psi'] * np.power(2.0, np.arange(0.0, 1 - filt_opt['J'] - 1, -1.0) / filt_opt['Q']))
    sigma_psi = list(filt_opt['sigma_psi'] * np.power(2.0, np.arange(0.0, filt_opt['J'], 1.0) / filt_opt['Q']))
    step = np.pi * np.power(2.0, -float(filt_opt['J']) / float(filt_opt['Q'])) * (
        1.0 - (0.25 * sigma0 / filt_opt['sigma_phi']) * np.power(2.0, 1.0 / filt_opt['Q'])) / float(filt_opt['P'])
    for i in range(int(filt_opt['J']), int(filt_opt['J'] + filt_opt['P']), 1):
        xi_psi.append(filt_opt['xi_psi'] *
                      np.power(2.0, -1 * int((filt_opt['J'] + 1.0) / filt_opt['Q'])) - step * (i - filt_opt['J'] + 1.0))
    for i in range(int(filt_opt['J'] + 1), int(filt_opt['J'] + 2 + filt_opt['P']), 1):
        sigma_psi.append(filt_opt['sigma_psi'] * np.power(2.0, (filt_opt['J'] - 1) / filt_opt['Q']))
    sigma_phi = filt_opt['sigma_phi'] * np.power(2.0, (filt_opt['J'] - 1) / filt_opt['Q'])
    bw_psi = ((np.pi / 2.0) * sigma0 / np.array(sigma_psi))
    if not filt_opt['phi_dirac']:
        bw_phi = (np.pi / 2.0) * sigma0 / sigma_phi
    else:
        bw_phi = 2.0 * np.pi
    return xi_psi, bw_psi, bw_phi
