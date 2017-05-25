import numpy as np
from math import *
import scipy

__all__ = ['wavelet_transform_1d']


def wavelet_transform_1d(layer, filters, scattering_option):
    layer_phi = {}
    layer_psi = {}
    if not 'path_margin' in scattering_option:
        scattering_option['path_margin'] = 0
    psi_xi, psi_bw, phi_bw = morlet_freq_1d(filters['meta'])
    if not 'bandwidth' in layer['meta']:
        layer['meta']['bandwidth'] = list()
        current_bw_list = list()
        current_bw_list.append(2.0*np.pi)
        layer['meta']['bandwidth'].append(current_bw_list)
    if not 'resolution' in layer['meta']:
        layer['meta']['resolution'] = list()
        current_res_list = list()
        current_res_list.append(0)
        layer['meta']['resolution'].append(current_res_list)
    layer_phi['signal'] = list(list())
    layer_phi['meta'] = {}
    layer_phi['meta']['bandwidth'] = list()
    layer_phi['meta']['resolution'] = list()
    layer_phi['meta']['J'] = []
    layer_psi['signal'] = list(list())
    layer_psi['meta'] = {}
    layer_psi['meta']['bandwidth'] = list()
    layer_psi['meta']['resolution'] = list()
    layer_psi['meta']['J'] = list()
    counter = 0
    generate_all = 1
    psi_bandwidth = list()
    psi_resolution = list()
    psi_meta_j = list()
    for p1 in range(0, len(layer['signal'])):
        current_bandwidth = layer['meta']['bandwidth'][0][p1] * np.power(2.0, scattering_option['path_margin'])
        psi_mask_req = generate_all & (current_bandwidth > psi_xi)
        scattering_option['x_resolution'] = layer['meta']['resolution'][0][p1]
        scattering_option['psi_mask'] = psi_mask_req
        x_phi, x_psi, meta_phi, meta_psi = wavelet1d(layer['signal'][p1], filters, scattering_option)
        layer_phi['signal'].append(x_phi)
        if not isinstance(layer['meta']['J'][0], float):
            layer_phi['meta']['J'].append(layer['meta']['J'][0][p1])
        else:
            layer_phi['meta']['J'].append(layer['meta']['J'][0])
        layer_phi['meta']['bandwidth'].append(meta_phi['bandwidth'])
        layer_phi['meta']['resolution'].append(meta_phi['resolution'])
        indices = np.arange(counter, counter + np.sum(psi_mask_req), dtype=int)
        for index, value in enumerate(x_psi):
            if psi_mask_req[index]:
                layer_psi['signal'].append(x_psi[index])
                psi_bandwidth.append(meta_psi['bandwidth'][0][index])
                psi_resolution.append(meta_psi['resolution'][0][index])
                psi_meta_j.append(meta_psi['J'][0][index])
        layer_psi['meta']['bandwidth'].append(psi_bandwidth)
        layer_psi['meta']['resolution'].append(psi_resolution)
        if not isinstance(layer['meta']['J'][0], float):
            j_multiplier = layer['meta']['J'][0][p1] * np.ones(len(indices))
        else:
            j_multiplier = layer['meta']['J'] * np.ones(len(indices))
        if j_multiplier.size is 0:
            layer_psi['meta']['J'].append(psi_meta_j)
        elif int(max(j_multiplier)) is 0 and int(min(j_multiplier)) is 0:
            layer_psi['meta']['J'].append(psi_meta_j)
        else:
            layer_psi['meta']['J'].append(np.array([j_multiplier, psi_meta_j]))
        counter += len(indices)

    return layer_phi, layer_psi


def convolution(fftx, filters, ds):
    signal_length = len(fftx)
    if filters['type'] == 'fourier_truncated':
        start = filters['start']
        coefficients = filters['coefft']
        if len(coefficients) > signal_length:
            start0 = np.mod(start - 1, filters['N']) + 1
            nCoeffts0 = len(coefficients)
            if start0 + nCoeffts0 - 1 <= filters['N']:
                rng = start0 + np.arange(0, nCoeffts0)
            else:
                rng = np.concatenate((np.arange(start0, filters['N']), np.arange(0, nCoeffts0 + start0 - filters['N'])))
            lowpass = np.zeros(len(coefficients))
            lowpass[rng < signal_length / 2] = 1.0
            lowpass[rng == signal_length / 2] = 1.0 / 2.0
            lowpass[rng == filters['N'] - signal_length / 2.0] = 0.5
            lowpass[rng > filters['N'] - signal_length / 2.0] = 1.0
            filtered_coeff = np.multiply(coefficients , lowpass)
            temp_coff = np.reshape(filtered_coeff, (signal_length, len(coefficients) / signal_length), order='F')
            coefficients = np.sum(temp_coff, 1)
        n_coeffts = len(coefficients)
        start = np.mod(start, len(fftx))
        if start + n_coeffts - 1 <= len(fftx):
            yf = np.multiply(fftx[np.arange(start, n_coeffts + start, 1, dtype=int)], coefficients)
        else:
            yf = np.multiply(fftx[np.concatenate((np.arange(start, len(fftx), 1, dtype=int),
                                                  np.arange(0, n_coeffts + start - len(fftx), 1, dtype=int)))],
                             coefficients)
        dsj = ds + np.log2(float(len((yf))) / float(signal_length))
        if dsj > 0.0:
            ysf_ds = np.reshape(
                np.sum(np.reshape(yf, [int(len(yf) / pow(2.0, dsj)), int(pow(2.0, dsj))]), 1),
                [int(len(yf) / pow(2.0, dsj))])
        elif dsj < 0.0:
            ysf_ds = np.concatenate((yf, np.zeros([(len(yf) * (pow(2.0, -dsj) - 1))])))
        else:
            ysf_ds = yf
        if filters['type'] == 'fourier_truncated':
            ysf_ds = np.roll(ysf_ds, int(filters['start'] - 1))
        y_ds = scipy.ifft(ysf_ds) / pow(2.0, ds / 2.0)

    return y_ds


def wavelet1d(X, filters, scatteropt):
    localscatteropt = scatteropt
    if not 'oversampling' in scatteropt:
        localscatteropt['oversampling'] = 1
    if not 'x_resolution' in scatteropt:
        localscatteropt['x_resolution'] = 0
    if not 'psi_mask' in scatteropt:
        localscatteropt['psi_mask'] = []
        localscatteropt['psi_mask'] = np.ones(len(filters['psi']['filter']), dtype=bool)
    temp, psi_bw, phi_bw = morlet_freq_1d(filters['meta'])
    j0 = localscatteropt['x_resolution']
    n_padded = filters['meta']['size_filter'] / np.power(2.0, j0)
    n_original = len(X)
    x = X
    x = np.pad(x, (0, int(n_padded - n_original)), filters['meta']['boundary'])
    xf = scipy.fft(x)
    ds = np.round(np.log2(2 * np.pi / phi_bw)) - j0 - scatteropt['oversampling']
    ds = np.max(ds, 0)
    x_phi = np.real(convolution(xf, filters['phi']['filter'], ds))
    x_phi = unpad_signal(x_phi, ds, n_original, 0)
    meta_phi = dict(J=-1, bandwidth=phi_bw, resolution=j0 + ds)
    x_psi = range(len(localscatteropt['psi_mask']))
    meta_psi = dict(J=-1.0 * np.ones([1, len(filters['psi']['filter'])]),
                    bandwidth=-1.0 * np.ones([1, len(filters['psi']['filter'])]),
                    resolution=-1.0 * np.ones([1, len(filters['psi']['filter'])]))
    for mask_count in range(0, len(localscatteropt['psi_mask'])):
        if localscatteropt['psi_mask'][mask_count]:
            ds = np.round(np.log2(2.0 * pi / psi_bw[mask_count] / 2.0)) - j0 - localscatteropt['oversampling']
            ds = np.max(ds, 0)
            x_psi[mask_count] = convolution(xf, filters['psi']['filter'][mask_count], ds)
            x_psi[mask_count] = unpad_signal(x_psi[mask_count], ds, n_original, 0)
            meta_psi['J'][:, mask_count] = mask_count
            meta_psi['bandwidth'][:, mask_count] = psi_bw[mask_count]
            meta_psi['resolution'][:, mask_count] = j0 + ds
    return x_phi, x_psi, meta_phi, meta_psi


def unpad_signal(x, res, target_size, center):
    padded_size = len(x)
    offset = 0.0
    if center:
        offset = (padded_size * pow(2.0, res - target_size)) / 2.0
    offset_ds = floor(offset / pow(2.0, res))
    target_size_ds = 1 + floor((target_size - 1) / pow(2.0, res))
    xc = x[int(offset_ds) + np.arange(0, int(target_size_ds), 1, dtype=int)]
    return xc


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
