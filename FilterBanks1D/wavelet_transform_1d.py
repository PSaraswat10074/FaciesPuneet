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
        layer['meta']['bandwidth'] = 2 * np.pi
    if not 'resolution' in layer['meta']:
        layer['meta']['resolution'] = 0
    layer_phi['signal'] = {}
    layer_phi['meta'] = {}
    layer_phi['meta']['bandwidth'] = []
    layer_phi['meta']['resolution'] = []
    layer_phi['meta']['J'] = []
    layer_psi['signal'] = {}
    layer_psi['meta'] = {}
    layer_psi['meta']['bandwidth'] = []
    layer_psi['meta']['resolution'] = []
    layer_psi['meta']['J'] = []
    counter = 0
    generate_all = 1
    for param in range(0, len(layer['signal'])):
        current_bandwidth = layer['meta']['bandwidth'][param] * np.power(2.0, scattering_option['path_margin'])
        psi_mask_req = generate_all & (current_bandwidth > psi_xi)
        scattering_option['x_resolution'] = layer['meta']['resolution'][param]
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


def convolution(fftx, filters, ds):
    signal_length = len(fftx)
    if filters['type'] == 'fourier_truncated':
        start = filters['start']
        coefficients = filters['coeff']
        if len(coefficients) > signal_length:
            start0 = np.mod(start - 1, filters['N']) + 1
            nCoeffts0 = len(coefficients)
            if start0 + nCoeffts0 - 1 <= filters['N']:
                rng = start0 + np.arange(0, nCoeffts0)
            else:
                rng = [np.arange(start0, filters['N']), np.arange(0, nCoeffts0 + start0 - filters['N'] - 1)]
            lowpass = np.zeros(len(coefficients))
            lowpass[rng < signal_length / 2 + 1] = 1.0
            lowpass[rng == signal_length / 2 + 1] = 1.0 / 2.0
            lowpass[rng == filters['N'] - signal_length / 2.0 + 1] = 0.5
            lowpass[rng > filters['N'] - signal_length / 2.0 + 1] = 1.0
            coefficients = np.sum(coefficients * lowpass, [signal_length, len(coefficients) / signal_length], 2)
        nCoeffts = len(coefficients)
        j = np.log2(nCoeffts / signal_length)
        start = np.mod(start - 1, len(fftx)) + 1
        if start + nCoeffts - 1 <= len(fftx):
            yf = np.multiply(fftx[start: nCoeffts + start - 1] * coefficients)
        else:
            yf = np.multiply(fftx[[np.arange(start, len(fftx), 1), np.arange(0, nCoeffts + start - len(fftx))]],
                             coefficients)
        dsj = ds + np.log2(len(yf, 1) / signal_length)
        if dsj > 0.0:
            ysf_ds = np.reshape(
                np.sum(np.reshape(yf, [int(len(yf, 1) / pow(2.0, dsj)), int(pow(2.0, dsj)), len(yf, 2)]), 2),
                [int(len(yf, 1) / pow(2.0, dsj))])
        elif dsj < 0.0:
            yf = [yf, np.zeros([(len(yf, 1) * (pow(2.0, -dsj) - 1)), len(yf, 2)])]
        if filters['type'] == 'fourier_truncated':
            ysf_ds = np.roll(ysf_ds, filters['start'] - 1)
        y_ds = scipy.ifft(ysf_ds) / pow(2.0, ds / 2.0)

    return y_ds


def wavelet1d(X, filters, scatteropt):
    localscatteropt = scatteropt
    if not 'oversampling' in scatteropt:
        localscatteropt['oversampling'] = 1
    if not 'x_resolution' in scatteropt:
        localscatteropt['x_resolution'] = 0
    if scatteropt['psi_mask']:
        localscatteropt['psi_mask'] = []
        localscatteropt['psi_mask'] = np.ones(len(filters['psi']['filter']), dtype=bool)
    temp, psi_bw, phi_bw = morlet_freq_1d(filters['meta'])
    j0 = localscatteropt['resolution']
    n_padded = filters['meta']['size_filter'] / np.power(2.0, j0)
    x = X
    x = pad_signal(x, n_padded, filters['meta']['boundary'])
    xf = scipy.fft(x)
    ds = np.round(np.log2(2 * np.pi / phi_bw)) - j0 - scatteropt['oversampling']
    ds = np.max(ds, 0)
    x_phi = np.real(convolution(xf, filters['phi']['filter'], ds))
    x_phi = unpad_signal(x_phi, ds, len(X), 0)
    meta_phi = dict(J=-1, bandwidth=phi_bw, resolution=j0 + ds)
    x_psi = range(len(localscatteropt['psi_mask']))
    meta_psi = dict(J=-1.0 * np.ones([1, len(filters['psi']['filter'])]),
                    bandwidth=-1.0 * np.ones([1, len(filters['psi']['filter'])]),
                    resolution=-1.0 * np.ones([1, len(filters['psi']['filter'])]))
    for mask_count in range(0, len(localscatteropt['psi_mask'])):
        ds = np.round(np.log2(2.0 * pi / psi_bw[mask_count] / 2.0)) - j0 - localscatteropt['oversampling']
        x_psi[mask_count] = convolution(xf, filters['psi']['filter'][mask_count], ds)
        x_psi[mask_count] = unpad_signal(x_psi[mask_count], ds, len(X), 0)
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
    x = x[offset_ds + np.arange(0, target_size_ds)]
    return x


def pad_signal(x, npad, boundary, center):
    orig_sz = len(x)
    if center:
        margins = floor(npad - orig_sz / 2.0)
    has_imag = np.iscomplex(x)
    n_original = orig_sz
    if boundary == 'symmetrical':
        index0 = [np.arange(0, n_original, 1), np.arange(n_original - 1, 0, -1)]
        conjugate_index = [np.zeros(n_original), np.ones(n_original)]
    elif boundary == 'periodic':
        index0 = np.arange(0, n_original, 1)
        conjugate_index = np.zeros(n_original)
    elif boundary == 'zero':
        index0 = np.arange(0, n_original, 1)
        conjugate_index = np.zeros(n_original, 1)
    if not boundary == 'zero':
        index_true = np.zeros(npad)
        conjugate_index_true = np.zeros([1, npad])
        index_true[0:n_original] = np.arange(0, n_original)
        conjugate_index_true[0: n_original] = np.zeros(n_original)
        source = np.mod(np.arange(n_original, n_original + int(np.floor((npad - n_original) / 2.0))), len(index0)) + 1
        destination = np.arange(n_original, n_original + int(np.floor((npad - n_original) / 2.0)))
        index_true[destination] = index0[source]
        conjugate_index_true[destination] = conjugate_index[source]
        source = np.mod(len(index0) - np.arange(0, int(np.ceil((npad - n_original) / 2.0)), 1), len(index0)) + 1
        destination = np.arange(npad, n_original + int(np.floor((npad - n_original) / 2.0))) + 1
        index_true[destination] = index0[source]
        conjugate_index_true[destination] = conjugate_index[source]
    else:
        index_true = np.arange(0, n_original)
        conjugate_index_true = np.zeros(n_original)
    x = x[index_true]
    if has_imag:
        x = x - 2.0 * 1.0j * (np.imag(x) * conjugate_index_true)
    if boundary == 'zero':
        x[n_original: npad - 1] = 0.0
    if center:
        x = np.roll(x, margins)
    return x


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
