__all__ = ['filters_bank']
from math import *

import numpy as np


def cumulative_filter_banks1D(n, filt_options):
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
        current_option = {}
        current_option['filter_type'] = filt_options['filter_type']
        current_option['Q'] = filt_options['Q'][k]
        current_option['B'] = filt_options['B']
        current_option['xi_psi'] = filt_options['xi_psi']
        current_option['sigma_psi'] = filt_options['sigma_psi']
        current_option['phi_bw_multiplier'] = filt_options['phi_bw_multiplier']
        current_option['sigma_phi'] = filt_options['sigma_phi']
        current_option['J'] = filt_options['J'][k]
        current_option['P'] = filt_options['P']
        current_option['spline_order'] = filt_options['spline_order']
        if filt_options['filter_type'] == 'morlet_1d' or filt_options['filter_type'] == 'garbor_1d':
            filters[k] = morlet_filter_bank_1d(n, current_option)
        else:
            filters = {}

    return filters


def morlet_filter_bank_1d(n, filter_options):
    filters = {}
    sigma0 = 2.0 / sqrt(3.0)
    "fill in default params"
    if filter_options['B'] is None:
        filter_options['B'] = filter_options['Q']
    if filter_options['xi_psi'] == None:
        filter_options['xi_psi'] = 0.5 * ((pow(2.0, -1.0 / filter_options['Q']) + 1.0) * np.pi)
    if filter_options['sigma_psi'] == None:
        filter_options['sigma_psi'] = 0.5 * sigma0 / (1.0 - pow(2.0, -1.0 / filter_options['B']))
    if filter_options['phi_bw_multiplier'] == None:
        filter_options['phi_bw_multiplier'] = 1 + (filter_options['Q'] == 1)
    if filter_options['sigma_phi'] == None: \
            filter_options['sigma_phi'] = filter_options['sigma_psi'] / filter_options['phi_bw_multiplier']
    if filter_options['P'] == None:
        filter_options['P'] = round((pow(2.0, -1 / filter_options['Q']) - (1.0 / (4.0 * sigma0)) /
                                     filter_options['sigma_phi']) / (1.0 - pow(2.0, -1.0 / filter_options['Q'])))
    if filter_options['boundary'] is None:
        filter_options['boundary'] = 'symm'
    if filter_options['phi_dirac'] is None:
        filter_options['phi_dirac'] = 0

    filters['meta'] = filter_options
    psi_ampl = 1.0
    sigN = n
    if filters['meta']['boundary'] == 'symm':
        sigN = 2 * n
    N = pow(2, ceil(log(sigN, 2)))
    filters['meta']['size_filter'] = N
    filters['psi']['filter'] = {}
    filters['phi'] = {}
    filters['psi']['meta'] = {}
    filters['psi']['meta']['k'] = []
    psi_center, psi_bw, phi_bw = morlet_freq_1d(filters['meta'])
    psi_sigma = sigma0 * np.pi / 2.0 / psi_bw
    phi_sigma = sigma0 * np.pi / 2.0 / phi_bw
    do_gabor = filter_options['filter_type'] == 'gabor_1d'
    S = np.zeros(N, 1)
    for j1 in range(0, filter_options['J'] - 1):
        temp = gabor(N, psi_center(j1), psi_sigma(j1))
        if not do_gabor:
            temp = morlet_transformation(temp, psi_sigma[j1])
        S = S + pow(abs(temp), 2)
    psi_ampl = sqrt(2.0 / max(S))
    "Apply normalization"
    for j1 in range(0, filter_options['J'] + filter_options['P'] - 1, 1):
        temp = gabor(N, psi_center[j1], psi_sigma[j1])
        if not do_gabor:
            temp = morlet_transformation(temp, psi_sigma[j1])
        filters['psi']['filters'][j1] = optimize_filter(psi_ampl * temp, 0, filter_options)
        filters['psi']['meta']['k'][j1, 0] = j1
    if  not filter_options['phi_dirac']:
        filters['phi']['filter']= gabor(N, 0, phi_sigma)
    else:
        filters['phi']['filter']= np.ones([N,1])
    filters['phi']['filter']= optimize_filter(filters['phi']['filter'],1, filter_options)
    filters['phi']['meta']['k'][1,1]= filter_options['J']+ filter_options['P']

    return filters


def optimize_filter(filter_f, lowpass, options):
    options['truncate_threshold'] = 1e-3
    if options['filter_format'] is None:
        options['filter_format'] = 'fourier_multires'
    if options['filter_format'] == 'fourier':
        filters = filter_f
    elif options['filter_format'] == 'fourier_truncated':
        filters = truncate_filter(filter_f, options['truncate_threshold'], lowpass)
    return filters


def truncate_filter(filter_f, threshold, lowpass):
    N = len(filter_f)
    filter = {}
    filter['type'] = 'fourier_truncated'
    filter['N'] = N
    filter['recenter'] = 1
    ind_max, temp = max(enumerate(filter_f), key=np.operator.itemgetter(1))
    filter_f = np.roll(filter_f, N / 2 - ind_max)
    ind1 = indices(abs(filter_f), lambda x: x > max(abs(filter_f)) * threshold)[0]
    ind2 = indices(abs(filter_f), lambda x: x > max(abs(filter_f)) * threshold)[-1]
    leng = ind2 - ind1 + 1
    leng = filter['N'] / pow(2.0, floor(log(filter['N'] / leng, 2)))
    ind1 = round(round((ind1 + ind2) / 2.0) - leng / 2.0)
    ind2 = ind1 + leng - 1
    filter_f = filter_f[modf(np.arange(ind1, ind2) - 1, filter['N']) + 1]
    filter['coefft'] = filter_f
    filter['start'] = ind1 - (N / 2.0 - ind2)
    return filter


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


def morlet_transformation(f, sigma):
    f0 = f[0]
    f = f - f0 * gabor(len(f), 0, sigma)
    return f


def gabor(N, xi, sigma):
    extent = 1.0
    sigma = 1.0 / sigma
    f = np.zeros(N, 1)
    for k in range(-extent, extent + 1, 1):
        f = f + exp(-(((np.arange(0, N - 1) - k * N))) / N * 2.0 * np.pi - xi)
    return f


def morlet_freq_1d(filt_opt):
    sigma0 = 2.0 / sqrt(3.0)
    xi_psi = filt_opt['xi_psi'] * pow(2.0, np.arange(0.0, 1 - filt_opt['J'], -1.0) / filt_opt['Q'])
    sigma_psi = filt_opt['sigma_psi'] * pow(2.0, np.arange(0.0, filt_opt['J'] - 1, 1.0) / filt_opt['Q'])
    step = np.pi * pow(2.0, -filt_opt['J'] / filt_opt['Q']) * (1.0 - (0.25 * sigma0 / filt_opt['sigma_phi']) *
                                                               pow(2.0, 1.0 / filt_opt['Q'])) / filt_opt['P']
    for i in range(filt_opt['J'], filt_opt['J'] + filt_opt['P'] - 1, 1):
        xi_psi[i] = filt_opt['xi_psi'] * pow(2.0, -(filt_opt['J'] + 1) / filt_opt['Q']) - step * (i - filt_opt['J'] + 1)
    for i in range(filt_opt['J'], filt_opt['J'] + filt_opt['P'], 1):
        sigma_psi[i] = filt_opt['sigma_psi'] * pow(2.0, (filt_opt['J'] - 1) / filt_opt['Q'])
    sigma_phi = filt_opt['sigma_phi'] * pow(2.0 * (filt_opt['J'] - 1) / filt_opt['Q'])
    bw_psi = (np.pi / 2.0) * sigma0 / sigma_psi
    if filt_opt['phi_dirac']:
        bw_phi = (np.pi / 2.0) * sigma0 / sigma_phi
    else:
        bw_phi = 2.0 * np.pi
    return xi_psi, bw_psi, bw_phi
