import numpy as np
from math import *

__all__ = ['cumulative_filter_banks_1d']


def cumulative_filter_banks_1d(n, filt_options):
    filters1d = dict()
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
    if not 'filter_format' in filt_options:
        filt_options['filter_format'] = 'fourier_truncated'
    bank_count = len(filt_options['Q'])
    for k in range(0, bank_count, 1):
        current_option1d = {}
        current_option1d['filter_type'] = filt_options['filter_type']
        current_option1d['Q'] = filt_options['Q'][k]
        current_option1d['B'] = filt_options['B']
        current_option1d['xi_psi'] = filt_options['xi_psi']
        current_option1d['sigma_psi'] = filt_options['sigma_psi']
        current_option1d['phi_bw_multiplier'] = filt_options['phi_bw_multiplier']
        current_option1d['sigma_phi'] = filt_options['sigma_phi']
        current_option1d['J'] = filt_options['J'][k]
        current_option1d['P'] = filt_options['P']
        current_option1d['filter_format'] = filt_options['filter_format']
        if filt_options['filter_type'] == 'morlet_1d' or filt_options['filter_type'] == 'garbor_1d':
            filters1d[k] = morlet_filter_bank_1d(n, current_option1d)
        else:
            filters1d[k] = None

    return filters1d


def morlet_filter_bank_1d(n, filter_options):
    filters = {}
    sigma0 = 2.0 / sqrt(3.0)
    "fill in default params"
    if filter_options['B'] == []:
        filter_options['B'] = filter_options['Q']
    if filter_options['xi_psi'] == []:
        filter_options['xi_psi'] = 0.5 * ((np.power(2.0, -1.0 / filter_options['Q']) + 1.0) * np.pi)
    if filter_options['sigma_psi'] == []:
        filter_options['sigma_psi'] = 0.5 * sigma0 / (1.0 - np.power(2.0, -1.0 / filter_options['B']))
    if filter_options['phi_bw_multiplier'] == []:
        filter_options['phi_bw_multiplier'] = 1 + (filter_options['Q'] == 1)
    if filter_options['sigma_phi'] == []:
        filter_options['sigma_phi'] = filter_options['sigma_psi'] / filter_options['phi_bw_multiplier']
    if filter_options['P'] == []:
        filter_options['P'] = round((np.power(2.0, -1.0 / filter_options['Q']) - (0.25 * sigma0) /
                                     filter_options['sigma_phi']) / (1.0 - np.power(2.0, -1.0 / filter_options['Q'])))
    if not 'boundary' in filter_options:
        filter_options['boundary'] = 'symm'
    if not 'phi_dirac' in filter_options:
        filter_options['phi_dirac'] = 0

    filters['meta'] = filter_options
    psi_ampl = 1.0
    sigN = n
    if filters['meta']['boundary'] == 'symm':
        sigN = 2 * n
    N = np.power(2, ceil(log(sigN, 2)))
    filters['meta']['size_filter'] = N
    filters['psi'] = {}
    filters['phi'] = {}
    filters['psi']['filter'] = {}
    filters['psi']['meta'] = {}
    filters['psi']['meta']['k'] = []
    filters['phi']['filter'] = {}
    filters['phi']['meta'] = {}
    filters['phi']['meta']['k'] = []
    psi_center, psi_bw, phi_bw = morlet_freq_1d(filters['meta'])
    psi_sigma = sigma0 * np.pi / 2.0 / psi_bw
    phi_sigma = sigma0 * np.pi / 2.0 / phi_bw
    do_gabor = filter_options['filter_type'] == 'gabor_1d'
    s = np.zeros(N)
    for j1 in range(0, int(filter_options['J']), 1):
        temp = gabor(N, psi_center[j1], psi_sigma[j1])
        if not do_gabor:
            temp = morlet_transformation(temp, psi_sigma[j1])
        s += np.power(abs(temp), 2)
    psi_ampl = sqrt(2.0 / max(s))
    "Apply normalization"
    for j1 in range(0, int(filter_options['J'] + filter_options['P']), 1):
        temp = gabor(N, psi_center[j1], psi_sigma[j1])
        if not do_gabor:
            temp = morlet_transformation(temp, psi_sigma[j1])
        filters['psi']['filter'][j1] = optimize_filter(psi_ampl * temp, 0, filter_options)
        filters['psi']['meta']['k'].append(j1)
    if not filter_options['phi_dirac']:
        filters['phi']['filter'] = gabor(N, 0, phi_sigma)
    else:
        filters['phi']['filter'] = np.ones([N, 1])
    filters['phi']['filter'] = optimize_filter(filters['phi']['filter'], 1, filter_options)
    filters['phi']['meta']['k'].append(filter_options['J'] + filter_options['P'])

    return filters


def optimize_filter(filter_f, lowpass, options):
    options['truncate_threshold'] = 1e-3
    if options['filter_format'] is None:
        options['filter_format'] = 'fourier_multires'
    if options['filter_format'] == 'fourier':
        filters = filter_f
    elif options['filter_format'] == 'fourier_truncated':
        filters = truncate_filter(filter_f, options['truncate_threshold'])
    return filters


def truncate_filter(filter_f, threshold):
    N = len(filter_f)
    filter = {}
    filter['type'] = 'fourier_truncated'
    filter['N'] = N
    filter['recenter'] = 1
    ind_max = np.argmax(filter_f)
    "ind_max, temp = max(enumerate(filter_f), key=np.operator.itemgetter(1))"
    filter_f = np.roll(filter_f, N / 2 - ind_max - 1)
    filtabs = np.abs(filter_f)
    maxfiltabs = np.max(filtabs) * threshold
    ind_list = [n for n, i in enumerate(filtabs) if i > maxfiltabs]
    ind1 = ind_list[0]
    ind2 = ind_list[-1]
    leng = ind2 - ind1 + 1
    leng = filter['N'] / np.power(2.0, floor(log(filter['N'] / leng, 2)))
    ind1 = round(round((ind1 + ind2) / 2.0) - leng / 2.0)
    ind2 = ind1 + leng - 1
    filtered_indices = np.mod(np.arange(int(ind1), int(ind2) + 1) - 1, int(filter['N'])) + 1
    if filtered_indices[0] >= N - 1:
        filtered_indices[0] = 0
    if filtered_indices[-1] >= N:
        filtered_indices[-1] = N - 1
    filtered_indices[filtered_indices > N - 1] = N - 1
    filter_f = filter_f[filtered_indices]
    filter['coefft'] = filter_f
    filter['start'] = ind1 - (N / 2.0 - ind_max - 1)
    return filter


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


def morlet_transformation(f, sigma):
    f0 = f[0]
    remove_gabor_comp = gabor(len(f), 0, sigma)
    temp_f = f - f0 * remove_gabor_comp
    return temp_f


def gabor(N, xi, sigma):
    extent = 1
    sigma = 1.0 / sigma
    f = np.zeros(N)
    for k in range(-extent, extent + 2, 1):
        summer = np.exp(-np.power((np.arange(0, N) * 1.0 - 1.0 * k * N) / (1.0 * N) * 2.0 * np.pi - xi, 2.0) / (
            2.0 * np.power(sigma, 2.0)))
        f += summer
    return f


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


if __name__ == "__main__":
    current_option = {}
    current_option['filter_type'] = 'morlet_1d'
    current_option['Q'] = [8, 1]
    current_option['B'] = []
    current_option['xi_psi'] = []
    current_option['sigma_psi'] = []
    current_option['phi_bw_multiplier'] = []
    current_option['sigma_phi'] = []
    current_option['J'] = [57, 12]
    current_option['P'] = []
    filters = cumulative_filter_banks_1d(65536, current_option)
    print 'done'
