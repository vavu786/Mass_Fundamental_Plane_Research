# *** USAGE: python scriptname.py ***

# Using data from ALL samples
# Plotting Mass Fundamental Plane (MFP) from z ~ 0.3 -> z ~ 4.0
# Star-forming and quiescent galaxies

import sys
from tabulate import tabulate
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import math
import scipy
from scipy import optimize
from scipy.stats import bootstrap
import pandas as pd
import astropy
from astropy.cosmology import WMAP9 as cosmo
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from time import perf_counter as pf

pd.set_option('display.colheader_justify', 'center')

ALPHA = 1.6287
BETA = 0.840
NEWTON_G = 6.67e-11

# A = 'A' of 4th eqn. in Remus et. al. 2017 pp. 3748.
# B = 'B' of 4th eqn. in Remus et. al. 2017 pp. 3748.
# a = 'A' of 6th eqn. in Remus et. al. 2017 pp. 3748.
# b = 'B' of 6th eqn. in Remus et. al. 2017 pp. 3748.
# Index 0 = Oser, 1 = Magneticum

A = (-0.57, -0.38)
B = (3.06, 1.32)
a = (0.71, 0.69)
b = (-2.83, -2.61)

# print("Oser: ")
# print(f"slope: {round(A[0]/a[0], 2)}")
# print(f"intercept: {round((B[0] - b[0]) / a[0], 2)}")

# print("Magneticum: ")
# print(f"slope: {round(A[1]/a[1], 2)}")
# print(f"intercept: {round((B[1] - b[1]) / a[1], 2)}")

conf_level = 0.68
num_samples = 1000


def get_lns_lbls(arr_axes):
    lines_labels = []

    for axis in np.reshape(arr_axes, -1):
        lines_labels.append(axis.get_legend_handles_labels())

    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    filtered_lines = []
    filtered_labels = []

    for line, label in zip(lines, labels):
        if isinstance(line, matplotlib.collections.PathCollection):
            if len(line.get_array()) != 0:
                if label not in filtered_labels:
                    filtered_lines.append(line)
                    filtered_labels.append(label)

        if isinstance(line, matplotlib.lines.Line2D):
            if len(line.get_xdata()) != 0:
                if label not in filtered_labels:
                    filtered_lines.insert(0, line)
                    filtered_labels.insert(0, label)

    return filtered_lines, filtered_labels


# Fits the MFP using either scipy.optimize.leastsq() for fixed slope, or numpy.polyfit() for a free slope
def fit_MFP(x, y):
    y_ax_func = lambda zero_pt, x: x - zero_pt
    errfunc = lambda zero_pt, x, y: abs(y_ax_func(zero_pt, x) - y) / math.sqrt(1 + ALPHA ** 2 + BETA ** 2)

    # fit_params in this case contains just an array with one element: the zero-point
    fit_params, _ = scipy.optimize.leastsq(errfunc, 4.4, (x, y))
    best_fit_line = y_ax_func(fit_params[0], x)

    return best_fit_line, fit_params


def fit_MFP_Bez_form(x, y, zspec):
    y_ax_func = lambda delta, x, zspec: x - 4.475 - (delta * np.log10(1 + zspec - 0.063))
    errfunc = lambda delta, x, y, zspec: abs(y_ax_func(delta, x, zspec) - y) / math.sqrt(1 + ALPHA ** 2 + BETA ** 2)

    # fit_params in this case contains just an array with one element: the zero-point
    fit_params, _ = scipy.optimize.leastsq(errfunc, -0.095, (x, y, zspec))
    best_fit_line = y_ax_func(fit_params[0], x, zspec)
    return best_fit_line, fit_params


def scatter(x, y, zpt):
    scat_arr = x - y - zpt
    return np.sqrt(np.mean(scat_arr ** 2))


def calc_f_dm(sigma_re, rekpc, lmass):
    lm_dyn = np.log10(5.0 * rekpc) + (2 * np.log10(sigma_re)) - np.log10(NEWTON_G)
    return lm_dyn - lmass


def num_to_label(num):
    if num == 0:
        return "SDSS"
    if num == 1:
        return "Newman+10"
    if num == 2:
        return "van Dokkum+09"
    if num == 4:
        return "van der Wel+05"
    if num == 5:
        return "Wuyts+05"
    if num == 6:
        return "van de Sande+13"
    if num == 7:
        return "Bezanson+Dec.2013"
    if num == 8:
        return "Bezanson+15"
    if num == 9:
        return "Belli+16"
    if num == 10:
        return "Forrest+22"
    if num == 11:
        return "de Graaff+21"
    else:
        return "INVALID"


def num_to_shape(num):
    if num == 0:
        return "."
    if num == 1:
        return 'H'
    if num == 2:
        return 'D'
    if num == 4:
        return 'd'
    if num == 5:
        return 'p'
    if num == 6:
        return 's'
    if num == 7:
        return '^'
    if num == 8:
        return '.'
    if num == 9:
        return 'P'
    if num == 10:
        return '*'
    if num == 11:
        return '+'
    else:
        return 'INVALID'


def main():

    start_t = pf()
    paper_nums = [0, 11, 1, 2, 4, 5, 6, 7, 8, 9, 10]

    for num in paper_nums:
        print(f"{num}: {num_to_label(num)}")

    # ______________________________DATA_____________________________________________________________

    data = np.genfromtxt("allhighz_simard.dat", names=True, dtype=None)

    # All data from Bezanson 2013 paper except SDSS; condition of massive galaxies, and not ground-based imaging
    data = data[data["highzcatnum"] != 0]
    toplot = (data['lmass_profile'] > 10.0) & (data['rekpc'] > 0.0) & (data['sigma_re'] > 0) & (data['zspec'] > 0) & (
                data['sizesource'] != 4)
    data = data[toplot]
    data_pd = pd.DataFrame(data)
    # data_pd.hist(column="zspec", bins=int((2.2-0.3)/0.1))
    data_pd["f_dm"] = calc_f_dm(data_pd["sigma_re"].to_numpy(), data_pd["rekpc"].to_numpy(),
                                data_pd["lmass"].to_numpy())
    data_useful_pd = data_pd[["highzcatnum", "zspec", "rekpc", "lmass", "sigma_re", "mustar", "f_dm"]]

    # Belli (24 galaxies) (Yellow). 1.526 < z < 2.435
    Belli = np.genfromtxt("z=1.7_Belli.dat", names=True, dtype=None, skip_header=2, skip_footer=18, encoding=None)
    Belli_pd = pd.DataFrame(Belli)
    Belli_pd["rekpc"] = Belli_pd["r_maj"].to_numpy() * np.sqrt(Belli_pd["q"].to_numpy())
    Belli_pd["highzcatnum"] = np.full(Belli_pd.shape[0], 9)
    Belli_pd["mustar"] = Belli_pd["lmass"].to_numpy() - np.log10(2 * np.pi * np.square(Belli_pd["rekpc"].to_numpy()))
    Belli_pd["f_dm"] = calc_f_dm(Belli_pd["sigma_re"].to_numpy(), Belli_pd["rekpc"].to_numpy(),
                                 Belli_pd["lmass"].to_numpy())
    Belli_useful_pd = Belli_pd[["highzcatnum", "zspec", "rekpc", "lmass", "sigma_re", "mustar", "f_dm"]]

    # Forrest (14 galaxies) (Magenta) 
    Forrest = np.genfromtxt("z=3.5_Forrest.dat", names=True, dtype=None, skip_header=1, skip_footer=1)
    Forrest_pd = pd.DataFrame(Forrest)
    Forrest_pd["highzcatnum"] = np.full(Forrest_pd.shape[0], 10)
    Forrest_pd["mustar"] = Forrest_pd["lmass"].to_numpy() - np.log10(
        2 * np.pi * np.square(Forrest_pd["rekpc"].to_numpy()))
    Forrest_pd["f_dm"] = calc_f_dm(Forrest_pd["sigma_re"].to_numpy(), Forrest_pd["rekpc"].to_numpy(),
                                   Forrest_pd["lmass"].to_numpy())
    Forrest_useful_pd = Forrest_pd[["highzcatnum", "zspec", "rekpc", "lmass", "sigma_re", "mustar", "f_dm"]]

    # LEGA-C data (1419 galaxies)
    legac_table = astropy.table.Table.read("legac_fp_selection.fits", format="fits")
    legac_pd = legac_table.to_pandas()
    legac_pd["highzcatnum"] = np.full(legac_pd.shape[0], 11)
    legac_pd["rekpc"] = 10 ** legac_pd["log_rec_kpc"].to_numpy()
    legac_pd["sigma_re"] = 10 ** legac_pd["log_sigma_stars"].to_numpy()
    legac_pd.rename(columns={"z_spec": "zspec", "log_mstar": "lmass", "log_Sigma_star": "mustar"}, inplace=True)
    legac_pd["f_dm"] = calc_f_dm(legac_pd["sigma_re"].to_numpy(), legac_pd["rekpc"].to_numpy(),
                                 legac_pd["lmass"].to_numpy())
    legac_useful_pd = legac_pd[["highzcatnum", "zspec", "rekpc", "lmass", "sigma_re", "mustar", "f_dm"]]

    # SDSS (18,573 galaxies)
    sdss_table = astropy.table.Table.read("sdss_fp_selection_magphys_pymorph.fits", format="fits")
    sdss_pd = sdss_table.to_pandas()
    sdss_pd["highzcatnum"] = np.full(sdss_pd.shape[0], 0)
    sdss_pd["rekpc"] = 10 ** sdss_pd["log_rec_kpc"].to_numpy()
    sdss_pd["sigma_re"] = 10 ** sdss_pd["log_sigma_re"].to_numpy()
    sdss_pd.rename(columns={"z": "zspec", "log_mstar": "lmass", "log_Sigma_star": "mustar"}, inplace=True)
    sdss_pd["f_dm"] = calc_f_dm(sdss_pd["sigma_re"].to_numpy(), sdss_pd["rekpc"].to_numpy(),
                                sdss_pd["lmass"].to_numpy())
    sdss_useful_pd = sdss_pd[["highzcatnum", "zspec", "rekpc", "lmass", "sigma_re", "mustar", "f_dm"]]

    all_data_pd = pd.concat([data_useful_pd, Belli_useful_pd, Forrest_useful_pd, legac_useful_pd, sdss_useful_pd])
    all_data_pd.set_index(pd.Index(range(all_data_pd.shape[0])), inplace=True)

    print(all_data_pd)
    z_dist = all_data_pd.hist(column="zspec", bins=int((4.1 - 0.3) / 0.1))
    plt.xlabel("Redshift")
    plt.ylabel("Number of galaxies")

    # _______________________________________________________________________________________________
    mfp_fig, mfp_ax = plt.subplots(2, 3)
    mfp_fig_allz, mfp_ax_allz = plt.subplots()
    RM_sigmaM_fig, RM_sigmaM_ax = plt.subplots(2, 5)
    RSigma_fig, RSigma_ax = plt.subplots(2, 3)
    fdm_M_fig, fdm_M_ax = plt.subplots(2, 3)

    zero_pts = []
    conf_ints = []
    std_devs = []
    scatters = []

    # Prepare all axes
    all_data_pd["lradius"] = np.log10(all_data_pd["rekpc"].to_numpy())
    all_data_pd["mfp"] = ALPHA * np.log10(all_data_pd["sigma_re"].to_numpy()) - BETA * all_data_pd["mustar"]
    all_data_pd["lsigma_re"] = np.log10(all_data_pd["sigma_re"].to_numpy())

    min_x = np.min(all_data_pd["lradius"].to_numpy())
    max_x = np.max(all_data_pd["lradius"].to_numpy())

    min_y = np.min(all_data_pd["mfp"].to_numpy())
    max_y = np.max(all_data_pd["mfp"].to_numpy())

    min_lmass = np.min(all_data_pd["lmass"].to_numpy())
    max_lmass = np.max(all_data_pd["lmass"].to_numpy())

    min_lsigma = np.min(np.log10(all_data_pd["sigma_re"].to_numpy()))
    max_lsigma = np.max(np.log10(all_data_pd["sigma_re"].to_numpy()))

    min_rekpc = np.min(all_data_pd["rekpc"].to_numpy())
    max_rekpc = np.max(all_data_pd["rekpc"].to_numpy())

    min_mustar = np.min(all_data_pd["mustar"].to_numpy())
    max_mustar = np.max(all_data_pd["mustar"].to_numpy())

    min_f_dm = np.min(all_data_pd["f_dm"].to_numpy())
    max_f_dm = np.max(all_data_pd["f_dm"].to_numpy())

    # Plot all data
    for i, MFP_axis, logR_M_axis, logsigma_M_axis, R_logSigma_axis, f_dm_lmass_axis, zmin, zmax in zip(
            range(5),
            np.reshape(mfp_ax, -1)[:5],
            np.reshape(RM_sigmaM_ax, -1)[:5],
            np.reshape(RM_sigmaM_ax, -1)[5:],
            np.reshape(RSigma_ax, -1)[:5],
            np.reshape(fdm_M_ax, -1)[:5],
            [0.0, 0.3, 0.9, 1.5, 2.5],
            [0.3, 0.9, 1.5, 2.5, 4.02]):

        # Condition to check the range of z	
        inz = (all_data_pd["zspec"] >= zmin) & (all_data_pd["zspec"] <= zmax)

        for num in paper_nums:
            axis_settings = {"cmap": "gist_rainbow_r", "vmin": 0.0, "vmax": 4.1, "marker": num_to_shape(num),
                             "label": num_to_label(num)}

            if num == 0:
                axis_settings = {**axis_settings, **{"alpha": 0.1}}
            elif num == 11:
                axis_settings = {**axis_settings, **{"alpha": 0.5}}

            equalsnum = (all_data_pd["highzcatnum"] == num)
            redshift = all_data_pd["zspec"][equalsnum & inz]

            # Mass Fundamental Plane
            MFP_x = all_data_pd["lradius"][equalsnum & inz].to_numpy()
            MFP_y = all_data_pd["mfp"][equalsnum & inz].to_numpy()
            MFP_axis.scatter(MFP_x, MFP_y, c=redshift, **axis_settings)

            # Mass-size relation
            logR_M_x = all_data_pd["lmass"][equalsnum & inz].to_numpy()
            logR_M_y = all_data_pd["lradius"][equalsnum & inz].to_numpy()
            logR_M_axis.scatter(logR_M_x, logR_M_y, c=redshift, **axis_settings)

            # Mass-velocity dispersion relation
            logsigma_M_x = all_data_pd["lmass"][equalsnum & inz].to_numpy()
            logsigma_M_y = all_data_pd["lsigma_re"][equalsnum & inz].to_numpy()
            logsigma_M_axis.scatter(logsigma_M_x, logsigma_M_y, c=redshift, **axis_settings)

            # Radius vs. stellar mass surface density
            R_logSigma_y = all_data_pd["lradius"][equalsnum & inz].to_numpy()
            R_logSigma_x = all_data_pd["mustar"][equalsnum & inz].to_numpy()
            R_logSigma_axis.scatter(R_logSigma_x, R_logSigma_y, c=redshift, **axis_settings)

            # Dark matter fraction vs. mass
            f_dm_lmass_x = all_data_pd["lmass"][equalsnum & inz].to_numpy()
            f_dm_lmass_y = all_data_pd["f_dm"][equalsnum & inz].to_numpy()
            f_dm_lmass_axis.scatter(f_dm_lmass_x, f_dm_lmass_y, c=redshift, **axis_settings)

        # Axis titles, labels, limits
        MFP_axis.set_title(
            f"({chr(97 + i)}) {round(zmin, 1)} < z < {round(zmax, 1)} ({round(cosmo.age(0).value - cosmo.age(zmin).value, 1)} - {round(cosmo.age(0).value - cosmo.age(zmax).value, 1)} Gyr ago)")
        logR_M_axis.set_title(f"({chr(97 + i)}) {round(zmin, 1)} < z < {round(zmax, 1)}")
        logsigma_M_axis.set_title(f"({chr(97 + i)}) {round(zmin, 1)} < z < {round(zmax, 1)}")

        MFP_axis.set_xlabel("$\log R_e\,[kpc]$")
        logR_M_axis.set_xlabel("$\log M_*\,[M_{\odot}]$")
        logsigma_M_axis.set_xlabel("$\log M_*\,[M_{\odot}]$")
        R_logSigma_axis.set_xlabel("$\log \Sigma_*$")
        f_dm_lmass_axis.set_xlabel("$\log M_*\,[M_{\odot}]$")

        MFP_axis.set_ylabel(f"${ALPHA}\,\log \sigma_* - {BETA}\,\log \Sigma_*$")
        logR_M_axis.set_ylabel("$\log R_e\,$ [kpc]")
        logsigma_M_axis.set_ylabel("$\log \sigma\,$ [km/s]")
        R_logSigma_axis.set_ylabel("$\log R_e$ [kpc]")
        f_dm_lmass_axis.set_ylabel("$\log M_{dyn}/M_*$")

        MFP_axis.set_xlim(min_x - 0.1, max_x + 0.1)
        logR_M_axis.set_xlim(min_lmass - 0.1, max_lmass + 0.1)
        logsigma_M_axis.set_xlim(min_lmass - 0.1, max_lmass + 0.1)
        R_logSigma_axis.set_xlim(min_mustar - 0.1, max_mustar + 0.1)
        f_dm_lmass_axis.set_xlim(min_lmass - 0.1, max_lmass + 0.1)

        MFP_axis.set_ylim(min_y - 0.1, max_y + 0.1)
        logR_M_axis.set_ylim(min_x - 0.1, max_x + 0.1)
        logsigma_M_axis.set_ylim(min_lsigma - 0.1, max_lsigma + 0.1)
        R_logSigma_axis.set_ylim(min_x - 0.1, max_x + 0.1)
        f_dm_lmass_axis.set_ylim(min_f_dm - 0.1, max_f_dm + 0.1)

        global_x_ax = all_data_pd["lradius"][inz].to_numpy()
        global_y_ax = all_data_pd["mfp"][inz].to_numpy()

        # Best-fit line and bootstrapping
        print(f"size: ({np.size(global_x_ax)}, {np.size(global_y_ax)})")
        bfline, fit_pms = fit_MFP(global_x_ax, global_y_ax)
        calc_zp = lambda x, y: fit_MFP(x, y)[1][0]
        zp_conf_int = bootstrap((global_x_ax, global_y_ax), calc_zp, confidence_level=conf_level, n_resamples=num_samples, vectorized=False, paired=True)

        # Calculate scatter for each z range
        scatters.append(round(scatter(global_x_ax, global_y_ax, fit_pms[0]), 3))
        calc_scatter = lambda x, y: scatter(x, y, fit_pms[0])
        scatter_conf_int = bootstrap((global_x_ax, global_y_ax), calc_scatter, confidence_level=conf_level, n_resamples=num_samples, vectorized=False, paired=True)

        # Append all important parameters in their arrays
        zero_pts.append(round(fit_pms[0], 3))
        conf_ints.append((round(zp_conf_int.confidence_interval.low, 3), round(zp_conf_int.confidence_interval.high, 3)))
        std_devs.append(round(zp_conf_int.standard_error, 3))

        MFP_axis.plot([min_x, max_x], [min_x - fit_pms[0], max_x - fit_pms[0]], c='tab:gray', label="Best-fit line")
        MFP_axis.plot([min_x, max_x], [min_x - zero_pts[0], max_x - zero_pts[0]], c='k', label="Best fit line from (a)")
        MFP_axis.text(-0.4, -2.8, f"$\gamma_z$ = {-abs(round(fit_pms[0], 3))} $\pm$ {std_devs[-1]}", size="x-large")
        MFP_axis.text(-0.4, -3.0, f"scatter = {round(scatters[-1], 3)} $\pm$ {round(scatter_conf_int.standard_error, 3)}", size="x-large")

        # log R vs. log Sigma_*
        global_R_logSigma_x = all_data_pd["mustar"][inz].to_numpy()
        global_R_logSigma_y = all_data_pd["lradius"][inz].to_numpy()

        R_logSigma_axis.plot([min_mustar, max_mustar], [(A[1] / a[1] * min_mustar) + ((B[1] - b[1]) / a[1]),
                                                        (A[1] / a[1] * max_mustar) + ((B[1] - b[1]) / a[1])], c='k',
                             label="Magneticum")
        R_logSigma_axis.plot([min_mustar, max_mustar], [(A[0] / a[0] * min_mustar) + ((B[0] - b[0]) / a[0]),
                                                        (A[0] / a[0] * max_mustar) + ((B[0] - b[0]) / a[0])],
                             c='tab:brown', label="Oser")

        slope, intercept = np.polyfit(global_R_logSigma_x, global_R_logSigma_y, 1)
        R_logSigma_axis.plot([min_mustar, max_mustar],
                             [(min_mustar * slope) + intercept, (max_mustar * slope) + intercept], c='tab:gray',
                             label="Best-fit")
        R_logSigma_axis.text(10.5, 1.0, f"m = {round(slope, 2)}")
        R_logSigma_axis.text(10.5, 0.8, f"b = {round(intercept, 2)}")
        
        # Mass-size relation from Nedkova et. al. 2021 and van der Wel et. al. 2014. The first number is Ned. and 2nd is VDW.
        quiescent_logA_vals = [(0.61, 0.60), (0.45, 0.42), (0.28, 0.22),  (0.18, 0.09), (0.18, -0.06)]
        quiescent_B_vals = [(0.68, 0.75), (0.64, 0.71), (0.63, 0.76), (0.61, 0.76), (0.61, 0.79)]
        sf_logA_vals = [(0.78, 0.86), (0.74, 0.78), (0.66, 0.70), (0.61, 0.65), (0.61, 0.51)]
        sf_B_vals = [(0.22, 0.16), (0.21, 0.16), (0.21, 0.17), (0.20, 0.18), (0.20, 0.19)]

        mass_size_line = lambda logA, B, lmass: (logA + (B*np.log10(1/(5e10)))) + B*lmass
        logR_M_axis.plot([min_lmass, max_lmass], [mass_size_line(quiescent_logA_vals[i][0], quiescent_B_vals[i][0], min_lmass), mass_size_line(quiescent_logA_vals[i][0], quiescent_B_vals[i][0], max_lmass)], c='k', label="Nedkova: Quiescent")
        logR_M_axis.plot([min_lmass, max_lmass], [mass_size_line(sf_logA_vals[i][0], sf_B_vals[i][0], min_lmass), mass_size_line(sf_logA_vals[i][0], sf_B_vals[i][0], max_lmass)], c='tab:gray', label="Nedkova: Star-forming")
        logR_M_axis.plot([min_lmass, max_lmass], [mass_size_line(quiescent_logA_vals[i][1], quiescent_B_vals[i][1], min_lmass), mass_size_line(quiescent_logA_vals[i][1], quiescent_B_vals[i][1], max_lmass)], c='k', linestyle='dashed', label="van der Wel: Quiescent")
        logR_M_axis.plot([min_lmass, max_lmass], [mass_size_line(sf_logA_vals[i][1], sf_B_vals[i][1], min_lmass), mass_size_line(sf_logA_vals[i][1], sf_B_vals[i][1], max_lmass)], c='tab:gray', linestyle='dashed',  label="van der Wel: Star-forming")


        # MFP_axis.plot([min_x, max_x], [min_x - 4.475 + (0.095 * np.log10(1 + ((zmin + zmax) / 2) - 0.063)), max_x - 4.475 + (0.095 * np.log10(1 + ((zmin + zmax) / 2) - 0.063))], c='tab:orange', label="Bezanson fitted line")
        # MFP_axis.plot([min_x, max_x], [min_x - 4.475 + (0.095 * np.log10(1 + ((0.3 + 0.9) / 2) - 0.063)), max_x - 4.475 + (0.095 * np.log10(1 + ((0.3 + 0.9) / 2) - 0.063))], c='tab:brown', label="Bezanson 0.3 < z < 0.9 fitted line")
        # MFP_axis.plot(global_x_ax, bfline, c='k')

    # Global analysis across all z
    mfp_ax_allz.set_title(f"0.0 < z < 4")
    mfp_ax_allz.set_xlabel("$\log R_e\,[kpc]$")
    mfp_ax_allz.set_ylabel(f"${ALPHA}\,\log \sigma_* - {BETA}\,\log \Sigma_*$")
    mfp_ax_allz.set_xlim(min_x - 0.1, max_x + 0.1)
    mfp_ax_allz.set_ylim(min_y - 0.1, max_y + 0.1)

    tot_x_ax = all_data_pd["lradius"].to_numpy()
    tot_y_ax = all_data_pd["mfp"].to_numpy()

    for num in paper_nums:
        if num == 0:
            opacity = 0.1
        elif num == 11:
            opacity = 0.5
        else:
            opacity = 1

        equalsnum = (all_data_pd["highzcatnum"] == num)
        x = tot_x_ax[equalsnum]
        y = tot_y_ax[equalsnum]
        redshift = all_data_pd["zspec"][equalsnum]
        mfp_ax_allz.scatter(x, y, c=redshift, cmap="gist_rainbow_r", vmin=0.3, vmax=4.1, marker=num_to_shape(num),
                    label=num_to_label(num), alpha=opacity)

    tot_bfline, tot_fit_pms = fit_MFP(tot_x_ax, tot_y_ax)
    calc_zp = lambda x, y: fit_MFP(x, y)[1][0]
    tot_zp_conf_int = bootstrap((tot_x_ax, tot_y_ax), calc_zp, confidence_level=conf_level, n_resamples=num_samples,
                                vectorized=False, paired=True)

    mfp_ax_allz.plot([min_x, max_x], [min_x - tot_fit_pms[0], max_x - tot_fit_pms[0]], c='k', label="Best-fit line")
    mfp_ax_allz.text(-0.4, -3.4,
             f"$\gamma_z$ = {-abs(round(tot_fit_pms[0], 3))} $\pm$ {round(tot_zp_conf_int.standard_error, 3)}",
             size="x-large")

    tot_scatter = scatter(tot_x_ax, tot_y_ax, tot_fit_pms[0])
    calc_scatter = lambda x, y: scatter(x, y, tot_fit_pms[0])
    tot_scatter_conf_int = bootstrap((tot_x_ax, tot_y_ax), calc_scatter, confidence_level=conf_level,
                                     n_resamples=num_samples, vectorized=False, paired=True)

    mfp_ax_allz.text(-0.4, -3.6, f"scatter = {round(tot_scatter, 3)} $\pm$ {round(tot_scatter_conf_int.standard_error, 3)}",
             size="x-large")

    print("------------------------------ ALL Z ANALYSIS ------------------------------------")
    print(f"Best-fit zp:   {round(tot_fit_pms[0], 3)} +- {round(tot_zp_conf_int.standard_error, 3)}")
    print(f"Total scatter: {round(tot_scatter, 3)} +- {round(tot_scatter_conf_int.standard_error, 3)}")

    # Printing parameters	
    data_to_print = list(map(list, zip(zero_pts, std_devs, [str(ci) for ci in conf_ints], scatters)))

    z_headers = ["0.0 < z < 0.3 | ", "0.3 < z < 0.9 | ", "0.9 < z < 1.5 | ", "1.5 < z < 2.5 | ", "3 < z < 4     | "]
    label_headers = ["Zero-points (fixed α and β)", "Std. dev.", "Conf-intervals", "Scatter"]

    print("----------------------------------------------------------------------------------")
    print(pd.DataFrame(data_to_print, z_headers, label_headers))
    print("----------------------------------------------------------------------------------")

    mfp_ax[1][2].legend(*get_lns_lbls(mfp_ax), loc='center left', prop={'size': 10})
    mfp_ax[1][2].axis("off")
    mfp_ax_allz.legend(*get_lns_lbls(mfp_ax_allz), loc='lower right', prop={'size': 10})
    RM_sigmaM_fig.legend(*get_lns_lbls(RM_sigmaM_ax), loc='center right', prop={'size': 8})
    RSigma_ax[1][2].legend(*get_lns_lbls(RSigma_ax), loc='center left', prop={'size': 10})
    RSigma_ax[1][2].axis("off")

    cbaxes = inset_axes(mfp_ax[1][2], width="5%", height="90%", loc="center right")

    mfp_fig.colorbar(
        matplotlib.cm.ScalarMappable(cmap="gist_rainbow_r", norm=matplotlib.colors.Normalize(vmin=0.0, vmax=4.1)),
        orientation='vertical',
        cax=cbaxes,
        pad=0.1,
        fraction=0.1,
        label='Redshift'
    )

    mfp_fig_allz.colorbar(
        matplotlib.cm.ScalarMappable(cmap="gist_rainbow_r", norm=matplotlib.colors.Normalize(vmin=0.0, vmax=4.1)),
        orientation='horizontal',
        ax=mfp_ax_allz,
        pad=0.1,
        fraction=0.04,
        label='Redshift'
    )

    RM_sigmaM_fig.colorbar(
        matplotlib.cm.ScalarMappable(cmap="gist_rainbow_r", norm=matplotlib.colors.Normalize(vmin=0.0, vmax=4.1)),
        orientation='horizontal',
        ax=RM_sigmaM_ax[1],
        pad=0.5,
        fraction=0.04,
        label='Redshift'
    )

    RSigma_fig.colorbar(
        matplotlib.cm.ScalarMappable(cmap="gist_rainbow_r", norm=matplotlib.colors.Normalize(vmin=0.0, vmax=4.1)),
        orientation='vertical',
        ax=RSigma_ax[1][2],
        pad=0,
        label='Redshift'

    )

    mfp_fig.subplots_adjust(
        top=0.88,
        bottom=0.07,
        left=0.125,
        right=0.9,
        hspace=0.3,
        wspace=0.2
    )

    RM_sigmaM_fig.subplots_adjust(
        top=0.88,
        bottom=0.19,
        left=0.125,
        right=0.91,
        hspace=0.3,
        wspace=0.31
    )

    RSigma_fig.subplots_adjust(
        right=0.885
    )
    end_t = pf()
    print(round(end_t - start_t, 2))
    plt.show()


if __name__ == "__main__":
    main()

