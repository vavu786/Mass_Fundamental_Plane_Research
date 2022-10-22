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


def log_likelihood(theta, x, y, yerr):
    m, b, log_f = theta
    model = m * x + b
    sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
    

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
    data_pd["f_dm"] = calc_f_dm(data_pd["sigma_re"].to_numpy(), data_pd["rekpc"].to_numpy(),
                                data_pd["lmass"].to_numpy())

    data_pd["esigma_re"] = data_pd["esigma_re_m"]
    data_useful_pd = data_pd[["highzcatnum", "zspec", "rekpc", "erekpc", "lmass", "sigma_re", "esigma_re", "mustar", "f_dm"]]

    # Belli (24 galaxies) (Yellow). 1.526 < z < 2.435
    Belli = np.genfromtxt("z=1.7_Belli.dat", names=True, dtype=None, skip_header=2, skip_footer=18, encoding=None)
    Belli_pd = pd.DataFrame(Belli)
    Belli_pd["rekpc"] = Belli_pd["r_maj"].to_numpy() * np.sqrt(Belli_pd["q"].to_numpy())
    Belli_pd["erekpc"] = Belli_pd["r_maj_unc"].to_numpy() * np.sqrt(Belli_pd["q"].to_numpy())
    Belli_pd["esigma_re"] = Belli_pd["sigma_re_unc"].to_numpy()
    Belli_pd["highzcatnum"] = np.full(Belli_pd.shape[0], 9)
    Belli_pd["mustar"] = Belli_pd["lmass"].to_numpy() - np.log10(2 * np.pi * np.square(Belli_pd["rekpc"].to_numpy()))
    Belli_pd["f_dm"] = calc_f_dm(Belli_pd["sigma_re"].to_numpy(), Belli_pd["rekpc"].to_numpy(),
                                 Belli_pd["lmass"].to_numpy())
    
    Belli_useful_pd = Belli_pd[["highzcatnum", "zspec", "rekpc", "erekpc", "lmass", "sigma_re", "esigma_re", "mustar", "f_dm"]]

    # Forrest (14 galaxies) (Magenta) 
    Forrest = np.genfromtxt("z=3.5_Forrest.dat", names=True, dtype=None, skip_header=1, skip_footer=1)
    Forrest_pd = pd.DataFrame(Forrest)
    Forrest_pd["highzcatnum"] = np.full(Forrest_pd.shape[0], 10)
    Forrest_pd["mustar"] = Forrest_pd["lmass"].to_numpy() - np.log10(
        2 * np.pi * np.square(Forrest_pd["rekpc"].to_numpy()))
    Forrest_pd["f_dm"] = calc_f_dm(Forrest_pd["sigma_re"].to_numpy(), Forrest_pd["rekpc"].to_numpy(),
                                   Forrest_pd["lmass"].to_numpy())

    Forrest_useful_pd = Forrest_pd[["highzcatnum", "zspec", "rekpc", "erekpc", "lmass", "sigma_re", "esigma_re", "mustar", "f_dm"]]

    # LEGA-C data (1419 galaxies)
    legac_table = astropy.table.Table.read("legac_fp_selection.fits", format="fits")
    legac_pd = legac_table.to_pandas()
    legac_pd["highzcatnum"] = np.full(legac_pd.shape[0], 11)
    legac_pd["rekpc"] = 10 ** legac_pd["log_rec_kpc"].to_numpy()
    legac_pd["erekpc"] = 10 ** legac_pd["e_log_rec_kpc"].to_numpy()
    legac_pd["esigma_re"] = 10 ** legac_pd["e_log_sigma_stars"].to_numpy()
    legac_pd["sigma_re"] = 10 ** legac_pd["log_sigma_stars"].to_numpy()
    legac_pd.rename(columns={"z_spec": "zspec", "log_mstar": "lmass", "log_Sigma_star": "mustar"}, inplace=True)
    legac_pd["f_dm"] = calc_f_dm(legac_pd["sigma_re"].to_numpy(), legac_pd["rekpc"].to_numpy(),
                                 legac_pd["lmass"].to_numpy())

    legac_useful_pd = legac_pd[["highzcatnum", "zspec", "rekpc", "erekpc", "lmass", "sigma_re", "esigma_re", "mustar", "f_dm"]]

    # SDSS (18,573 galaxies)
    sdss_table = astropy.table.Table.read("sdss_fp_selection_magphys_pymorph.fits", format="fits")
    sdss_pd = sdss_table.to_pandas()
    sdss_pd["highzcatnum"] = np.full(sdss_pd.shape[0], 0)
    sdss_pd["rekpc"] = 10 ** sdss_pd["log_rec_kpc"].to_numpy()
    sdss_pd["sigma_re"] = 10 ** sdss_pd["log_sigma_re"].to_numpy()
    sdss_pd.rename(columns={"z": "zspec", "log_mstar": "lmass", "log_Sigma_star": "mustar"}, inplace=True)
    sdss_pd["f_dm"] = calc_f_dm(sdss_pd["sigma_re"].to_numpy(), sdss_pd["rekpc"].to_numpy(),
                                sdss_pd["lmass"].to_numpy())

    sdss_pd["erekpc"] = np.full(sdss_pd.shape[0], 0)
    sdss_pd["esigma_re"] = np.full(sdss_pd.shape[0], 0)
    
    sdss_useful_pd = sdss_pd[["highzcatnum", "zspec", "rekpc", "erekpc", "lmass", "sigma_re", "esigma_re", "mustar", "f_dm"]]

    all_data_pd = pd.concat([data_useful_pd, Belli_useful_pd, Forrest_useful_pd, legac_useful_pd, sdss_useful_pd])
    all_data_pd.set_index(pd.Index(range(all_data_pd.shape[0])), inplace=True)

    # _______________________________________________________________________________________________

    mfp_fig, mfp_ax = plt.subplots(2, 3)

    zero_pts = []
    conf_ints = []
    std_devs = []
    scatters = []

    # Prepare all axes
    all_data_pd["lradius"] = np.log10(all_data_pd["rekpc"].to_numpy())
    all_data_pd["mfp"] = ALPHA * np.log10(all_data_pd["sigma_re"].to_numpy()) - BETA * all_data_pd["mustar"]
    all_data_pd["lsigma_re"] = np.log10(all_data_pd["sigma_re"].to_numpy())
    
    
    print(all_data_pd[["lradius", "mfp"]])

    min_x = np.min(all_data_pd["lradius"].to_numpy())
    max_x = np.max(all_data_pd["lradius"].to_numpy())

    min_y = np.min(all_data_pd["mfp"].to_numpy())
    max_y = np.max(all_data_pd["mfp"].to_numpy())
 
    # Plot all data
    for i, MFP_axis, zmin, zmax in zip(range(5), np.reshape(mfp_ax, -1)[:5], [0.0, 0.3, 0.9, 1.5, 2.5], [0.3, 0.9, 1.5, 2.5, 4.02]):

        # Condition to check the range of z	
        inz = (all_data_pd["zspec"] >= zmin) & (all_data_pd["zspec"] <= zmax)
        
        # Setting opacity, equalsnum condition, and plotting (not fitting) data points
        for num in paper_nums:
            axis_settings = {"cmap": "gist_rainbow_r", "vmin": 0.0, "vmax": 4.1, "marker": num_to_shape(num),
                             "label": num_to_label(num)}
            if num == 0:
                axis_settings = {**axis_settings, **{"alpha": 0.1}}
            elif num == 11:
                axis_settings = {**axis_settings, **{"alpha": 0.5}}

            equalsnum = (all_data_pd["highzcatnum"] == num)
            redshift = all_data_pd["zspec"][equalsnum & inz]

            MFP_x = all_data_pd["lradius"][equalsnum & inz].to_numpy()
            MFP_y = all_data_pd["mfp"][equalsnum & inz].to_numpy()
            MFP_axis.scatter(MFP_x, MFP_y, c=redshift, **axis_settings)
     
        # Axis titles, labels, limits
        MFP_axis.set_title(
            f"({chr(97 + i)}) {round(zmin, 1)} < z < {round(zmax, 1)} ({round(cosmo.age(0).value - cosmo.age(zmin).value, 1)} - {round(cosmo.age(0).value - cosmo.age(zmax).value, 1)} Gyr ago)")
        MFP_axis.set_xlabel("$\log R_e\,[kpc]$")
        MFP_axis.set_ylabel(f"${ALPHA}\,\log \sigma_* - {BETA}\,\log \Sigma_*$")
        MFP_axis.set_xlim(min_x - 0.1, max_x + 0.1)
        MFP_axis.set_ylim(min_y - 0.1, max_y + 0.1)
        
        # Axes for each subplot
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
        
        # Plot the best-fit lines and write text on plots
        MFP_axis.plot([min_x, max_x], [min_x - fit_pms[0], max_x - fit_pms[0]], c='tab:gray', label="Best-fit line")
        MFP_axis.plot([min_x, max_x], [min_x - zero_pts[0], max_x - zero_pts[0]], c='k', label="Best fit line from (a)")
        MFP_axis.text(-0.4, -2.8, f"$\gamma_z$ = {-abs(round(fit_pms[0], 3))} $\pm$ {std_devs[-1]}", size="x-large")
        MFP_axis.text(-0.4, -3.0, f"scatter = {round(scatters[-1], 3)} $\pm$ {round(scatter_conf_int.standard_error, 3)}", size="x-large")

        
    # Printing parameters	
    data_to_print = list(map(list, zip(zero_pts, std_devs, [str(ci) for ci in conf_ints], scatters)))

    z_headers = ["0.0 < z < 0.3 | ", "0.3 < z < 0.9 | ", "0.9 < z < 1.5 | ", "1.5 < z < 2.5 | ", "3 < z < 4     | "]
    label_headers = ["Zero-points (fixed α and β)", "Std. dev.", "Conf-intervals", "Scatter"]

    print("----------------------------------------------------------------------------------")
    print(pd.DataFrame(data_to_print, z_headers, label_headers))
    print("----------------------------------------------------------------------------------")

    mfp_ax[1][2].legend(*get_lns_lbls(mfp_ax), loc='center left', prop={'size': 10})
    mfp_ax[1][2].axis("off")

    cbaxes = inset_axes(mfp_ax[1][2], width="5%", height="90%", loc="center right")

    mfp_fig.colorbar(
        matplotlib.cm.ScalarMappable(cmap="gist_rainbow_r", norm=matplotlib.colors.Normalize(vmin=0.0, vmax=4.1)),
        orientation='vertical',
        cax=cbaxes,
        pad=0.1,
        fraction=0.1,
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

    end_t = pf()
    print(round(end_t - start_t, 2))
    plt.show()


if __name__ == "__main__":
    main()

