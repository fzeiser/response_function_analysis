import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import fnmatch
import os
import re
from tqdm import tqdm
from uncertainties import unumpy

import sys

import pandas as pd
# sys.path.append("../exp")

from compare import SpectrumComparison, uinterp1D

def save_coords_from_click(fig, fname="coords.txt"):
    try:
        coords = np.loadtxt(fname)
        print("loading file (instead)")
        # plt.show()
    except OSError:
        coords = []

        def onclick(event):
            """ depends on global `coords` """
            ix, iy = event.xdata, event.ydata
            print(f'x = {ix:.0f}, y = {iy:.0f}')
            coords.append((ix, iy))
            return coords

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        fig.canvas.mpl_disconnect(cid)

        np.savetxt(fname, np.array(coords),
                   header="x y")
    coords = np.array(coords)
    coords = coords.astype(int)
    return coords


def get_fom(fnisotope,
            fname_exp, fname_bg, fwhm_pars,
            measure_time_exp, measure_time_bg, idets,
            Efit_low, Efit_high,
            do_plot=True, printout=False, hotfix_low=None):
    """ get figure of merrit

    fnisotope: str, like "60Co", or "152Eu"
    """
    fname_sims = []
    for file in os.listdir('mama_spectra/root_files'):
        if fnmatch.fnmatch(file, f'grid_9_*{fnisotope}*_all.m'):
            fname_sims.append(os.path.join("mama_spectra/root_files", file))
    fname_sims.sort()
    grid_points = np.full_like(fname_sims, np.nan, dtype=float)
    foms = np.zeros((len(fname_sims), 3))

    for i, fname_sim in enumerate(tqdm(fname_sims)):
        # if i > 2:
        #     break
        if printout:
            print("fitting: ", fname_sim)
        sc = SpectrumComparison()
        sc.get_data(fname_sim, fname_exp, fname_bg, fwhm_pars,
                    measure_time_exp, measure_time_bg, idet=idets,
                    recalibrate=True)
        sc.scale_sim_to_exp_area(Efit_low, Efit_high)

        xsim = sc.xsim
        fexp = sc.fexp
        fsim_scaled = sc.fsim_scaled
        Emax_hotfix = 200
        if fnisotope == "60Co":
            denom = fsim_scaled(xsim)
            denom[denom==0] = np.nan
            hotfix_low = fexp(xsim)/denom
            hotfix_low[unumpy.isnan(hotfix_low)] = 1
            hotfix_low[xsim > 200] = 1
            sc.uysim_scaled *= hotfix_low
            sc.fsim_scaled = uinterp1D(xsim, sc.uysim_scaled)
        else:
            pass
            sc.uysim_scaled *= hotfix_low
            sc.fsim_scaled = uinterp1D(xsim, sc.uysim_scaled)
            # denom = fsim_scaled(xsim)
            # denom[denom==0] = np.nan
            # hotfix_low = fexp(xsim)/denom
            # sc.uysim_scaled *= hotfix_low

        print("scale factor:", sc.scale_factor)
        chi2 = sc.get_chi2()
        rel_diff, rel_diff_smooth = sc.get_rel_diff(smooth_window_keV=20)

        foms[i, :] = sc.fom(Ecompare_low, Ecompare_high, printout=False)
        # print(sc.fom(Ecompare_low, Ecompare_high))
        grid_points[i] = int(re.search(r"grid_(\d*)_", fname_sim)[1])
        if do_plot:
            fig, _ = sc.plots(title=fname_sim, xmax=1400)
        fig.savefig(f"figs_hotfix_low/{fnisotope}_{grid_points[i]:.0f}.png")
        # plt.show()
        plt.close(fig)

    if printout:
        ltab = [[name, *foms[i, :]] for i, name in enumerate(fname_sims)]
        print("\nComparisons between {} and {} keV:"
              .format(Ecompare_low, Ecompare_high))
        print(tabulate(ltab,
                       headers=["Name", "chi2", "rel_diff[%]",
                                "rel_diff_smoothed[%]"],
                       floatfmt=".2f"))
    df = pd.DataFrame(foms, columns=[f"chi2_{fnisotope}",
                                     f"rel_diff_{fnisotope}",
                                     f"rel_diff_smoothed_{fnisotope}"])
    df["grid_point"] = grid_points
    df = df[df.grid_point.notnull()]  # workaround if going through whole loop
    df = df.astype({"grid_point": 'int'}, copy=False)

    return df, hotfix_low


if __name__ == "__main__":
    # fwhm_pars = np.array([73.2087, 0.50824, 9.62481e-05])
    # Frank June 2020
    fwhm_pars = np.array([60.6499, 0.458252, 0.000265552])

    files = {
        "133Ba": {"t": 1049.48},
        "60Co": {"t":  1123.57},
        "152Eu": {"t": 1065.1},
        "137Cs": {"t": 676.307},
        "241Am": {"t": 969.774},
        "Bg": {"t":    1432.19}}

    # dets with bet resolution (internat peak structure vissible)
    idets = [1, 2, 6, 8, 10, 11, 12,
             14, 15, 16, 17, 18, 19, 20, 21, 22,
             24, 25, 27, 29]

    # # 60Co
    fname_exp = "exp/60Co.txt"
    fname_bg = "exp/Bg.txt"
    measure_time_exp = files["60Co"]["t"]  # //seconds
    measure_time_bg = files["Bg"]["t"]  # //seconds
    Efit_low = 1173 - 50 - 50
    Efit_high = 1173 + 50 + 50
    Ecompare_low = 50
    Ecompare_high = 1000

    hotfix_low = None

    df = get_fom("60Co",
                 fname_exp, fname_bg, fwhm_pars,
                 measure_time_exp, measure_time_bg, idets,
                 Efit_low, Efit_high,
                 do_plot=True, printout=False)

    df_all, hotfix_low = df

    # 152Eu
    fname_exp = "exp/152Eu.txt"
    fname_bg = "exp/Bg.txt"
    measure_time_exp = files["152Eu"]["t"]  # //seconds
    measure_time_bg = files["Bg"]["t"]  # //seconds
    Efit_low = 720
    Efit_high = 830
    Ecompare_low = 50
    Ecompare_high = 1000

    df, hotfix_low = get_fom("152Eu",
                 fname_exp, fname_bg, fwhm_pars,
                 measure_time_exp, measure_time_bg, idets,
                 Efit_low, Efit_high,
                 do_plot=True, printout=False, hotfix_low=hotfix_low)
    df[df.grid_point.notnull()]
    df_all = df_all.merge(df, on="grid_point", how="outer")


    # 133Ba
    fname_exp = "exp/133Ba.txt"
    fname_bg = "exp/Bg.txt"
    measure_time_exp = files["133Ba"]["t"]  # //seconds
    measure_time_bg = files["Bg"]["t"]  # //seconds
    Efit_low = 285
    Efit_high = 320
    # Efit_low = 330
    # Efit_high = 400
    Ecompare_low = 50
    Ecompare_high = 300

    df, hotfix_low = get_fom("133Ba",
                 fname_exp, fname_bg, fwhm_pars,
                 measure_time_exp, measure_time_bg, idets,
                 Efit_low, Efit_high,
                 do_plot=True, printout=False, hotfix_low=hotfix_low)
    df_all = df_all.merge(df, on="grid_point", how="outer")

    # # 137Cs
    fname_exp = "exp/137Cs.txt"
    fname_bg = "exp/Bg.txt"
    measure_time_exp = files["137Cs"]["t"]  # //seconds
    measure_time_bg = files["Bg"]["t"]  # //seconds
    Efit_low = 600
    Efit_high = 700
    Ecompare_low = 50
    Ecompare_high = 300

    df, hotfix_low = get_fom("137Cs",
                 fname_exp, fname_bg, fwhm_pars,
                 measure_time_exp, measure_time_bg, idets,
                 Efit_low, Efit_high,
                 do_plot=True, printout=False, hotfix_low=hotfix_low)

    df_all = df_all.merge(df, on="grid_point", how="outer")
    # df_all.to_pickle("chi2_df.pickle")
    print(df_all[:8])
