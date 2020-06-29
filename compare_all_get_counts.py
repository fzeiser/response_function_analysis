import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import fnmatch
import os
import re
from tqdm import tqdm
from datetime import datetime

from uncertainties import unumpy
import sys

import pandas as pd
# sys.path.append("../exp")

from compare import SpectrumComparison, fFWHM

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
            do_plot=True, printout=False,
            manual_ratio=None):
    """ get figure of merrit

    fnisotope: str, like "60Co", or "152Eu"
    """
    fname_sims = []
    for file in os.listdir('mama_spectra/root_files'):
        if fnmatch.fnmatch(file, f'*{fnisotope}*_all.m'):
            fname_sims.append(os.path.join("mama_spectra/root_files", file))
    fname_sims.sort()
    grid_points = np.full_like(fname_sims, np.nan, dtype=float)
    foms = np.zeros((len(fname_sims), 3))

    for i, fname_sim in enumerate(tqdm(fname_sims)):
        if i > 3:
            break
        if printout:
            print("fitting: ", fname_sim)
        sc = SpectrumComparison()
        sc.get_data(fname_sim, fname_exp, fname_bg, fwhm_pars,
                    measure_time_exp, measure_time_bg, idet=idets,
                    recalibrate=True)
        if manual_ratio is None:
            sc.scale_sim_to_exp_area(Efit_low, Efit_high)
        else:
            sc.scale_sim_to_exp_manual(manual_ratio)

        ncounts_Elims = np.zeros((len(Elimits), 2))
        for i, (E1, E2) in enumerate(Elimits):
            ncounts_Elims[i, 0] = sc.get_area(sc.exp, E1, E2)
            sim_scaled = np.c_[sc.xsim, unumpy.nominal_values(sc.uysim_scaled)]
            ncounts_Elims[i, 1] = sc.get_area(sim_scaled, E1, E2)


        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        counts_exp = unumpy.uarray(ncounts_Elims[:, 0],
                                   np.sqrt(ncounts_Elims[:, 0]))
        counts_sim = unumpy.uarray(ncounts_Elims[:, 1]/manual_ratio,
                                   np.sqrt(ncounts_Elims[:, 1])/manual_ratio)
        counts_sim *= manual_ratio
        counts_sim *= 5  # *5 due to binwidth
        diff = (counts_exp-counts_sim)/counts_exp * 100
        ax1.errorbar(Elimits.mean(axis=1), unumpy.nominal_values(counts_exp),
                     yerr=unumpy.std_devs(counts_exp), fmt="o")
        ax1.errorbar(Elimits.mean(axis=1), unumpy.nominal_values(counts_sim),
                     yerr=unumpy.std_devs(counts_sim), fmt="x")
        ax2.errorbar(Elimits.mean(axis=1), unumpy.nominal_values(diff),
                     yerr=unumpy.std_devs(diff), fmt="x")

        # # syst. error (scaling from activities)
        # counts_exp_low = ncounts_Elims[:, 0] * 0.97
        # counts_exp_high = ncounts_Elims[:, 0] * 1.03
        # diff_low = (counts_exp_low-counts_sim)/counts_exp_low *100
        # diff_high = (counts_exp_high-counts_sim)/counts_exp_high *100
        # ax1.plot(Elimits.mean(axis=1), counts_exp_low, "o", c="b", ms=1, alpha=0.8)
        # ax2.plot(Elimits.mean(axis=1), diff_low, "x", c="b", ms=1, alpha=0.8)
        # ax1.plot(Elimits.mean(axis=1), counts_exp_high, "o", c="b", ms=1, alpha=0.8)
        # ax2.plot(Elimits.mean(axis=1), diff_high, "x", c="b", ms=1, alpha=0.8)
        ax2.axhline(0, c="k", ls="--")
        ax2.set_ylim(-20, 20)


        tot_exp = sc.get_area(sc.exp, 100, 1300)
        tot_sim = sc.get_area(sim_scaled, 100, 1300)
        tot_sim *= 5 # 5 due to binwitdth
        tot_diff = (tot_exp-tot_sim)/tot_exp * 100
        print(f"Tot diff Elims1 [%]: {tot_diff:.2f}")

        tot_exp = sc.get_area(sc.exp, 50, 200)
        tot_sim = sc.get_area(sim_scaled, 50, 200)
        tot_sim *= 5 # 5 due to binwitdth
        tot_diff = (tot_exp-tot_sim)/tot_exp * 100
        print(f"Tot diff Elims2 [%]: {tot_diff:.2f}")

        plt.show()

        chi2 = sc.get_chi2()
        rel_diff, rel_diff_smooth = sc.get_rel_diff(smooth_window_keV=20)

        foms[i, :] = sc.fom(Ecompare_low, Ecompare_high, printout=False)
        # print(sc.fom(Ecompare_low, Ecompare_high))
        grid_points[i] = int(re.search(r"grid_(-*\d*)_", fname_sim)[1])
        if do_plot:
            fig, _ = sc.plots(title=fname_sim, xmax=1500)
        fig.savefig(f"figs/{fnisotope}_{grid_points[i]:.0f}.png")
        plt.show()
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

    return df


if __name__ == "__main__":
    # fwhm_pars = np.array([73.2087, 0.50824, 9.62481e-05])
    # Frank June 2020
    fwhm_pars = np.array([60.6499, 0.458252, 0.000265552])


    # print(fFWHM(80, fwhm_pars))
    # sys.exit()

    files = {
        "133Ba": {"t": 1049.48, "ndecays": 4.28917e+07},
        "60Co": {"t":  1123.57, "ndecays": 2.74162e+07},
        "152Eu": {"t": 1065.10, "ndecays": 3.45591e+07},
        "137Cs": {"t": 676.307, "ndecays": 2.77619e+07},
        "241Am": {"t": 969.774},
        "Bg": {"t":    1432.19, }}

    ndecays_sim = 1e5
    diff_binwidth = 5

    # dets with bet resolution (internat peak structure vissible)
    idets = [1, 2, 6, 8, 10, 11, 12,
             14, 15, 16, 17, 18, 19, 20, 21, 22,
             24, 25, 27, 29]

    # # 60Co
    Elimits = np.array([
                    [1110,  1230],
                    [1270,  1350] # very short due to Bg subraction problem
                    ])


    fname_exp = "exp/60Co.txt"
    fname_bg = "exp/Bg.txt"
    measure_time_exp = files["60Co"]["t"]  # //seconds
    measure_time_bg = files["Bg"]["t"]  # //seconds
    Efit_low = 1173 - 50 - 50
    Efit_high = 1173 + 50 + 50
    Ecompare_low = 50
    Ecompare_high = 1000

    df = get_fom("60Co",
                 fname_exp, fname_bg, fwhm_pars,
                 measure_time_exp, measure_time_bg, idets,
                 Efit_low, Efit_high,
                 do_plot=True, printout=True,
                 manual_ratio=files["60Co"]["ndecays"]/ndecays_sim/diff_binwidth)

    # df_all = df

    # 152Eu
    fname_exp = "exp/152Eu.txt"
    fname_bg = "exp/Bg.txt"
    measure_time_exp = files["152Eu"]["t"]  # //seconds
    measure_time_bg = files["Bg"]["t"]  # //seconds
    Efit_low = 720
    Efit_high = 830
    Ecompare_low = 50
    Ecompare_high = 1000

    Elimits = np.array([
                    [105,  130],
                    [220,  260],
                    [315,  360],
                    [390,  420],
                    [420,  460],
                    [660,  700],
                    [740,  810],
                    [830,  900],
                    [910,  1000],
                    [1030, 1160],
                    [1170, 1240],
                    [1260, 1342]])

    df = get_fom("152Eu",
                 fname_exp, fname_bg, fwhm_pars,
                 measure_time_exp, measure_time_bg, idets,
                 Efit_low, Efit_high,
                 do_plot=True, printout=True,
                 manual_ratio=files["152Eu"]["ndecays"]/ndecays_sim/diff_binwidth)
    df[df.grid_point.notnull()]
    # df_all = df_all.merge(df, on="grid_point", how="outer")


    # # 133Ba
    # fname_exp = "exp/133Ba.txt"
    # fname_bg = "exp/Bg.txt"
    # measure_time_exp = files["133Ba"]["t"]  # //seconds
    # measure_time_bg = files["Bg"]["t"]  # //seconds
    # Efit_low = 285
    # Efit_high = 320
    # # Efit_low = 330
    # # Efit_high = 400
    # Ecompare_low = 50
    # Ecompare_high = 300

    # df = get_fom("133Ba",
    #              fname_exp, fname_bg, fwhm_pars,
    #              measure_time_exp, measure_time_bg, idets,
    #              Efit_low, Efit_high,
    #              do_plot=True, printout=False,
    #              manual_ratio=files["133Ba"]["ndecays"]/ndecays_sim/diff_binwidth)
    # # df_all = df_all.merge(df, on="grid_point", how="outer")

    # # # 137Cs
    # fname_exp = "exp/137Cs.txt"
    # fname_bg = "exp/Bg.txt"
    # measure_time_exp = files["137Cs"]["t"]  # //seconds
    # measure_time_bg = files["Bg"]["t"]  # //seconds
    # Efit_low = 600
    # Efit_high = 700
    # Ecompare_low = 50
    # Ecompare_high = 300

    # df = get_fom("137Cs",
    #              fname_exp, fname_bg, fwhm_pars,
    #              measure_time_exp, measure_time_bg, idets,
    #              Efit_low, Efit_high,
    #              do_plot=True, printout=False,
    #              manual_ratio=files["137Cs"]["ndecays"]/ndecays_sim/diff_binwidth)

    # # df_all = df_all.merge(df, on="grid_point", how="outer")

    # now = datetime.now()
    # df_all.to_pickle(f'chi2_df_{now.strftime("%y%d%m_%H%M%S")}.pickle')
    # print(df_all[:8])
