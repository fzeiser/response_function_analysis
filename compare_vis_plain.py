import matplotlib.pyplot as plt
import ompy as om
from pathlib import Path

basedir = Path("mama_spectra")
files = [file for file in basedir.iterdir()]
files.sort()

# files = [files[0],
#          "mama_spectra/grid_2700_run_60Co.root.m",
#          "mama_spectra/grid_2701_run_60Co.root.m",
#          "mama_spectra/grid_1585_run_60Co.root.m", ]

data = [om.Vector(path=file) for file in files]

Nrows = int((len(files) / 5)/2 + 0.5)
if Nrows < 2:
    Nrows = 2
fig, ax = plt.subplots(Nrows, 2, sharex=True)

ax = ax.flatten()
for i, vec in enumerate(data):
    # vec.save(str(files[i]) + ".txt")
    vec.rebin(factor=10)
    print(f"{vec.values.sum():.2e}")
    vec.values /= vec.values.sum()
    vec.plot(ax=ax[0])

    axdiff = ax[i % (Nrows*2-1)+1]
    if i == 0:
        vec0 = vec
    else:
        diff = (vec0 - vec)/vec0 * 100
        diff.plot(ax=axdiff, label=files[i])
    # if i == 10:
    #     break

ax[0].set_yscale("log")
ax[0].set_xlim(0.2, 1.5)

for axes in ax[1:]:
    axes.set_ylim(-10, 10)
    axes.legend(fontsize="xx-small")
# ax1.set_ylim(-10, 10)



plt.show()
