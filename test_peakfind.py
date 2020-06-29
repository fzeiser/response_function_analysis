import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences

import ompy as om

data = om.Vector(path="mama_spectra/root_files/grid_-1_run_152Eu.root.m")

x = data.values
peaks, properties = find_peaks(x, prominence=200)
# plt.plot(data.E[peaks], x[peaks], "xr")
# plt.plot(x)
# plt.legend(['distance'])

prominences = peak_prominences(x, peaks)[0]
contour_heights = x[peaks] - prominences
plt.plot(data.E, x)
plt.plot(data.E[peaks], x[peaks], "x")
plt.vlines(x=data.E[peaks], ymin=contour_heights, ymax=x[peaks])
# plt.show()

print(np.c_[data.E[peaks]*1000, prominences])

plt.show()
