import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce

def plotting(df):
    df_norm = df.copy()
    for name in ynames:
        df_norm[name] /= df_norm[name].min()

    # g = sns.FacetGrid(df, row=grid.keys())

    def rand_jitter(arr, loc=0, scale=0.1):
        return np.random.normal(loc, scale, len(arr))

    y_spread = np.linspace(-len(ynames)/2, len(ynames), len(ynames))
    for xname in grid.keys():
        if xname == "grid_point":
            continue
        fig, ax = plt.subplots()
        for iy, yname in enumerate(ynames):
            try:
                scatter = abs(np.diff(df[xname].unique()).min() * 0.02)
            except ValueError:
                scatter = 0.01
            x = rand_jitter(df[xname], df[xname]+y_spread[iy]*scatter, scatter)
            ax = sns.scatterplot(x=x, y=yname, data=df, label=yname,
                                 x_jitter=0.05)
            ax.set_title(xname)
    print(df_sorted)


df = pd.read_pickle("chi2_df.pickle")

grid = pd.read_pickle("/home/fabiobz/Desktop/Masterthesis/geant_projects/"
                      "OCL/OscarBuild_new/grid.pickle")
df = df.merge(grid, on="grid_point", how="left")

ynames = ["rel_diff_" + name for name in ["60Co", "152Eu", "133Ba", "137Cs"]]
# df_short = df[[*ynames, *grid.keys()]]

print(df[df.det==16.0])

df_sorted = []
for name in ynames:
    df_sort = df.sort_values(name)["grid_point"][:20]
    df_sorted.append(df_sort)

df_sorted = np.array(df_sorted)
ncols = np.array(df_sorted).shape[0]
index_cols = np.arange(ncols)
common = []
for i in range(ncols):
    cols_subset = np.delete(index_cols, i)
    intersect_subset = reduce(np.intersect1d, df_sorted[cols_subset])
    common.append(intersect_subset.tolist())
common = [item for sublist in common for item in sublist]
df_common = df[df.grid_point.isin(common)][grid.keys()]
print(df[df.grid_point.isin(common)][grid.keys()])

# df[df.grid_point.isin(common)][grid.keys()]

sns.pairplot(data=df_common)

# print(np.array(df_sorted).shape[0])
# for i in df_sorted.shape
# print(reduce(np.intersect1d, df_sorted[:2]))


plotting(df[df.grid_point>=0])
plotting(df[df.det==16.0])

plt.show()
