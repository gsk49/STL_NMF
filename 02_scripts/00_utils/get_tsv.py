# Some methods require tsv input
# This script converts the csv files to tsv files

import os
import pandas as pd
import numpy as np

x_names = os.listdir("00_synthetic/00_PDAC_A/01_X_Noise/05_clust/01_x")
genes = np.array(pd.read_csv("./zzz_outputs/Genes.csv", header=None, sep=" ").squeeze())
locs = np.array(pd.read_csv("./01_real_data/locs.csv", header=None, sep=" ").squeeze())

print(genes)
print(locs)

for name in x_names:
    x = pd.read_csv("00_synthetic/00_PDAC_A/01_X_Noise/05_clust/01_x/"+name, sep=",", header=None)
    x = x
    x.columns = locs
    x.index = genes
    # np.savetxt("00_synthetic/00_PDAC_A/00_No_X_Noise/03_clust/03_x_tsv/"+name[:-3]+"tsv", x, fmt="%d", delimiter="\t")
    x.to_csv("00_synthetic/00_PDAC_A/01_X_Noise/05_clust/03_x_tsv/"+name[:-3]+"tsv", sep="\t", index=True, header=True, index_label="Genes")