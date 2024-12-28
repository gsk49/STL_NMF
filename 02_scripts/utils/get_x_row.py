import pandas as pd
import numpy as np
import os

# S = np.array(pd.read_csv("01_real_data/S_GSE111672_PDAC-A-indrop-filtered-expMat_select.csv", header=0, sep=","))
# x_row = S[:,-1]

# X = np.array(pd.read_csv("01_real_data/pdac_a.tsv", header=0, sep="\t"))
# print(len(X[0]))
# np.savetxt("X_genes_real.csv", X[:,0], fmt='%s', delimiter=",")


# files = os.listdir("00_synthetic/00_PDAC_A/00_No_X_Noise/03_clust/01_x")
# print(files)
# np.savetxt("NoNoise_3clust_X_files.csv", files, fmt="%s", delimiter=",")

files = os.listdir("00_synthetic/00_PDAC_A/00_No_X_Noise/03_clust/01_x/")
print(files)
np.savetxt("xFiles_clean_3clust.csv", files, fmt="%s", delimiter=",")