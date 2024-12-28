import pandas as pd
import numpy as np



# i1 = np.array(pd.read_csv("00_synthetic/00_PDAC_A/00_No_X_Noise/05_clust/02_v/intensity/i23_1.csv", sep=" ", header=None))
# i2 = np.array(pd.read_csv("00_synthetic/00_PDAC_A/00_No_X_Noise/05_clust/02_v/intensity/i3_1.csv", sep=" ", header=None))

# print(sum(i1.T))

# sc_2_ct = np.array(pd.read_csv("./01_real_data/new_cluster_number.csv", sep=",", header=0))
# sc_2_ct = np.array([sc_2_ct[:, 1], sc_2_ct[:, 2]]).T
# np.savetxt("sc_ann.tsv", sc_2_ct, fmt="%s", delimiter="\t")

# S = pd.read_csv("./01_real_data/S_GSE111672_PDAC-A-indrop-filtered-expMat_select.csv", sep=",", header=0, index_col=-1).T
# print(S)
# S.to_csv("S_T.tsv", sep="\t", index=True, header=True)

# X = pd.read_csv("X_real.tsv", sep="\t", header=0, index_col=0).T
# genes = np.array(pd.read_csv("Genes.csv", header=None, sep=" ").squeeze())
# locs = np.array(pd.read_csv("./01_real_data/locs.csv", header=None, sep=" ").squeeze())

# X.index = genes
# X.columns = locs

# X.to_csv("XT_real.tsv", sep="\t", index=True, header=True)

B = pd.read_csv("./01_real_data/B_GSE111672_PDAC-A-indrop-filtered-expMat_ave.csv", sep=",", header=0)
B = np.array(B)[:,:]
B = np.array([B[:,0],B[:,1],B[:,7],B[:,9],B[:,6]]).T
np.savetxt("b_pdac_05clust.csv", B, fmt="%f", delimiter=",")