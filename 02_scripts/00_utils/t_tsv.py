# Transposes a TSV and keeps index/column names

import pandas as pd
import numpy as np


x = pd.read_csv("./zzz_outputs/S_T.tsv", sep="\t", header=0, index_col=0)
x = x.T
x.columns = x.columns
x.index = x.index
# np.savetxt("00_synthetic/00_PDAC_A/00_No_X_Noise/03_clust/03_x_tsv/"+name[:-3]+"tsv", x, fmt="%d", delimiter="\t")
x.to_csv("S.tsv", sep="\t", index=True, header=True)