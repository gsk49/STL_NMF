import pandas as pd
import numpy as np

X = np.array(pd.read_csv("01_real_data/pdac_a.tsv", sep="\t", header=None, dtype=str))
colnames = X[0,1:]
rownames = X[:,0]
X = X[1:,1:]
X = np.array(X, dtype=int)
rownames = rownames.reshape(-1,1)
colnames = colnames.reshape(1,-1)

for i in range(50):
    np.random.shuffle(X)
    simX = np.vstack([colnames, X])
    simX = np.hstack([rownames, simX])
    np.savetxt("00_synthetic/00_PDAC_A/02_X_shuf/X_"+str(i)+".tsv", simX, fmt="%s", delimiter="\t")