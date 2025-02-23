import pandas as pd
import numpy as np

V = pd.read_csv('00_synthetic/ruitao/square_2x2/V.csv', header=0, index_col=0)
locs = V.index
B = pd.read_csv('00_synthetic/ruitao/square_2x2/B.csv', header=0, index_col=0)
genes = B.index

V = np.array(V).T
B = np.array(B)

X = np.matmul(B, V)
X = pd.DataFrame(X, index=genes, columns=locs)

X.to_csv('00_synthetic/ruitao/square_2x2/X.tsv', sep='\t')