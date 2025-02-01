import pandas as pd
import numpy as np

X = pd.read_csv("zzz_outputs/X_real.tsv", header=0, index_col=0, sep='\t')

cells = X.columns
x, y = np.array([int(cell.split(',')[0]) for cell in cells], dtype=str), np.array([int(cell.split(',')[1]) for cell in cells], dtype=str)
print(cells)
print(x)
print(y)
print(np.char.add(np.char.add(x, 'x'), y))

mat = np.stack([np.char.add(np.char.add(x, 'x'), y), x, y], axis=1)
np.savetxt("real_loc.csv", mat, fmt="%s", delimiter=',')