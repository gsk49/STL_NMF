import pandas as pd
import numpy as np

C = pd.read_csv('00_synthetic/ruitao/square_2x2/C.csv', header=0, index_col=0)
SC = C.index
clusters = ["CT1", "CT2", "CT3", "CT4"]

C = np.array(C)
CT = np.argmax(C, axis=1)
CT = np.array([clusters[i] for i in CT])

meta = np.stack([SC, CT], axis=1)
meta = pd.DataFrame(meta, columns=['SC', 'CT'], )

meta.to_csv('00_synthetic/ruitao/square_2x2/meta.csv', index=False)