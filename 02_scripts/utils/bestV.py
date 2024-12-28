import pandas as pd
import numpy as np

V_err = pd.read_csv("resultsV_admm3_grid.csv", sep=",", header=None)
print(V_err.idxmin())