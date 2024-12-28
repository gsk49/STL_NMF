import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


xerr = np.array(pd.read_csv("01_real_data/xerr.csv", sep=",", header=None))
xerr = np.squeeze(xerr)
print(xerr)

plt.plot(range(len(xerr)), xerr)
plt.show()