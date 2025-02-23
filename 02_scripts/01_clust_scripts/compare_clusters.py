import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

cl1 = np.array(pd.read_csv("./01_real_data/sc3_PDAC_20C.csv", header=0))[:,1]
cl2 = np.array(pd.read_csv("./01_real_data/seurat_PDAC_A_C.csv", header=0))

cl1 = np.array(cl1, dtype=int) - np.ones(len(cl1))
# range1 = [3,0,2,4,1]
# range2 = [0,1,3,4,2]
range1 = [19,7,6,3,0,2,1,12,18,10,4,13,16,9,11,5,8,17,14,15]
range2 = [3,13,14,7,0,8,12,4,19,18,10,11,6,9,15,2,17,1,16,5]
plt.subplot(2,1,1)
plt.plot(range(len(cl1)), range1[cl1[:]], "o")
plt.title("Seurat")
plt.subplot(2,1,2)
plt.plot(range(len(cl2)), range2[int(cl2[:,1])], "o")
plt.title("SC3")

plt.show()