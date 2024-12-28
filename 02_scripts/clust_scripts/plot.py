import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, OPTICS

### Cluster Cells by Gene Makeup

# Load in PDAC A data
# B = pd.read_csv("03_clustering/S_GSE111672_PDAC-A-indrop-filtered-expMat_select.csv", header=0, sep=',')
# B = np.array(B.values.T)[:-1, :]

B = pd.read_csv("03_clustering/S_OB_6_runs_processed_seurat.dge_filtered_select.csv", header=0, sep=',')
B = np.array(B.values.T)[1:, :]

# Create indices
x = range(len(B))

# Very simple Baseline

kmeans = KMeans(n_clusters=5, n_init="auto").fit(B)
plt.subplot(2,2,1)
plt.plot(x, kmeans.labels_, "o")
np.savetxt("kmeans_MOB.csv", kmeans.labels_, delimiter=",")

agg = AgglomerativeClustering(n_clusters=5).fit(B)
plt.subplot(2,2,2)
plt.plot(x, agg.labels_, "o")
np.savetxt("agg_MOB.csv", agg.labels_, delimiter=",")

cl1 = np.array(pd.read_csv("01_real_data/sc3_PDAC_20C.csv", header=0))
cl2 = np.array(pd.read_csv("01_real_data/seurat_PDAC_A_C.csv", header=0))



plt.subplot(2,2,3)
plt.plot(range(len(cl1)), cl1[:,1], "o")
plt.title("SC3")
plt.subplot(2,2,4)
plt.plot(range(len(cl2)), cl2[:,1], "o")
plt.title("Seurat")

plt.show()
