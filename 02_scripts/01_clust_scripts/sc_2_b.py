import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


## Load in cluster results
c_ref = pd.read_csv("01_real_data/seurat_MOB_C.csv", header=0, sep=",")
c_ref = (np.array(c_ref)[:,1]).T
# c_ref = pd.read_csv("01_real_data/sc3_MOB_5c.csv", header=0, sep=",")
# c_ref = (np.array(c_ref)[:,1]).T
# c_ref = np.array(c_ref, dtype=int)
# c_ref = pd.read_csv("01_real_data/seurat_PDAC_A_C.csv", header=0, sep=",")
# c_ref = (np.array(c_ref)[:,1]).T
# c_ref = pd.read_csv("01_real_data/sc3_PDAC_20C.csv", header=0, sep=",")
# c_ref = (np.array(c_ref)[:,1]).T
# c_ref = np.array(c_ref, dtype=int)
### KMEANS and AGG
# c_ref = pd.read_csv("kmeans_MOB.csv", header=None, sep=",")
# c_ref = np.array(c_ref, dtype=int).T
# c_ref = pd.read_csv("agg_MOB.csv", header=None, sep=",")
# c_ref = np.array(c_ref, dtype=int).T
# c_ref = pd.read_csv("kmeans_pdac.csv", header=None, sep=",")
# c_ref = np.array(c_ref, dtype=int).T
# c_ref = pd.read_csv("agg_pdac.csv", header=None, sep=",")
# c_ref = np.array(c_ref, dtype=int).T


## Change Vector to Matrix for multiplication
# # seurat
C = np.zeros((c_ref.size, c_ref.max()+1))
C[np.arange(c_ref.size), c_ref] = 1
# sc3
# C = np.zeros((c_ref.size, c_ref.max()))
# C[np.arange(c_ref.size), c_ref-1] = 1

## Load in Other C result to ensure columns align
real_C = np.array(pd.read_csv("03_clustering/C_GSE121891_Figure_2_metadata_filtered_sort.csv", header=0, sep=","))[:,1]
# real_C_idx = np.array(pd.read_csv("01_real_data/new_cluster_number.csv", header=0, sep=","))[:,3]
# print(real_C_idx)
real_C_idx = []
for i in real_C:
    if i == "EPL-IN":
        real_C_idx.append(0)
    elif i == "GC":
        real_C_idx.append(1)
    elif i == "M/TC":
        real_C_idx.append(2)
    elif i == "OSs":
        real_C_idx.append(3)
    elif i == "PGC":
        real_C_idx.append(4)

real_C_idx = np.array(real_C_idx, dtype=int)

C2 = np.zeros((real_C_idx.size, real_C_idx.max()+1))
C2[np.arange(real_C_idx.size), real_C_idx] = 1
# C2 = np.zeros((real_C_idx.size, real_C_idx.max()))
# C2[np.arange(real_C_idx.size), real_C_idx-1] = 1

# print(C.shape)
# print(C2.shape)

# ## Only needs to be run once, sees which cols correspond to which cols
distance_matrix = cdist(C.T, C2.T, 'euclidean')
row_ind, col_ind = linear_sum_assignment(distance_matrix)
for i in range(len(row_ind)):
    print(f"Column {row_ind[i]} of Matrix C is matched with Column {col_ind[i]} of Matrix C2")


## Rearrange C cols to match Appropriate Cell-Types
## MOB/PDAC Seurat/SC3
C = np.array([C[:,3], C[:,0], C[:,2], C[:,4], C[:,1]]).T
# C = np.array([C[:,0], C[:,1], C[:,3], C[:,4], C[:,2]]).T
# C = np.array([C[:,19], C[:,7], C[:,6], C[:,3], C[:,0], C[:,2], C[:,1], C[:,12], C[:,18], C[:,10], C[:,4], C[:,13], C[:,16], C[:,9], C[:,11], C[:,5], C[:,8], C[:,17], C[:,14], C[:,15]]).T
# C = np.array([C[:,3], C[:,13], C[:,14], C[:,7], C[:,0], C[:,8], C[:,12], C[:,4], C[:,19], C[:,18], C[:,10], C[:,11], C[:,6], C[:,9], C[:,15], C[:,2], C[:,17], C[:,1], C[:,16], C[:,5]]).T
## Kmeans/Agg : MOB/PDAC
# C = np.array([C[:,0], C[:,1], C[:,4], C[:,2], C[:,3]]).T
# C = np.array([C[:,0], C[:,1], C[:,2], C[:,3], C[:,4]]).T
# C = np.array([C[:,13], C[:,6], C[:,11], C[:,16], C[:,0], C[:,4], C[:,19], C[:,18], C[:,2], C[:,1], C[:,12], C[:,14], C[:,15], C[:,5], C[:,9], C[:,7], C[:,3], C[:,10], C[:,17], C[:,8]]).T
# C = np.array([C[:,15], C[:,18], C[:,1], C[:,9], C[:,4], C[:,3], C[:,5], C[:,13], C[:,10], C[:,16], C[:,12], C[:,6], C[:,19], C[:,11], C[:,7], C[:,8], C[:,14], C[:,4], C[:,0], C[:,17]]).T

## Norm Error
# for i in range(len(C[0])):
#     C[:,i] /= sum(C[:,i])
#     C2[:,i] /= sum(C2[:,i])

## Error
n = np.identity(len(C[0]))
n2 = np.identity(len(C[0]))
for i in range(len(C[0])):
    n[i,i] /= sum(C[:,i])
    n2[i,i] /= sum(C2[:,i])

# print(C)
# print(C2)
## Load in S
S = pd.read_csv("03_clustering/S_OB_6_runs_processed_seurat.dge_filtered_select.csv", header=0, sep=",")
S = np.array(S)[:,1:]
# S = pd.read_csv("03_clustering/S_GSE111672_PDAC-A-indrop-filtered-expMat_select.csv", header=0, sep=",")
# S = np.array(S)[:,:-1]
S = np.array(S, dtype=float)

## Load in correct B matrix
real_B = pd.read_csv("03_clustering/B_OB_6_runs_processed_seurat.dge_filtered_ave.csv", header=0, sep=",")
# real_B = pd.read_csv("03_clustering/B_GSE111672_PDAC-A-indrop-filtered-expMat_ave.csv", header=0, sep=",")
real_B = np.array(real_B, dtype=float)

## Calculate our B, based on clustering results
# B = np.matmul(S,C)
B = np.matmul(S, np.matmul(C,n))
B = np.array(B, dtype=float)

## Debugging and seeing results
# print(B)
# print(real_B)

print(np.linalg.norm(B-real_B)/np.linalg.norm(real_B))
print(np.linalg.norm(C-C2)/np.linalg.norm(C2))

# print(np.linalg.norm(B-real_B))
# print(np.linalg.norm(C-C2))


np.savetxt("./03_clustering/res/N_true_MOB.csv", n2, delimiter=",")
np.savetxt("./03_clustering/res/C_true_MOB.csv", C2, delimiter=",")

# np.savetxt("./03_clustering/res/N_seurat_PDAC.csv", n, delimiter=",")
# np.savetxt("./03_clustering/res/B_seurat_PDAC.csv", B, delimiter=",")
