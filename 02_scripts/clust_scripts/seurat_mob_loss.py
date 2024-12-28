import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


## Load in cluster results
c_ref = pd.read_csv("01_real_data/SEURAT_MOB_louvain.csv", header=0, sep=",")
c_ref = (np.array(c_ref)[:,1]).T
# c_ref = pd.read_csv("01_real_data/SEURAT_MOB_mlRefinement.csv", header=0, sep=",")
# c_ref = (np.array(c_ref)[:,1]).T
# c_ref = pd.read_csv("01_real_data/SEURAT_MOB_slm.csv", header=0, sep=",")
# c_ref = (np.array(c_ref)[:,1]).T
# c_ref = pd.read_csv("01_real_data/SEURAT_MOB_leiden.csv", header=0, sep=",")
# c_ref = (np.array(c_ref)[:,1]).T



## Change Vector to Matrix for multiplication
# # Standard
C = np.zeros((c_ref.size, c_ref.max()+1))
C[np.arange(c_ref.size), c_ref] = 1
# # LEIDEN
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


## Only needs to be run once, sees which cols correspond to which cols
distance_matrix = cdist(C.T, C2.T, 'euclidean')
row_ind, col_ind = linear_sum_assignment(distance_matrix)
for i in range(len(row_ind)):
    print(f"Column {row_ind[i]} of Matrix C is matched with Column {col_ind[i]} of Matrix C2")


## Rearrange C cols to match Appropriate Cell-Types
## MOB/PDAC Seurat/SC3
C = np.array([C[:,3], C[:,0], C[:,2], C[:,4], C[:,1]]).T
# C = np.array([C[:,3], C[:,0], C[:,2], C[:,4], C[:,1]]).T
# C = np.array([C[:,3], C[:,0], C[:,2], C[:,4], C[:,1]]).T
# C = np.array([C[:,4], C[:,0], C[:,2], C[:,3], C[:,1]]).T

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
B = np.matmul(S, np.matmul(C, n))
B = np.array(B, dtype=float)

## Debugging and seeing results
# print(B)
# print(real_B)

print(np.linalg.norm(B-real_B)/np.linalg.norm(real_B))
print(np.linalg.norm(C-C2)/np.linalg.norm(C2))


# np.savetxt("C_true_MOB.csv", C2, delimiter=",")
# np.savetxt("B_sc3_MOB.csv", B, delimiter=",")
