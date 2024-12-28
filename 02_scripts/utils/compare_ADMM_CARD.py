import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from matplotlib.colors import ListedColormap


# Marker Genes
X = pd.read_csv("01_real_data/X_Rep12_MOB_count_matrix_select.csv", sep=",", header=0)
X.set_index("Genes", inplace=True)
X_locs = X.columns
X = np.array([X.loc["Penk"], X.loc["Apold1"], X.loc["S100a5"], X.loc["Cdhr1"]], dtype=float)
X /= sum(X)+np.ones(len(X.T))*.00001
for i in range(len(X)):
    X[i,:] /= max(X[i,:])+.00001
np.savetxt("marker_genes_MOB.csv", X, fmt="%f", delimiter=",")




# card
card_V = pd.read_csv("01_real_data/MOB_CARD.csv", sep=",", header=0, index_col=0)
card_locs = card_V.index
card_V = np.array(card_V).T
for i in range(len(card_V)):
    card_V[i,:] /= max(card_V[i,:])+.00001
card_B = pd.read_csv("01_real_data/MOB_B_CARD.csv", sep=",", header=0, index_col=0)
card_genes = card_B.index
card_B = np.array(card_B)

card_x, card_y = zip(*[k.split(',') for k in card_locs.tolist()])
card_locs2 = [f'{x}{"x"}{y}' for x, y in zip(card_x, card_y)]

card_x = np.array(card_x, dtype=float)
card_y = np.array(card_y, dtype=float)

# ADMM2 files
# admm_C = np.array(pd.read_csv("01_real_data/ADMM2_Cinit_C.csv", sep=",", header=None))
# admm_V = np.array(pd.read_csv("01_real_data/ADMM2_Cinit_V.csv", sep=",", header=None))

# admm_C = np.array(pd.read_csv("01_real_data/C_admm_bestparams.csv", sep=",", header=None))
# admm_V = np.array(pd.read_csv("01_real_data/V_admm_bestparams.csv", sep=",", header=None))
admm_C = np.array(pd.read_csv("./c2_admm2_oct28.csv", sep=",", header=None))
admm_V = np.array(pd.read_csv("./v2_admm2_oct28.csv", sep=",", header=None))

print(np.sum(admm_V, axis=1))



admm_N = np.identity(len(admm_C[0]))
admm_N /= sum(admm_C)+np.ones(sum(admm_C).shape)*.0000000001
# for i in range(len(admm_C[0])):
#     admm_N[i,i] /= sum(admm_C[:,i])

admm_N_inv = np.identity(len(admm_C[0]))
for i in range(len(admm_C[0])):
    admm_N_inv[i,i] /= admm_N[i,i] +.00000001

# ADMM2 finals

# GT S,B
S = pd.read_csv("01_real_data/S_OB_6_runs_processed_seurat.dge_filtered_select.csv", sep=",", header=0, index_col=0)
S_index = S.index
S = np.array(S)
B = pd.read_csv("B_OB.csv", sep=",", header=None)
B.index = S_index
B2 = B[B.index.isin(card_genes)]
B2 = np.array(B2)
B = np.array(B)

admm_B = np.matmul(np.matmul(S, admm_C), admm_N)
# admm_B = pd.DataFrame(admm_B, index=S_index)
# admm_B = admm_B[admm_B.index.isin(card_genes)]
admm_B = np.array(admm_B)
# admm_V = np.matmul(admm_N_inv, admm_V)
# admm_V /= sum(admm_V)+np.ones(len(admm_V.T))*.00001
for i in range(len(admm_V)):
    admm_V[i,:] /= max(admm_V[i,:])+.00001

# ### Comparisons
# print("real B vs admm SCN")
# print(np.linalg.norm(B-admm_B)/np.linalg.norm(B))
# print("real B vs card B")
# print(np.linalg.norm(B2-card_B)/np.linalg.norm(B2))


### PLOTS

locs = np.array(pd.read_csv("01_real_data/location_Rep12_MOB_count_matrix_loc_sort.csv", header=0, index_col=0))
x = locs[:,0]
y = locs[:,1]

original_cmap = plt.get_cmap("Reds")
colors = original_cmap(np.arange(original_cmap.N))
colors[0] = [1, 1, 1, 1]  # Set the first color to white
custom_cmap = ListedColormap(colors)


plt.subplot(5,3,1)
plt.scatter(x, y, c=X[0,:], cmap=custom_cmap, s=10, marker="s")
plt.colorbar()
plt.subplot(5,3,4)
plt.scatter(x, y, c=X[1,:], cmap=custom_cmap, s=10, marker="s")
plt.colorbar()
plt.subplot(5,3,7)
plt.scatter(x, y, c=X[2,:], cmap=custom_cmap, s=10, marker="s")
plt.colorbar()
plt.subplot(5,3,10)
plt.scatter(x, y, c=X[3,:], cmap=custom_cmap, s=10, marker="s")
plt.colorbar()

plt.subplot(5,3,2)
plt.scatter(x, y, c=admm_V[4,:], cmap=custom_cmap, s=10, marker="s")
plt.colorbar()
plt.subplot(5,3,5)
plt.scatter(x, y, c=admm_V[3,:], cmap=custom_cmap, s=10, marker="s")
plt.colorbar()
plt.subplot(5,3,8)
plt.scatter(x, y, c=admm_V[0,:], cmap=custom_cmap, s=10, marker="s")
plt.colorbar()
plt.subplot(5,3,11)
plt.scatter(x, y, c=admm_V[1,:], cmap=custom_cmap, s=10, marker="s")
plt.colorbar()
plt.subplot(5,3,14)
plt.scatter(x, y, c=admm_V[2,:], cmap=custom_cmap, s=10, marker="s")
plt.colorbar()

plt.subplot(5,3,3)
plt.scatter(card_x, card_y, c=card_V[0,:], cmap=custom_cmap, s=10, marker="s")
plt.colorbar()
plt.subplot(5,3,6)
plt.scatter(card_x, card_y, c=card_V[1,:], cmap=custom_cmap, s=10, marker="s")
plt.colorbar()
plt.subplot(5,3,9)
plt.scatter(card_x, card_y, c=card_V[3,:], cmap=custom_cmap, s=10, marker="s")
plt.colorbar()
plt.subplot(5,3,12)
plt.scatter(card_x, card_y, c=card_V[2,:], cmap=custom_cmap, s=10, marker="s")
plt.colorbar()
# plt.subplot(5,3,15)
# plt.scatter(card_x, card_y, c=card_V[4,:], cmap=custom_cmap, s=10, marker="s")
# plt.colorbar()

plt.show()


### COMPARE X

X = pd.read_csv("01_real_data/X_Rep12_MOB_count_matrix_select.csv", sep=",", header=0, index_col=-1)
X = X[X.index.isin(card_genes)]
X = X.loc[:, X.columns.isin(card_locs2)]
marker_g = np.array([X.loc["Penk"], X.loc["Apold1"], X.loc["S100a5"], X.loc["Cdhr1"]], dtype=float)
X = np.array(X)
marker_g /= sum(marker_g)+np.ones(len(marker_g.T))*.00001
for i in range(len(marker_g)):
    marker_g[i,:] /= max(marker_g[i,:])+.00001


admm_B = pd.DataFrame(admm_B, index=S_index)
admm_B = np.array(admm_B[admm_B.index.isin(card_genes)])

# print(X.shape)
# print(admm_B.shape)
# print(admm_V.shape)
admm_V = pd.read_csv("01_real_data/ADMM2_seurat_V.csv", sep=",", header=None)
admm_V.columns = X_locs
# admm_V = np.array(admm_V[admm_V.columns.isin(card_locs)])
admm_V = admm_V.loc[:, admm_V.columns.isin(card_locs2)]
admm_V = np.array(admm_V)
admm_V = np.matmul(admm_N_inv, admm_V)


admm_x = np.matmul(admm_B, admm_V)
# print(admm_x)
# print(np.linalg.norm(admm_x-X))


card_x = np.matmul(card_B, card_V)
# print(card_x)
# print(np.linalg.norm(card_x-X))


### COMPARE C
admm_C = np.array(pd.read_csv("01_real_data/C_admm_bestparams.csv", sep=",", header=None))

real_C = np.array(pd.read_csv("01_real_data/C_GSE121891_Figure_2_metadata_filtered_sort.csv", header=0, sep=","))[:,1]
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

### sees which cols correspond to which cols
distance_matrix = cdist(admm_C.T, C2.T, 'euclidean')
row_ind, col_ind = linear_sum_assignment(distance_matrix)
for i in range(len(row_ind)):
    print(f"Column {row_ind[i]} of Matrix C is matched with Column {col_ind[i]} of Matrix C2")

### Reorders clusters to be aligned
df = pd.DataFrame({'x': col_ind, 'y': row_ind})
df = df.sort_values('x')
df = np.array(df)
df = df[:, 1]

admm_C = admm_C[:, df]

admm_N = np.identity(len(admm_C[0]))
real_N = np.identity(len(admm_C[0]))
for i in range(len(admm_C[0])):
    admm_N[i,i] /= sum(admm_C[:,i])
    real_N[i,i] /= sum(C2[:,i])

real_B = np.matmul(np.matmul(S, C2),real_N)
admm_B = np.matmul(np.matmul(S,admm_C),admm_N)

# print(np.linalg.norm(admm_C-C2))
# print(np.linalg.norm(admm_B-real_B))




### Compare B

### sees which cols correspond to which cols
B = pd.read_csv("B_OB.csv", sep=",", header=None)
B.index = S_index
B2 = B[B.index.isin(card_genes)]
B2 = np.array(B2)

# print(B2.shape)
# print(card_B.shape)

distance_matrix = cdist(card_B.T, B2.T, 'euclidean')
row_ind, col_ind = linear_sum_assignment(distance_matrix)
# for i in range(len(row_ind)):
    # print(f"Column {row_ind[i]} of Matrix C is matched with Column {col_ind[i]} of Matrix C2")

### Reorders clusters to be aligned
df = pd.DataFrame({'x': col_ind, 'y': row_ind})
df = df.sort_values('x')
df = np.array(df)
df = df[:, 1]

card_B = card_B[:, df]
admm_test_V = np.vstack((admm_V[4,:], admm_V[3,:], admm_V[2,:], admm_V[1,:]))
print(np.linalg.norm(marker_g-admm_test_V)/np.linalg.norm(marker_g))
print(np.linalg.norm(marker_g-card_V[:-1,:])/np.linalg.norm(marker_g))

print(sum(admm_C))