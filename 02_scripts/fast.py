#!/usr/bin/python3
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import time


def plot_real_v(v, loc, keys, type, title):
    lower = 0
    upper = np.max(v)
    loc1 = loc[:, 0]
    loc2 = loc[:, 1]

    n1 = 4
    n2 = 5

    # fig, axes = plt.subplots(nrows=n1, ncols=n2)

    for i in range(len(type)):
        plt.subplot(n1, n2, i+1)
        plt.axis('off')
        plt.axis('equal')
        im = plt.scatter(loc1, loc2, s=10, marker='o', c=v[type[i], :], cmap='Reds', vmin = lower, vmax = upper)
        plt.title(keys[type[i]], loc='left')

    # fig.colorbar(im, ax=axes.ravel().tolist())

    # plt.suptitle(title)
    # plt.scatter(loc1, loc2, marker='o', c=cancer_a_gene)
    # plt.title(str(np.round( np.corrcoef(cancer_a_dist, cancer_a_gene)[0,1], 3)))
    plt.show()





def model_nmf(x0, b0, D, A, lam1, lam2):
    n_loc = x0.shape[1]
    n_type = b0.shape[1]


    # constant
    Ic = np.ones((n_type, 1))
    Ip = np.ones((n_loc, 1))
    IcIpT = np.matmul(Ic, Ip.T)
    IcIcT = np.matmul(Ic, Ic.T)
    btb = np.matmul(b0.T, b0)

    # initilize v
    v = np.ones([n_type, n_loc], dtype=float)
    for i in range(n_loc):
        v[:, i] = v[:, i] / np.sum(v[:, i])

    # loop
    for i in range(10000):
        v_old = v.copy()

        upper = np.matmul(b0.T, x0) + lam1 * np.matmul(v, A) + lam2 * IcIpT
        lower = np.matmul(btb, v) + lam1 * np.matmul(v, D) + lam2 * np.matmul(IcIcT, v)
        v = upper * v / (lower + 1e-8)

        # normalize v at each step
        for k in range(n_loc):
            v[:, k] = v[:, k] / np.sum(v[:, k])

        err = np.linalg.norm(v - v_old, ord='fro')
        print(i, err)
        if err <= 1e-3:
            break

    for k in range(n_loc):
        v[:, k] = v[:, k] / np.sum(v[:, k])


    return v, np.linalg.norm(x0 - np.matmul(b0, v), ord='fro')/n_loc



# import data
# st_selected = pd.read_csv('data\\PDAC-B\\GSM3405534_PDAC-B-ST1-filtered_4.csv', sep=',', header=0) # X
# st_loc = pd.read_csv('data\\PDAC-B\\GSM3405534_PDAC-B-ST1-filtered_loc.csv', sep=',', header=0)
# sc_ave = pd.read_csv('data\\PDAC-B\\GSE111672_PDAC-B-indrop-filtered-expMat_ave.csv', sep=',', header=0) # B
# sc_meta = pd.read_csv('data\\PDAC-B\\GSE111672_PDAC-B-indrop-filtered-expMat_meta.csv', sep=',', header=0) # B

# st_selected = pd.read_csv("./00_synthetic/00_PDAC_A/02_X_shuf/X_0.tsv", sep="\t", header=0)
# S = pd.read_csv("./01_real_data/S_GSE111672_PDAC-A-indrop-filtered-expMat_select.csv", sep=",", header=0, index_col=-1)
# merged_data = pd.merge(st_selected, S, on=st_selected.columns[0], how='inner')
# # Extract common labels
# labs1 = merged_data[st_selected.columns[0]].values
# labs2 = np.array(st_selected.values[:,0])
# st_selected = st_selected[np.isin(labs2, labs1)]

st_selected = pd.read_csv("./00_synthetic/00_PDAC_A/01_X_Noise/05_clust/01_x/x0_0_0_sim.csv", sep=",", header=None)

st_loc = pd.read_csv("./01_real_data/locs_titled.csv", header=0)
sc_ave = pd.read_csv("./03_results/b_pdac_05clust.csv", header=None)
# sc_meta = pd.read_csv("03_clustering/C_GSE121891_Figure_2_metadata_filtered_sort.csv", header=0)


# print(st_selected)
# print(sc_ave)


# x0 = np.array( st_selected.drop( labels='Genes', axis=1) ,dtype=float )
x0 = np.array( st_selected,dtype=float )
b0 = np.array(sc_ave)


n_loc = x0.shape[1]
n_gene = b0.shape[0]
n_type = b0.shape[1]


loc = np.zeros((n_loc, 3))
loc[:, 0] = st_loc['x']
loc[:, 1] = st_loc['y']


# compute graph laplacian
mu_intensity = 0
mu_distance = 1
sigma = 1   # the larger the smoother
A = np.zeros((n_loc,n_loc))
D = np.zeros((n_loc,n_loc))
for i in range(n_loc):
    for j in range(n_loc):
        distance = (loc[i, 0]-loc[j, 0])**2 + (loc[i, 1]-loc[j, 1])**2
        # intensity_dis = np.linalg.norm(intensity_vec[:, i] - intensity_vec[:, j], 2)
        if distance <= 99999:
            # A[i, j] = np.exp(-( mu_distance * distance + mu_intensity * intensity_dis ) / sigma)
            A[i, j] = np.exp(-(mu_distance * distance) / sigma)
for i in range(n_loc):
    A[i,i] = 1
    D[i,i] = np.sum(A[i,:])
L = D-A

print("start")
v, mse = model_nmf(x0, b0, D, A, lam1=320, lam2=1)

np.savetxt("FAST_realx_simv.csv", v, delimiter=",", fmt="%f")


# output v
# np.savetxt('results\\v_real_nmf.txt', v, fmt='%.8f')

# keys, keynum =  np.unique(sc_meta['CellType'], return_counts=True)
# type = np.arange(0,n_type).tolist()
# plot_real_v(v, loc, keys, type, title='PDAC-B')
