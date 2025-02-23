#!/usr/bin/python3
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import time
import random
# import plot
from scipy.signal import convolve2d
import time


def gkernel(l=3, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def get_v(alpha=0.7, beta=0.15, add_noise=False, smooth=False, n_clust=3):

    
    # import the raw data
    # x = pd.read_csv('PDAC\\PDAC-A-AX.csv', sep=',', header=0) # ST
    # annote = pd.read_csv('PDAC\\Figure4A_layer_annote.csv', sep=',', header=0) # ST
    x = pd.read_csv('./01_real_data/pdac_a.tsv', sep="\t", header=0)
    annote = pd.read_csv('./01_real_data/Figure4A_layer_annote.csv', sep=",", header=0)
        
    # 

    if n_clust == 3:
        keys = ['Acinar cells', 'Cancer clone A', 'Cancer clone B']
        gamma = 1 - alpha - beta
    elif n_clust == 5:
        keys = ['Acinar cells', 'Cancer clone A', "Endocrine cells", "Fibroblasts", "Ductal - terminal ductal like"]
        a = np.random.randint(100)
        b = np.random.randint(100)
        g = np.random.randint(100)
        d = np.random.randint(100)
        l = np.random.randint(100)
        denom = np.exp(-a)+np.exp(-b)+np.exp(-g)+np.exp(-d)+np.exp(-l)
        alpha = np.exp(-a)/denom
        beta = np.exp(-b)/denom
        gamma = np.exp(-g)/denom
        delta = np.exp(-d)/denom
        lam = np.exp(-l)/denom

    # plot annotation
    loc0 = np.array(x.drop('Genes', axis=1, inplace=False).keys())
    loc = np.zeros((len(loc0), 3)) ## K CLUST
    n_loc = len(loc)
    n_type = len(keys)

    v = np.zeros([n_type, n_loc], dtype=float)
    intensity_vec = np.zeros([3, n_loc], dtype=float)

    for i in range(n_loc):
        loc[i, 0:2] = np.array(loc0[i].split('x'), dtype=float)
        loc[i, 2] = 0  # for later use


    mean = np.zeros((n_loc, n_type)) # V
    for i in range(n_loc):
        if i not in range(len(annote["Region"])):
            break
        ## f = annote[(annote['x']==loc[i,0])&(annote['y']==loc[i,1])]
        ## region = ""
        ## if not f.empty:
        ##     region = f.iloc[0]['Region']

        # print(annote['Region'][i])
        if annote["Region"][i] == 'Duct Epithelium' or annote["Region"][i] == 'Pancreatic':
        # if region == 'Duct Epithelium' or region == 'Pancreatic':
            loc[i, 2] = 1 # for later use
            if len(keys) == 3:
                mean[i, :] = np.array([gamma, beta, alpha])
            elif len(keys) == 5:
                mean[i, :] = np.array([beta, delta, alpha, lam, gamma])
            # intensity[annote['x'][i], annote['y'][i], 0] = 1
            intensity_vec[0, i] = 1
        if annote['Region'][i] == 'Stroma':
        # if region == 'Stroma':
            loc[i, 2] = 2  # for later use
            if len(keys) == 3:
                mean[i, :] = np.array([alpha, gamma, beta])
            elif len(keys) == 5:
                mean[i, :] = np.array([gamma, lam, beta, delta, alpha])
            # intensity[annote['x'][i], annote['y'][i], 1] = 1
            intensity_vec[1, i] = 1
        if annote['Region'][i] == 'Cancer':
        # if region == 'Cancer':
            loc[i, 2] = 3  # for later use
            if len(keys) == 3:
                mean[i, :] = np.array([beta, alpha, gamma])
            elif len(keys) == 5:
                mean[i, :] = np.array([delta, alpha, lam, gamma, beta])
            # intensity[annote['x'][i], annote['y'][i], 2] = 1
            intensity_vec[2, i] = 1

    v = mean.T


    if add_noise==True:
        v += np.random.normal(loc=0, scale=0.05, size=v.shape)
        v[v < 0] = 0
        intensity_vec += np.random.normal(loc=0, scale=0.05, size=intensity_vec.shape)
        intensity_vec[intensity_vec < 0] = 0
        intensity_vec[intensity_vec > 1] = 1
    # construct a matrix
    space = np.zeros((annote['x'].max() + 1, annote['y'].max() + 1, n_clust)) # matrix version of V
    intensity = np.zeros((annote['x'].max() + 1, annote['y'].max() + 1, 3)) # matrix version if intensity
    for j in range(n_loc):
        if j not in range(len(annote["x"])):
            break
        space[annote['x'][j], annote['y'][j], :] = v[:,j]
        intensity[annote['x'][j], annote['y'][j], :] = intensity_vec[:, j]
    # smooth each cell type:
    if smooth == True:
        kernel = gkernel(l=40, sig=0.5)
        for j in range(n_type):
            space[:,:,j] = convolve2d(space[:,:,j], kernel, mode='same')
        for j in range(3):
            intensity[:,:,j] = convolve2d(intensity[:,:,j], kernel, mode='same')
        # save the space into v
        for j in range(n_loc):
            if j not in range(len(annote['x'])):
                break
            v[:, j] = space[annote['x'][j], annote['y'][j], :]
        v[v < 0] = 0

    # normalize v
    for k in range(n_loc):
        v[:, k] = v[:, k] / (np.sum(v[:, k])+np.exp(-10))

    # print(annote)
    # get vector of intensity
    for j in range(n_loc):
        if j not in range(len(annote['x'])):
            break
        intensity_vec[:,j] = intensity[annote['x'][j], annote['y'][j], :]

    # plt.imshow(intensity)
    # plt.axis('off')
    # plt.show()

    return v, intensity_vec


def get_x(v, num_x = 10, sigma=0.5):
    # x = pd.read_csv('PDAC\\PDAC-A-AX_simulate.csv', sep=',', header=0)  # ST
    ##
    x = pd.read_csv('./01_real_data/pdac_a.tsv', sep="\t", header=0)
    locs = list(x.columns)[1:]
    # np.savetxt("locs.csv", locs, '%s', delimiter=",")
    # print(locs)
    x0 = np.array(x.drop('Genes', axis=1))  # ST
    x00 = np.array(x.values[:, 0])
    # print(x)
    
##
    # extract needed cell types
    # b = pd.read_csv('./01_real_data/pdac_a_ref.tsv', sep="\t", header=0)
    b = pd.read_csv('./01_real_data/B_GSE111672_PDAC-A-indrop-filtered-expMat_ave.csv', sep=",", header=0)
    s = pd.read_csv('./01_real_data/S_GSE111672_PDAC-A-indrop-filtered-expMat_select.csv', sep=",", header=0)
    b00 = np.array(s.values[:, -1])
    b["Genes"] = b00
    # print(b)
    # print(b00)
    keys = ['Acinar cells', 'Cancer clone A', "Endocrine cells", "Fibroblasts", "Ductal - terminal ductal like"]
    # keys = ['Acinar cells', 'Cancer clone A', 'Cancer clone B']

    ## I SHOULD CHANGE THIS. LOOP IS BAD
    merged_data = pd.merge(x, b, on=x.columns[0], how='inner')
    # Extract common labels
    c_labs = np.unique(merged_data[x.columns[0]].values)

    # c_labs = [value for value in x00 if value in b00]

    b_part = pd.DataFrame(columns=keys)
    for i in range(len(keys)):
        b_part[keys[i]] = b[keys[i]]
    # print(b_part)
    b0 = np.array(b_part)
    b0 = b0[np.isin(b00, c_labs)]
    np.savetxt('b.csv', b0[:,:], fmt='%d', delimiter=',')
    #
    bv = np.matmul(b0, v)
    # print(bv.shape)

    # xx = np.zeros((num_x, x0.shape[0], x0.shape[1]), dtype=int)
    xx = np.zeros((num_x, bv.shape[0], bv.shape[1]), dtype=int)

    for i in range(num_x):
        xx[i, :, :] = bv + np.random.normal(0, sigma, size=bv.shape)
    xx[xx<0]=0
    x0 = x0[np.isin(x00, c_labs)]
    # print(x0.shape)

    # np.savetxt('x.csv', x0[:,:], fmt='%d', delimiter=',')

    return xx


def main():
    v, intensity_vec = get_v(alpha=0.7, beta=0.15, add_noise=True, smooth=True)
    #print(v)
    # np.savetxt('v_sim.txt', v)
    #np.savetxt('intensity_sim.txt', intensity_vec)

    num_x = 1 # number of simulated X
    sigma = 0 # noise level

    x_sim = get_x(v, num_x=num_x, sigma=sigma)
    # for i in range(num_x):
       # np.savetxt('../000_synthetic/xx' + str(i) + '_sim.csv', x_sim[i, :, :], fmt='%d', delimiter=',')

# main()
