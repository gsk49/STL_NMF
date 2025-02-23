import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LogNorm, ListedColormap
import numpy as np
import torch

cardV = np.array(pd.read_csv("./CARD/card_real.csv", sep=",", header=0)).T
stlnmf = np.array(pd.read_csv("./DEEPNMF_real5.csv", sep=",", header=None))
stereoV = np.array(pd.read_csv("stereoscope/pdac_a_stereo/03_X_real/XT_real/W.2024-08-21175235.111338.tsv", sep="\t", header=0, index_col=0))
stereoV = np.c_[stereoV[:,0],stereoV[:,1],stereoV[:,7],stereoV[:,9],stereoV[:,6]].T
stereoV /= sum(stereoV)
realV = pd.read_csv("./01_real_data/pdac_a.tsv", sep="\t", header=0)
realV.set_index("Genes", inplace=True)
realV = np.array([realV.loc["REG3A"], realV.loc["TM4SF1"], realV.loc["PPY"], realV.loc["LUM"], realV.loc["KRT7"]], dtype=float) # CHGA
realV /= sum(realV)+np.ones(len(realV.T))*.001


loss_cardV = torch.tensor(np.array(cardV), dtype=torch.float32)
loss_stlnmf = torch.tensor(np.array(stlnmf), dtype=torch.float32)
loss_stereoV = torch.tensor(np.array(stereoV), dtype=torch.float32)
loss_realV = torch.tensor(np.array(realV), dtype=torch.float32)


# print("CARD:")
# print(torch.norm(loss_rV - loss_realV).item())
# print((torch.norm(loss_rV - loss_realV)/torch.norm(loss_realV)).item())

# print("dnmf:")
# print(torch.norm(loss_sV - loss_realV).item())
# print((torch.norm(loss_sV - loss_realV)/torch.norm(loss_realV)).item())

locs = np.array(pd.read_csv("./01_real_data/locs.csv", sep=",", header=None))


stlnmf_ct1 = np.zeros((22,24))
stlnmf_ct2 = np.zeros((22,24))
stlnmf_ct3 = np.zeros((22,24))
stlnmf_ct4 = np.zeros((22,24))
stlnmf_ct5 = np.zeros((22,24))

card_ct1 = np.zeros((22,24))
card_ct2 = np.zeros((22,24))
card_ct3 = np.zeros((22,24))
card_ct4 = np.zeros((22,24))
card_ct5 = np.zeros((22,24))

stereo_ct1 = np.zeros((22,24))
stereo_ct2 = np.zeros((22,24))
stereo_ct3 = np.zeros((22,24))
stereo_ct4 = np.zeros((22,24))
stereo_ct5 = np.zeros((22,24))

real_ct1 = np.zeros((22,24))
real_ct2 = np.zeros((22,24))
real_ct3 = np.zeros((22,24))
real_ct4 = np.zeros((22,24))
real_ct5 = np.zeros((22,24))

for _ in range(len(stlnmf)):
    for j in range(len(stlnmf[0])):
        stlnmf_ct1[locs[j][0]-7][locs[j][1]-7] = stlnmf[0][j]
        stlnmf_ct2[locs[j][0]-7][locs[j][1]-7] = stlnmf[1][j]
        stlnmf_ct3[locs[j][0]-7][locs[j][1]-7] = stlnmf[2][j]
        stlnmf_ct4[locs[j][0]-7][locs[j][1]-7] = stlnmf[3][j]
        stlnmf_ct5[locs[j][0]-7][locs[j][1]-7] = stlnmf[4][j]

        card_ct1[locs[j][0]-7][locs[j][1]-7] = cardV[0][j]
        card_ct2[locs[j][0]-7][locs[j][1]-7] = cardV[1][j]
        card_ct3[locs[j][0]-7][locs[j][1]-7] = cardV[2][j]
        card_ct4[locs[j][0]-7][locs[j][1]-7] = cardV[3][j]
        card_ct5[locs[j][0]-7][locs[j][1]-7] = cardV[4][j]

        stereo_ct1[locs[j][0]-7][locs[j][1]-7] = stereoV[0][j]
        stereo_ct2[locs[j][0]-7][locs[j][1]-7] = stereoV[1][j]
        stereo_ct3[locs[j][0]-7][locs[j][1]-7] = stereoV[2][j]
        stereo_ct4[locs[j][0]-7][locs[j][1]-7] = stereoV[3][j]
        stereo_ct5[locs[j][0]-7][locs[j][1]-7] = stereoV[4][j]

        real_ct1[locs[j][0]-7][locs[j][1]-7] = realV[0][j]
        real_ct2[locs[j][0]-7][locs[j][1]-7] = realV[1][j]
        real_ct3[locs[j][0]-7][locs[j][1]-7] = realV[2][j]
        real_ct4[locs[j][0]-7][locs[j][1]-7] = realV[3][j]
        real_ct5[locs[j][0]-7][locs[j][1]-7] = realV[4][j]

original_cmap = plt.get_cmap("Reds")
colors = original_cmap(np.arange(original_cmap.N))
colors[0] = [1, 1, 1, 1]  # Set the first color to white
custom_cmap = ListedColormap(colors)

fig = plt.figure(figsize=(11,12), layout="constrained")
ax1 = plt.subplot(5,4,4)
ax2 = plt.subplot(5,4,8)
ax3 = plt.subplot(5,4,12)
ax4 = plt.subplot(5,4,16)
ax5 = plt.subplot(5,4,20)

ax6 = plt.subplot(5,4,2)
ax7 = plt.subplot(5,4,6)
ax8 = plt.subplot(5,4,10)
ax9 = plt.subplot(5,4,14)
ax10 = plt.subplot(5,4,18)

ax11 = plt.subplot(5,4,1)
ax12 = plt.subplot(5,4,5)
ax13 = plt.subplot(5,4,9)
ax14 = plt.subplot(5,4,13)
ax15 = plt.subplot(5,4,17)

ax16 = plt.subplot(5,4,3)
ax17 = plt.subplot(5,4,7)
ax18 = plt.subplot(5,4,11)
ax19 = plt.subplot(5,4,15)
ax20 = plt.subplot(5,4,19)

for ax in fig.axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

im_ct1s = ax1.pcolormesh(stlnmf_ct1, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct2s = ax2.pcolormesh(stlnmf_ct2, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct3s = ax3.pcolormesh(stlnmf_ct3, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct4s = ax4.pcolormesh(stlnmf_ct4, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct5s = ax5.pcolormesh(stlnmf_ct5, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))

im_ct1r = ax6.pcolormesh(card_ct1, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct2r = ax7.pcolormesh(card_ct2, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct3r = ax8.pcolormesh(card_ct3, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct4r = ax9.pcolormesh(card_ct4, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct5r = ax10.pcolormesh(card_ct5, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))

im_ct1real = ax11.pcolormesh(real_ct1, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct2real = ax12.pcolormesh(real_ct2, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct3real = ax13.pcolormesh(real_ct3, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct4real = ax14.pcolormesh(real_ct4, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct5real = ax15.pcolormesh(real_ct5, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))

im_ct1stereo = ax16.pcolormesh(stereo_ct1, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct2stereo = ax17.pcolormesh(stereo_ct2, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct3stereo = ax18.pcolormesh(stereo_ct3, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct4stereo = ax19.pcolormesh(stereo_ct4, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct5stereo = ax20.pcolormesh(stereo_ct5, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))

ax11.set_ylabel("Cell Type 1: Acinar", weight="bold")
ax12.set_ylabel("Cell Type 2: Cancer Clone A", weight="bold")
ax13.set_ylabel("Cell Type 3: Endocrine", weight="bold")
ax14.set_ylabel("Cell Type 4: Fibroblasts", weight="bold")
ax15.set_ylabel("Cell Type 5: Terminal ductal like", weight="bold")

ax15.set_xlabel("Ground Truth", weight="bold")
ax10.set_xlabel("CARD", weight="bold")
ax20.set_xlabel("Stereoscope", weight="bold")
ax5.set_xlabel("STL NMF", weight="bold")

ax11.set_title(label="Five Clusters, Real Data",
          loc="left",
          fontstyle='italic', fontsize=15, weight="bold")

fig.colorbar(im_ct1s, ax=[ax1, ax2, ax3, ax4, ax5], orientation='vertical', fraction=0.3, pad=0.04)
fig.get_layout_engine().set(w_pad=.1, h_pad=.1)

fig.savefig("v_real5_lognorm.png")





fig = plt.figure(figsize=(11,12), layout="constrained")
ax1 = plt.subplot(5,4,4)
ax2 = plt.subplot(5,4,8)
ax3 = plt.subplot(5,4,12)
ax4 = plt.subplot(5,4,16)
ax5 = plt.subplot(5,4,20)

ax6 = plt.subplot(5,4,2)
ax7 = plt.subplot(5,4,6)
ax8 = plt.subplot(5,4,10)
ax9 = plt.subplot(5,4,14)
ax10 = plt.subplot(5,4,18)

ax11 = plt.subplot(5,4,1)
ax12 = plt.subplot(5,4,5)
ax13 = plt.subplot(5,4,9)
ax14 = plt.subplot(5,4,13)
ax15 = plt.subplot(5,4,17)

ax16 = plt.subplot(5,4,3)
ax17 = plt.subplot(5,4,7)
ax18 = plt.subplot(5,4,11)
ax19 = plt.subplot(5,4,15)
ax20 = plt.subplot(5,4,19)

for ax in fig.axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

im_ct1s = ax1.pcolormesh(stlnmf_ct1, cmap=custom_cmap)
im_ct2s = ax2.pcolormesh(stlnmf_ct2, cmap=custom_cmap)
im_ct3s = ax3.pcolormesh(stlnmf_ct3, cmap=custom_cmap)
im_ct4s = ax4.pcolormesh(stlnmf_ct4, cmap=custom_cmap)
im_ct5s = ax5.pcolormesh(stlnmf_ct5, cmap=custom_cmap)

im_ct1r = ax6.pcolormesh(card_ct1, cmap=custom_cmap)
im_ct2r = ax7.pcolormesh(card_ct2, cmap=custom_cmap)
im_ct3r = ax8.pcolormesh(card_ct3, cmap=custom_cmap)
im_ct4r = ax9.pcolormesh(card_ct4, cmap=custom_cmap)
im_ct5r = ax10.pcolormesh(card_ct5, cmap=custom_cmap)

im_ct1real = ax11.pcolormesh(real_ct1, cmap=custom_cmap)
im_ct2real = ax12.pcolormesh(real_ct2, cmap=custom_cmap)
im_ct3real = ax13.pcolormesh(real_ct3, cmap=custom_cmap)
im_ct4real = ax14.pcolormesh(real_ct4, cmap=custom_cmap)
im_ct5real = ax15.pcolormesh(real_ct5, cmap=custom_cmap)

im_ct1stereo = ax16.pcolormesh(stereo_ct1, cmap=custom_cmap)
im_ct2stereo = ax17.pcolormesh(stereo_ct2, cmap=custom_cmap)
im_ct3stereo = ax18.pcolormesh(stereo_ct3, cmap=custom_cmap)
im_ct4stereo = ax19.pcolormesh(stereo_ct4, cmap=custom_cmap)
im_ct5stereo = ax20.pcolormesh(stereo_ct5, cmap=custom_cmap)

ax11.set_ylabel("Cell Type 1: Acinar", weight="bold")
ax12.set_ylabel("Cell Type 2: Cancer Clone A", weight="bold")
ax13.set_ylabel("Cell Type 3: Endocrine", weight="bold")
ax14.set_ylabel("Cell Type 4: Fibroblasts", weight="bold")
ax15.set_ylabel("Cell Type 5: Terminal ductal like", weight="bold")

ax15.set_xlabel("Ground Truth", weight="bold")
ax10.set_xlabel("CARD", weight="bold")
ax20.set_xlabel("Stereoscope", weight="bold")
ax5.set_xlabel("STL NMF", weight="bold")

ax11.set_title(label="Five Clusters, Real data",
          loc="left",
          fontstyle='italic', fontsize=15, weight="bold")

fig.colorbar(im_ct1s, ax=[ax1, ax2, ax3, ax4, ax5], orientation='vertical', fraction=0.3, pad=0.04)
fig.get_layout_engine().set(w_pad=.1, h_pad=.1)


fig.savefig("v_real5.png")