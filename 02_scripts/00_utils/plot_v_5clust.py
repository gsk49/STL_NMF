import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LogNorm, ListedColormap
import numpy as np
import torch

# stlnmfV = np.array(pd.read_csv("DEEPNMF_5c_v0_0.csv", sep=",", header=None))
cardV = np.array(pd.read_csv("./CARD/noisy_5_v/card_x0_0_0_sim.csv", sep=",", header=0)).T
cardV /= sum(cardV)
stlnmf = np.array(pd.read_csv("./zzz_outputs/DEEPNMF_x0_0_noisy5.csv", sep=",", header=None))
stlnmf /= sum(stlnmf)
realV = np.array(pd.read_csv("./00_synthetic/00_PDAC_A/01_X_Noise/05_clust/02_v/v/v0_0.csv", sep=" ", header=None))
realV /= sum(realV)
stereoV = np.array(pd.read_csv("stereoscope/pdac_a_stereo/01_X_Noise/5ct/x0_0_0_sim/W.2024-11-20200526.342252.tsv", sep="\t", header=0, index_col=0))
stereoV = np.c_[stereoV[:,0],stereoV[:,1],stereoV[:,7],stereoV[:,9],stereoV[:,6]].T
# stereoV = np.array([stereoV[:,0],stereoV[:,1],stereoV[:,7],stereoV[:,9],stereoV[:,6]])
stereoV /= sum(stereoV)
spotV = np.array(pd.read_csv("./SPOTlight/noisy5/spotlight_x0_0_0_sim.tsv", sep="\t", header=0, index_col=0))
spotV = np.c_[spotV[:,0],spotV[:,1],spotV[:,7],spotV[:,9],spotV[:,6]].T
spotV /= sum(spotV)

loss_cardV = torch.tensor(np.array(cardV), dtype=torch.float32)
loss_stlnmf = torch.tensor(np.array(stlnmf), dtype=torch.float32)
loss_stereoV = torch.tensor(np.array(stereoV), dtype=torch.float32)
loss_realV = torch.tensor(np.array(realV), dtype=torch.float32)
loss_spotV = torch.tensor(np.array(spotV), dtype=torch.float32)



print("CARD:")
print(torch.norm(loss_cardV - loss_realV).item())
print((torch.norm(loss_cardV - loss_realV)/torch.norm(loss_realV)).item())

print("Stereo:")
print(torch.norm(loss_stereoV - loss_realV).item())
print((torch.norm(loss_stereoV - loss_realV)/torch.norm(loss_realV)).item())

print("STLNMF:")
print(torch.norm(loss_stlnmf - loss_realV).item())
print((torch.norm(loss_stlnmf - loss_realV)/torch.norm(loss_realV)).item())

print("SPOTlight:")
print(torch.norm(loss_spotV - loss_realV).item())
print((torch.norm(loss_spotV - loss_realV)/torch.norm(loss_realV)).item())


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

spot_ct1 = np.zeros((22,24))
spot_ct2 = np.zeros((22,24))
spot_ct3 = np.zeros((22,24))
spot_ct4 = np.zeros((22,24))
spot_ct5 = np.zeros((22,24))

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

        spot_ct1[locs[j][0]-7][locs[j][1]-7] = spotV[0][j]
        spot_ct2[locs[j][0]-7][locs[j][1]-7] = spotV[1][j]
        spot_ct3[locs[j][0]-7][locs[j][1]-7] = spotV[2][j]
        spot_ct4[locs[j][0]-7][locs[j][1]-7] = spotV[3][j]
        spot_ct5[locs[j][0]-7][locs[j][1]-7] = spotV[4][j]

original_cmap = plt.get_cmap("Reds")
colors = original_cmap(np.arange(original_cmap.N))
colors[0] = [1, 1, 1, 1]  # Set the first color to white
custom_cmap = ListedColormap(colors)

fig = plt.figure(figsize=(5,5), layout="constrained")
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

ax11.set_ylabel("Cell Type 1", weight="bold")
ax12.set_ylabel("Cell Type 2", weight="bold")
ax13.set_ylabel("Cell Type 3", weight="bold")
ax14.set_ylabel("Cell Type 4", weight="bold")
ax15.set_ylabel("Cell Type 5", weight="bold")

ax15.set_xlabel("Ground Truth", weight="bold")
ax10.set_xlabel("CARD", weight="bold")
ax20.set_xlabel("Stereoscope", weight="bold")
ax5.set_xlabel("Proposed", weight="bold")

ax11.set_title(label="Five Clusters, Noisy X",
          loc="left",
          fontstyle='italic', fontsize=15, weight="bold")

fig.colorbar(im_ct1s, ax=[ax1, ax2, ax3, ax4, ax5], orientation='vertical', fraction=0.3, pad=0.04)
fig.get_layout_engine().set(w_pad=.1, h_pad=.1)

# fig.savefig("v_noisy5_lognorm.png")





fig = plt.figure(figsize=(9,8), layout="constrained")
ax1 = plt.subplot(5,5,4)
ax2 = plt.subplot(5,5,9)
ax3 = plt.subplot(5,5,14)
ax4 = plt.subplot(5,5,19)
ax5 = plt.subplot(5,5,24)

ax6 = plt.subplot(5,5,2)
ax7 = plt.subplot(5,5,7)
ax8 = plt.subplot(5,5,12)
ax9 = plt.subplot(5,5,17)
ax10 = plt.subplot(5,5,22)

ax11 = plt.subplot(5,5,1)
ax12 = plt.subplot(5,5,6)
ax13 = plt.subplot(5,5,11)
ax14 = plt.subplot(5,5,16)
ax15 = plt.subplot(5,5,21)

ax16 = plt.subplot(5,5,3)
ax17 = plt.subplot(5,5,8)
ax18 = plt.subplot(5,5,13)
ax19 = plt.subplot(5,5,18)
ax20 = plt.subplot(5,5,23)

ax21 = plt.subplot(5,5,5)
ax22 = plt.subplot(5,5,10)
ax23 = plt.subplot(5,5,15)
ax24 = plt.subplot(5,5,20)
ax25 = plt.subplot(5,5,25)

for ax in fig.axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

im_ct1s = ax6.pcolormesh(stlnmf_ct1, cmap=custom_cmap)
im_ct2s = ax7.pcolormesh(stlnmf_ct2, cmap=custom_cmap)
im_ct3s = ax8.pcolormesh(stlnmf_ct3, cmap=custom_cmap)
im_ct4s = ax9.pcolormesh(stlnmf_ct4, cmap=custom_cmap)
im_ct5s = ax10.pcolormesh(stlnmf_ct5, cmap=custom_cmap)

im_ct1r = ax1.pcolormesh(card_ct1, cmap=custom_cmap)
im_ct2r = ax2.pcolormesh(card_ct2, cmap=custom_cmap)
im_ct3r = ax3.pcolormesh(card_ct3, cmap=custom_cmap)
im_ct4r = ax4.pcolormesh(card_ct4, cmap=custom_cmap)
im_ct5r = ax5.pcolormesh(card_ct5, cmap=custom_cmap)

im_ct1real = ax11.pcolormesh(real_ct1, cmap=custom_cmap)
im_ct2real = ax12.pcolormesh(real_ct2, cmap=custom_cmap)
im_ct3real = ax13.pcolormesh(real_ct3, cmap=custom_cmap)
im_ct4real = ax14.pcolormesh(real_ct4, cmap=custom_cmap)
im_ct5real = ax15.pcolormesh(real_ct5, cmap=custom_cmap)

im_ct1stereo = ax21.pcolormesh(stereo_ct1, cmap=custom_cmap)
im_ct2stereo = ax22.pcolormesh(stereo_ct2, cmap=custom_cmap)
im_ct3stereo = ax23.pcolormesh(stereo_ct3, cmap=custom_cmap)
im_ct4stereo = ax24.pcolormesh(stereo_ct4, cmap=custom_cmap)
im_ct5stereo = ax25.pcolormesh(stereo_ct5, cmap=custom_cmap)

im_ct1spot = ax16.pcolormesh(spot_ct1, cmap=custom_cmap)
im_ct2spot = ax17.pcolormesh(spot_ct2, cmap=custom_cmap)
im_ct3spot = ax18.pcolormesh(spot_ct3, cmap=custom_cmap)
im_ct4spot = ax19.pcolormesh(spot_ct4, cmap=custom_cmap)
im_ct5spot = ax20.pcolormesh(spot_ct5, cmap=custom_cmap)

ax11.set_ylabel("Cell Type 1", weight="bold")
ax12.set_ylabel("Cell Type 2", weight="bold")
ax13.set_ylabel("Cell Type 3", weight="bold")
ax14.set_ylabel("Cell Type 4", weight="bold")
ax15.set_ylabel("Cell Type 5", weight="bold")

ax15.set_xlabel("Ground Truth", weight="bold")
ax5.set_xlabel("CARD", weight="bold")
ax25.set_xlabel("Stereoscope", weight="bold")
ax10.set_xlabel("Proposed", weight="bold")
ax20.set_xlabel("SPOTlight", weight="bold")


ax11.set_title(label="Five Clusters, Noisy X",
          loc="left",
          fontstyle='italic', fontsize=15, weight="bold")

fig.colorbar(im_ct1stereo, ax=[ax21, ax22, ax23, ax24, ax25], orientation='vertical', fraction=0.3, pad=0.04)
fig.get_layout_engine().set(w_pad=.1, h_pad=.1)


fig.savefig("v_noisy5.png")

# fig = plt.figure(figsize=(7,17))
# ax1 = plt.subplot(5,2,1)
# ax2 = plt.subplot(5,2,2)

# ax3 = plt.subplot(5,2,3)
# ax4 = plt.subplot(5,2,4)

# ax5 = plt.subplot(5,2,5)
# ax6 = plt.subplot(5,2,6)

# ax7 = plt.subplot(5,2,7)
# ax8 = plt.subplot(5,2,8)

# ax9 = plt.subplot(5,2,9)
# ax10 = plt.subplot(5,2,10)

# card_ct1 = ax1.pcolormesh(np.abs(card_ct1-real_ct1), cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
# ax1.set_xlabel("CARD, ct1")
# fig.colorbar(card_ct1, ax=ax1)
# STLNMF_ct1 = ax2.pcolormesh(np.abs(stlnmf_ct1-real_ct1), cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
# ax2.set_xlabel("STLNMF, ct1")
# fig.colorbar(STLNMF_ct1, ax=ax2)

# card_ct2 = ax3.pcolormesh(np.abs(card_ct2-real_ct2), cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
# ax3.set_xlabel("CARD, ct2")
# fig.colorbar(card_ct2, ax=ax3)
# STLNMF_ct2 = ax4.pcolormesh(np.abs(stlnmf_ct2-real_ct2), cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
# ax4.set_xlabel("STLNMF, ct2")
# fig.colorbar(STLNMF_ct2, ax=ax4)

# card_ct3 = ax5.pcolormesh(np.abs(card_ct3-real_ct3), cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
# ax5.set_xlabel("CARD, ct3")
# fig.colorbar(card_ct3, ax=ax5)
# STLNMF_ct3 = ax6.pcolormesh(np.abs(stlnmf_ct3-real_ct3), cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
# ax6.set_xlabel("STLNMF, ct3")
# fig.colorbar(STLNMF_ct3, ax=ax6)

# card_ct4 = ax7.pcolormesh(np.abs(card_ct4-real_ct4), cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
# ax7.set_xlabel("CARD, ct4")
# fig.colorbar(card_ct4, ax=ax7)
# STLNMF_ct4 = ax8.pcolormesh(np.abs(stlnmf_ct4-real_ct4), cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
# ax8.set_xlabel("STLNMF, ct4")
# fig.colorbar(STLNMF_ct4, ax=ax8)

# card_ct5 = ax9.pcolormesh(np.abs(card_ct4-real_ct4), cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
# ax9.set_xlabel("CARD, ct5")
# fig.colorbar(card_ct5, ax=ax9)
# STLNMF_ct5 = ax10.pcolormesh(np.abs(stlnmf_ct4-real_ct4), cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
# ax10.set_xlabel("STLNMF, ct5")
# fig.colorbar(STLNMF_ct5, ax=ax10)

# fig.savefig("card_vs_STLNMF_error.png")