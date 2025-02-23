import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LogNorm, ListedColormap
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable



stlnmfV = np.array(pd.read_csv("V_square_2x2.csv", sep=",", header=None))
stlnmfV /= sum(stlnmfV)
stereoV = np.array(pd.read_csv("stereoscope/res/ruitao/XT/W.2025-01-14193415.940102.tsv", sep="\t", header=0, index_col=0))
stereoV = stereoV.T
stereoV /= sum(stereoV)
cardV = np.array(pd.read_csv("./zzz_outputs/card_ruitao.csv", sep=",", header=0)).T
cardV /= sum(cardV)
realV = np.array(pd.read_csv("./00_synthetic/ruitao/square_2x2/V.csv", sep=",", header=0, index_col=0)).T
realV /= sum(realV)
spotV = np.array(pd.read_csv("./result-SPOTlight.csv", sep=",", header=0, index_col=0)).T
spotV /= sum(spotV)

loss_stlnmfV = torch.tensor(np.array(stlnmfV), dtype=torch.float32)
loss_realV = torch.tensor(np.array(realV), dtype=torch.float32)
loss_stereoV = torch.tensor(np.array(stereoV), dtype=torch.float32)
loss_cardV = torch.tensor(np.array(cardV), dtype=torch.float32)
loss_spotV = torch.tensor(np.array(spotV), dtype=torch.float32)



print("STL NMF:")
print(torch.norm(loss_stlnmfV - loss_realV).item())
print((torch.norm(loss_stlnmfV - loss_realV)/torch.norm(loss_realV)).item())

print("Stereo:")
print(torch.norm(loss_stereoV - loss_realV).item())
print((torch.norm(loss_stereoV - loss_realV)/torch.norm(loss_realV)).item())

print("CARD:")
print(torch.norm(loss_cardV - loss_realV).item())
print((torch.norm(loss_cardV - loss_realV)/torch.norm(loss_realV)).item())

print("SPOTLight:")
print(torch.norm(loss_spotV - loss_realV).item())
print((torch.norm(loss_spotV - loss_realV)/torch.norm(loss_realV)).item())

locs = np.array(pd.read_csv("./00_synthetic/ruitao/square_2x2/regions.csv", sep=",", header=None))


stlnmf_ct1 = np.zeros((16,16))
stlnmf_ct2 = np.zeros((16,16))
stlnmf_ct3 = np.zeros((16,16))
stlnmf_ct4 = np.zeros((16,16))

stereo_ct1 = np.zeros((16,16))
stereo_ct2 = np.zeros((16,16))
stereo_ct3 = np.zeros((16,16))
stereo_ct4 = np.zeros((16,16))

card_ct1 = np.zeros((16,16))
card_ct2 = np.zeros((16,16))
card_ct3 = np.zeros((16,16))
card_ct4 = np.zeros((16,16))

real_ct1 = np.zeros((16,16))
real_ct2 = np.zeros((16,16))
real_ct3 = np.zeros((16,16))
real_ct4 = np.zeros((16,16))

spot_ct1 = np.zeros((16,16))
spot_ct2 = np.zeros((16,16))
spot_ct3 = np.zeros((16,16))
spot_ct4 = np.zeros((16,16))

for _ in range(len(stlnmfV)):
    for j in range(len(stlnmfV[0])):
        stlnmf_ct1[locs[j][1]-1][locs[j][0]-1] = stlnmfV[0][j]
        stlnmf_ct2[locs[j][1]-1][locs[j][0]-1] = stlnmfV[1][j]
        stlnmf_ct3[locs[j][1]-1][locs[j][0]-1] = stlnmfV[2][j]
        stlnmf_ct4[locs[j][1]-1][locs[j][0]-1] = stlnmfV[3][j]

        stereo_ct1[locs[j][1]-1][locs[j][0]-1] = stereoV[0][j]
        stereo_ct2[locs[j][1]-1][locs[j][0]-1] = stereoV[1][j]
        stereo_ct3[locs[j][1]-1][locs[j][0]-1] = stereoV[2][j]
        stereo_ct4[locs[j][1]-1][locs[j][0]-1] = stereoV[3][j]

        card_ct1[locs[j][1]-1][locs[j][0]-1] = cardV[0][j]
        card_ct2[locs[j][1]-1][locs[j][0]-1] = cardV[1][j]
        card_ct3[locs[j][1]-1][locs[j][0]-1] = cardV[2][j]
        card_ct4[locs[j][1]-1][locs[j][0]-1] = cardV[3][j]

        real_ct1[locs[j][1]-1][locs[j][0]-1] = realV[0][j]
        real_ct2[locs[j][1]-1][locs[j][0]-1] = realV[1][j]
        real_ct3[locs[j][1]-1][locs[j][0]-1] = realV[2][j]
        real_ct4[locs[j][1]-1][locs[j][0]-1] = realV[3][j]

        spot_ct1[locs[j][1]-1][locs[j][0]-1] = spotV[0][j]
        spot_ct2[locs[j][1]-1][locs[j][0]-1] = spotV[1][j]
        spot_ct3[locs[j][1]-1][locs[j][0]-1] = spotV[2][j]
        spot_ct4[locs[j][1]-1][locs[j][0]-1] = spotV[3][j]



original_cmap = plt.get_cmap("Reds")
colors = original_cmap(np.arange(original_cmap.N))
colors[0] = [1, 1, 1, 1]  # Set the first color to white
custom_cmap = ListedColormap(colors)

fig = plt.figure(figsize=(8,6), layout='constrained')

ax1 = plt.subplot(4,5,1)
ax2 = plt.subplot(4,5,6)
ax3 = plt.subplot(4,5,11)
ax4 = plt.subplot(4,5,16)

ax5 = plt.subplot(4,5,2)
ax6 = plt.subplot(4,5,7)
ax7 = plt.subplot(4,5,12)
ax8 = plt.subplot(4,5,17)

ax9 = plt.subplot(4,5,3)
ax10 = plt.subplot(4,5,8)
ax11 = plt.subplot(4,5,13)
ax12 = plt.subplot(4,5,18)

ax13 = plt.subplot(4,5,4)
ax14 = plt.subplot(4,5,9)
ax15 = plt.subplot(4,5,14)
ax16 = plt.subplot(4,5,19)

ax17 = plt.subplot(4,5,5)
ax18 = plt.subplot(4,5,10)
ax19 = plt.subplot(4,5,15)
ax20 = plt.subplot(4,5,20)

for ax in fig.axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')


im_ct1s = ax5.pcolormesh(stlnmf_ct1, cmap=custom_cmap)
im_ct2s = ax6.pcolormesh(stlnmf_ct2, cmap=custom_cmap)
im_ct3s = ax7.pcolormesh(stlnmf_ct3, cmap=custom_cmap)
im_ct4s = ax8.pcolormesh(stlnmf_ct4, cmap=custom_cmap)

im_ct1r = ax1.pcolormesh(real_ct1, cmap=custom_cmap)
im_ct2r = ax2.pcolormesh(real_ct2, cmap=custom_cmap)
im_ct3r = ax3.pcolormesh(real_ct3, cmap=custom_cmap)
im_ct4r = ax4.pcolormesh(real_ct4, cmap=custom_cmap)

im_ct1st = ax17.pcolormesh(stereo_ct1, cmap=custom_cmap)
im_ct2st = ax18.pcolormesh(stereo_ct2, cmap=custom_cmap)
im_ct3st = ax19.pcolormesh(stereo_ct3, cmap=custom_cmap)
im_ct4st = ax20.pcolormesh(stereo_ct4, cmap=custom_cmap)

im_ct1c = ax13.pcolormesh(card_ct1, cmap=custom_cmap)
im_ct2c = ax14.pcolormesh(card_ct2, cmap=custom_cmap)
im_ct3c = ax15.pcolormesh(card_ct3, cmap=custom_cmap)
im_ct4c = ax16.pcolormesh(card_ct4, cmap=custom_cmap)

im_ct1sp = ax9.pcolormesh(spot_ct1, cmap=custom_cmap)
im_ct2sp = ax10.pcolormesh(spot_ct2, cmap=custom_cmap)
im_ct3sp = ax11.pcolormesh(spot_ct3, cmap=custom_cmap)
im_ct4sp = ax12.pcolormesh(spot_ct4, cmap=custom_cmap)



ax1.set_ylabel("Cell Type 1", weight="bold")
ax2.set_ylabel("Cell Type 2", weight="bold")
ax3.set_ylabel("Cell Type 3", weight="bold")
ax4.set_ylabel("Cell Type 4", weight="bold")

ax4.set_xlabel("Real", weight="bold")
ax8.set_xlabel("STL NMF", weight="bold")
ax20.set_xlabel("Stereoscope", weight="bold")
ax16.set_xlabel("CARD", weight="bold")
ax12.set_xlabel("SPOTlight", weight="bold")


ax1.set_title(label="Synthetic Data",
          loc="left",
          fontstyle='italic', fontsize=15, weight="bold")

fig.colorbar(im_ct1st, ax=[ax17, ax18, ax19, ax20], orientation='vertical', fraction=0.3, pad=0.04)
fig.get_layout_engine().set(w_pad=.1, h_pad=.1)
# fig.tight_layout()

fig.savefig("v_square_2x2.png")
