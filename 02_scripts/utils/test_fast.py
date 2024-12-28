import pandas as pd
import numpy as np

card = pd.read_csv("../CARD/noisy_5_v/card_x0_0_0_sim.csv", sep=",", header=0)
fast = pd.read_csv("./FAST_realx_simv.csv", sep=",", header=None)
dnmf = pd.read_csv("./DEEPNMF_X_0.csv", sep=",", header=None)

card = np.array(card).T
fast = np.array(fast)
dnmf = np.array(dnmf)

print(card.shape)
print(fast.shape)
print(dnmf.shape)

print(np.linalg.norm(card-fast, ord="fro"))
print(np.linalg.norm(card - dnmf, ord="fro"))