import numpy as np
import pandas as pd

V = pd.read_csv("00_synthetic/ruitao/square_2x2/V.csv", header=0).T
locs = V.iloc[0]
x = np.array([int(item.split('_')[0]) for item in locs])
y = np.array([int(item.split('_')[1]) for item in locs])

region = []
for i in range(len(x)):
    if x[i]>8:
        if y[i]>8:
            region.append(3)
        else:
            region.append(1)
    else:
        if y[i]>8:
            region.append(2)
        else:
            region.append(0)

result = np.column_stack((x, y, region))
np.savetxt('regions.csv', result, fmt="%d", delimiter=",")

i_1 = [1,0,0,0]
i_2 = [0,1,0,0]
i_3 = [0,0,1,0]
i_4 = [0,0,0,1]

stack = np.stack([i_1, i_2, i_3, i_4])
I = stack[region].squeeze().T

np.savetxt("i.csv", I, fmt="%d", delimiter=',')