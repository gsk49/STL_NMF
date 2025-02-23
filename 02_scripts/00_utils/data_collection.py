from gen_synthetic import get_v, get_x
import numpy as np
import time

for i in range(150):
    a = np.random.randint(100)
    b = np.random.randint(100)
    g = np.random.randint(100)
    denom = np.exp(-a)+np.exp(-b)+np.exp(-g)
    alpha = np.exp(-a)/denom
    beta = np.exp(-b)/denom
    for j in range(3):
        v, intensity_vec = get_v(alpha=alpha, beta=beta, add_noise=(j==0), smooth=(j>1), n_clust=5)
        np.savetxt('./00_synthetic/00_PDAC_A/00_No_X_Noise/05_clust/02_v/v/v'+str(i)+'_'+str(j)+'.csv', v)
        np.savetxt('./00_synthetic/00_PDAC_A/00_No_X_Noise/05_clust/02_v/intensity/i'+str(i)+'_'+str(j)+'.csv', intensity_vec)

        num_x = 1 # number of simulated X
        sigma = 0 # noise level
        ###
        x_sim = get_x(v, num_x=num_x, sigma=sigma)
        for n in range(num_x):
            np.savetxt('./00_synthetic/00_PDAC_A/00_No_X_Noise/05_clust/01_x/x'+str(i)+'_' +str(j)+ '_' + str(n) + '_sim.csv', x_sim[n, :, :], fmt='%d', delimiter=',')
        ####
    print(i)
