import os
import sys
import numpy as np

pth = r'D:\edm_lt\github\mf6_dev\modsim_mf6_transport.git\data\modsim_data\uzf_support'

files = os.listdir(pth)
for file in files:
    if "pET_" in file:
        pET = np.loadtxt(os.path.join(pth, file))
        pET_reduced = pET[:, :125]
        
        np.savetxt(os.path.join(pth, file + '.new'), pET_reduced, fmt='%10.6f', delimiter='')
        
        os.remove(os.path.join(pth, file))
        os.rename(os.path.join(pth, file + '.new'), 
                    os.path.join(pth, file))
        print('finished with ' + file)


