import os
import numpy as np
import datetime

pth = r'G:\For_qw_ops'
fname = 'modsim.uzf'

nper = 9132
nrow = 64
ncol = 133

fpth = os.path.join(pth, fname)
finf = []
pET = []
dt = []
dt.append('9/30/1989')
flg=False  # For a 1-time skip
with open(fpth, 'r') as f:
    next(f)  # advance through the header line
    for line in f:
        if "FINF" in line:
            m_arr = line.strip().split()
            finf.append(float(m_arr[1]))
            if flg:
                dt.append(m_arr[-1])
            flg = True
        if "pET" in line:
            m_arr = line.strip().split()
            #dt = m_arr[-1]
            if "CONSTANT" in line:
                pet_arr = np.ones((nrow, ncol)) * float(m_arr[1])
                pET.append(pet_arr)
            else:
                # read nrow rows into a numpy array
                pet_arr = []
                for i in range(nrow):
                    line = next(f)
                    m_arr = line.strip().split()
                    for itm in m_arr:
                        pet_arr.append(float(itm))
                
                pet_arr = np.array(pet_arr)
                pet_arr = pet_arr.reshape((nrow, ncol))
                pET.append(pet_arr)
                print('Finished with pET, ' + dt[-1])


print('Error msg will follow the above for loop, no biggie, it is expected.')
pET = np.array(pET)

# Look at new ibound arrays since their shape has changed slightly
new_pth = r'D:\edm_lt\github\mf6_dev\modsim_mf6_transport\data\modsim_data\bas_support'
new_ibnd1_fname = 'ibnd1_2xmodel.txt'

new_ibnd1_pth = os.path.join(new_pth, new_ibnd1_fname)

ibnd_new = np.loadtxt(new_ibnd1_pth)

# Do a few manipulations to get the old ibnd as 0's & 1's
old_ibnd = os.path.join(new_pth, 'ibnd1.txt')
old_ibnd = np.loadtxt(old_ibnd)
old_ibnd_truc = old_ibnd[:,:ibnd_new.shape[1]]

diff = ibnd_new - old_ibnd_truc

# Reduce the length of the second dimension from 133 to 125
pET_new = []
for t in range(pET.shape[0]):
    pET_new.append(pET[t,:,:125])

pET = np.array(pET_new)

# Cycle through each time index of the 3D pET and update the newly active cells
# with a "naturally vegetated" pET value

for t in np.arange(1,pET.shape[0]):
    val = pET[t, 25, 0]
    pET[t][24:31, 87:] = val


# Save the arrays (don't use the length of dt as indication of how many arrays)
sv_pth = r'D:\edm_lt\github\mf6_dev\modsim_mf6_transport\data\modsim_data\uzf_support'
for i in range(pET.shape[0]):
    dt_str = datetime.datetime.strptime(dt[i], '%m/%d/%Y').strftime('%Y-%m-%d')
    sv_fl = os.path.join(sv_pth, 'pET_' + dt_str + '.txt')
    np.savetxt(sv_fl, pET[i], fmt='%10.6f', delimiter='')
    print("Finished writing " + dt_str)


# print finf values to txt file
with open(os.path.join(sv_pth, 'finf_25yr.txt'), 'w') as sw:
    for i, val in enumerate(finf[:-1]):
        sw.write(format(val, '.6f') + '   #' + dt[i] + '\n')

