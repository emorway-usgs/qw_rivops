import numpy as np
import os
import datetime

base_pth = r'D:\edm_lt\github\mf6_dev\modsim_mf6_transport'

#iuzfbnd = np.loadtxt(os.path.join('..', 'data', 'modsim_data', 'uzf_support', 'iuzfbnd_2xmodel.txt'
new_ibnd = os.path.join(base_pth, 'data', 'modsim_data', 'uzf_support')
new_ibnd1_fname = 'iuzfbnd_2xmodel.txt'
new_ibnd1_fl = os.path.join(new_ibnd, new_ibnd1_fname)
iuzbnd = np.loadtxt(new_ibnd1_fl)

pET_pth = os.path.join(base_pth, 'data', 'modsim_data', 'uzf_support')
finf_fl_pth = os.path.join(base_pth, 'data', 'modsim_data', 'uzf_support', 'finf_25yr.txt')
finf_fl = open(finf_fl_pth, 'r')

dts = []
d = datetime.date(1989, 9, 30)
dts.append(d)
while d < datetime.date(2014, 9, 30):
    d += datetime.timedelta(days=1)
    dts.append(d)


# Ensure directory where UZF perioddata files will go exists
if not os.path.isdir(os.path.join(base_pth, 'examples', 'modsimx2', 'uzf_io')):
    uzf_io_dir = os.path.join(base_pth, 'examples', 'modsimx2', 'uzf_io')
    os.mkdir(uzf_io_dir)


# Generate UZF input external to FloPy script
# (but to be referenced by FloPy script)
extdp = 5.4864
extwc = 0.08
for d in dts:
    spd = []  # a list of each day's UZF stress period data. Reinitialize each day
    # Form of the UZF Period Data
    #<iuzno> <finf> <pet> <extdp> <extwc> <ha> <hroot> <rootact> [<aux(naux)>]
    line = finf_fl.readline()
    finf_ln = line.strip().split()
    finf = float(finf_ln[0])
    pET_arr = np.loadtxt(os.path.join(pET_pth, 'pET_' + d.strftime("%Y-%m-%d") + '.txt'))
    
    iuzno = -1
    for i in np.arange(iuzbnd.shape[0]):
        for j in np.arange(iuzbnd.shape[1]):
            if iuzbnd[i, j] != 0:
                iuzno += 1
                pET = float(pET_arr[i, j])
                # The last zero is for the concentration of the finf (rainfall)
                pd_row = [iuzno, finf, pET, extdp, extwc, 0.0, 0.0, 0.0, 0.0]
                spd.append(pd_row)
    
    # 
    spd_fl = os.path.join(uzf_io_dir, 'uzf_pd_' + d.strftime("%Y-%m-%d") + '.txt')
    with open(spd_fl, 'w') as f:
        for _list in spd:
            for _string in _list:
                bytes = f.write(str(_string) + ' ')
            bytes = f.write('\n')
    
    # Update status
    print('Finished with: ' + d.strftime("%Y-%m-%d"))


finf_fl.close()

