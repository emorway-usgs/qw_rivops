# An example for testing MODSIM - MF6 integration using BMI interface
# This problem first appeared in [Morway et al. (2016)](https://www.sciencedirect.com/science/article/pii/S136481521630113X) and is patterned after an irrigated river valley.  Advanced package options include a reservoir, 4 major diversions, simulation of the unsaturated zone, and the use of Mover to shunt water between packages. Thousands of connections inside the Mover package simulate irrigation events, either from surface water (SFR $\rightarrow$ UZF) or from groundwater (WEL $\rightarrow$ UZF), among other connection types.  MODSIM can be used to simulate a minimum instream flow requirement at the bottom of the system, reservoir storage accounts, and diversions in order of priority (prior-appropriation).  
# This particular application of MODSIM-MODFLOW 6 is used to explore water quality controled river operations.  The simulation itself is comprised of two copies of the original hypothetical model, one downstream of the other.  The reasoning behind this is to increase the complexity of the river/reservoir operations environment while exploring the ability of MODSIM-MODFLOW 6 to improve water quality at the model outlet.
# ![MODSIM-MODFLOW6 hypothetical model](../images/modsim_model_view.png)
# ### Import flopy for constructing, running, and post-processing the test problem
# (For now, remember that this is running a customized version of flopy with support for MODFLOW 6's GWT process)

# Append to system path to include the common subdirectory
import os
import sys
sys.path.append(os.path.join("..", "common"))

# #### Imports
import matplotlib.pyplot as plt
import flopy
import numpy as np
import pandas as pd
import config
from figspecs import USGSFigure
from flopy.utils.util_array import read1d
import flopy.utils.binaryfile as bf
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

sys.path.append(os.path.join("..", "common", "modsim_data"))
import build_modsim_helper_funcs as modsimBld

mf6exe = os.path.abspath(config.mf6_exe)
assert os.path.isfile(mf6exe)
print(mf6exe)

# Set figure properties specific to this problem
figure_size = (6, 4.5)

# Base simulation and model name and workspace
ws = config.base_ws
example_name = "modsimx2"

# #### Model units
length_units = "meters"
time_units = "days"

# Table
nlay = 4       # Number of layers
nrow = 64      # Number of rows
ncol = 125     # Number of columns
delr = 400.0   # Column width ($m$)
delc = 400.0   # Row width ($m$)
prsity = 0.3   # Porosity
perlen = 365   # Simulation time ($days$)
k11 = 10.0     # Horizontal hydraulic conductivity ($m/d$)
k33 = 2.0      # Vertical hydraulic conductivity ($m/d$)
Ss  = 1e-6     # Storativity
sy_1 = 0.28    # Specific yield and porosity of layer 1
sy_2 = 0.30    # Specific yield and porosity of layer 2
sy_3 = 0.32    # Specific yield and porosity of layer 3
sy_4 = 0.34    # Specific yield and porosity of layer 4
surfdep = 0.5  # Land-surface depression ($m$)
vks  = 0.12    # Saturated hydraulic conductivity of unsaturated zone ($m/d$)
thtr = 0.05    # Residual water content
thts = 0.42    # Saturated water content
thti = 0.15    # Initial water content (unsaturated zone)
eps  = 7.1     # Brooks-Corey epsilon
al = 2.        # Longitudinal dispersivity ($m$)
rhob_1 = 1.5   # Bulk density of layer 1 ($g/cm^3$)
rhob_2 = 1.7   # Bulk density of layer 2 ($g/cm^3$)
rhob_3 = 1.8   # Bulk density of layer 3 ($g/cm^3$)
rhob_4 = 1.9   # Bulk density of layer 4 ($g/cm^3$)
Kd =  0.176    # Distribution coefficient ($cm^3/g$)

# #### Additional model input
# Time related
rng     = pd.date_range(start='10/1/1989', end='10/1/1990')
numdays = 366
perlen  = [1] * numdays
nper    = len(perlen)
nstp    = [1] * numdays
tsmult  = [1.] * numdays
# Recharge related
ss_rch = np.loadtxt(os.path.join('..', 'data', 'modsim_data', 'rch_support', 'ss_rch_2xmodel.txt'))
tr_rch = np.loadtxt(os.path.join('..', 'data', 'modsim_data', 'rch_support', 'tr_rch_2xmodel.txt'))
# UZF related
iuzfbnd = np.loadtxt(os.path.join('..','data','modsim_data','uzf_support','iuzfbnd_2xmodel.txt'))
ha = 0.
hroot = 0.
rootact = 0.
finf_ss  = 0.0001    # 0.1 mm
pet_ss   = 0.002     # 2.0 mm
extdp_ss = 5.4864    # m (18 ft)
extwc_ss = 0.08
extdp = extdp_ss
extwc = extwc_ss

# Transport related
mixelm = 0   # advection scheme 
ath1   = al  # horizontal transverse dispersivity
atv    = al  # vertical transverse dispersivity
dmcoef = 0.  # molecular diffusion
xt3d = [False]

icelltype = 1
iconvert = 1
sy_1 = np.ones((nrow, ncol)) * sy_1
sy_2 = np.ones((nrow, ncol)) * sy_2
sy_3 = np.ones((nrow, ncol)) * sy_3
sy_4 = np.ones((nrow, ncol)) * sy_4
sy = [sy_1, sy_2, sy_3, sy_4]
poro_1 = sy_1.copy()
poro_2 = sy_2.copy()
poro_3 = sy_3.copy()
poro_4 = sy_4.copy()
prsity = [poro_1, poro_2, poro_3, poro_4]

# ## Note: The downstream model will be lower by 151.04 m

# #### Model geometry and Active model domain
top_orig  = np.loadtxt(os.path.join('..','data','modsim_data','dis_support','top1.txt'))
# To set the bottom elevations for the two new models, determine the 
# original thicknesses and then apply these thicknesses to the new models
# Remember that the original model's surface elevations were adjusted upstream
# of the reservoir.  Also don't forget that the number of columns was reduced from
# 133 to 125.
bot1_orig = np.loadtxt(os.path.join('..','data','modsim_data','dis_support','bot1.txt'))
bot2_orig = np.loadtxt(os.path.join('..','data','modsim_data','dis_support','bot2.txt'))
bot3_orig = np.loadtxt(os.path.join('..','data','modsim_data','dis_support','bot3.txt'))
bot4_orig = np.loadtxt(os.path.join('..','data','modsim_data','dis_support','bot4.txt'))
# Now get the new land surface elevation
top_2x = np.loadtxt(os.path.join('..','data','modsim_data', 'dis_support','top_2xmodel.txt'))
# Next, apply the layer 1 thicknesses to the new land surface elevation.
bot1_2x = top_2x - (top_orig[:, :125] - bot1_orig[:,: 125])
bot2_2x = bot1_2x - (bot1_orig[:, :125] - bot2_orig[:, :125])
bot3_2x = bot2_2x - (bot2_orig[:, :125] - bot3_orig[:, :125])
bot4_2x = bot3_2x - (bot3_orig[:, :125] - bot4_orig[:, :125])

botm_upstream_model = [bot1_2x, bot2_2x, bot3_2x, bot4_2x]
botm_np = np.array(botm_upstream_model)

ibnd1 = np.loadtxt(os.path.join('..','data','modsim_data','bas_support','ibnd1_2xmodel.txt'))
ibnd2 = np.loadtxt(os.path.join('..','data','modsim_data','bas_support','ibnd2_2xmodel.txt'))
ibnd3 = np.loadtxt(os.path.join('..','data','modsim_data','bas_support','ibnd3_2xmodel.txt'))
ibnd4 = np.loadtxt(os.path.join('..','data','modsim_data','bas_support','ibnd4_2xmodel.txt'))
ibnda = np.array([ibnd1, ibnd2, ibnd3, ibnd4])
strt1 = np.loadtxt(os.path.join('..','data','modsim_data','bas_support','strt_2xmodel.txt'))
strt  = [strt1, strt1, strt1, strt1]
strt  = np.array(strt)

# Set some of the storage parameters
# if using 1.5 g/cm^3 (layer 1), unit conversion is 100cm x 100cm x 100cm / 1m^3 = 1e6 g/m^3
rhob_1 = np.ones_like(poro_1) * rhob_1 * 1e6
rhob_2 = np.ones_like(poro_2) * rhob_2 * 1e6
rhob_3 = np.ones_like(poro_3) * rhob_3 * 1e6
rhob_4 = np.ones_like(poro_4) * rhob_4 * 1e6
rhob   = [rhob_1, rhob_2, rhob_3, rhob_4]
# Kd: "Distribution coefficient"
# if using 0.176 cm^3/g, unit conversion is 1m^3 / (100cm x 100cm x 100cm) = 1e-6 m^3/g
Kd_1   = np.ones_like(poro_1) * Kd * 1e-6
Kd_2   = np.ones_like(poro_2) * Kd * 1e-6
Kd_3   = np.ones_like(poro_3) * Kd * 1e-6
Kd_4   = np.ones_like(poro_4) * Kd * 1e-6
Kd     = [Kd_1, Kd_2, Kd_3, Kd_4]

# Set solver parameter values (and related)
nouter, ninner = 100, 300              
hclose, rclose, relax = 1e-2, 1e2, 0.97

# Some variables that are needed globally
iuzno_cell_dict = {}  # Setting up a dictionary to help with SFR -> UZF connections in the next sub-section
iuzno_dict_rev  = {}


def convert_mainConc2gperm3(SC):
    DSmgL = 0.793 * SC - 89.256  # conversion provided by Miller et al. (2010), Table 3 (Ark Riv @ Avondale)
    DSgperm3 = DSmgL * 1000 / 1000  # 1g/1000mg * 1000L/1m^3
    return DSgperm3


def convert_tribConc2gperm3(SC):
    DSmgL = 1.033 * SC - 420.487  # conversion provided by Miller et al. (2010), Table 3 (Ark Riv @ Avondale)
    DSgperm3 = DSmgL * 1000 / 1000  # 1g/1000mg * 1000L/1m^3
    return DSgperm3


def generate_rcha_starting_concentrations(gwf):
    # Function for instantiating starting concentration arrays:
    # Need to generate the starting concentration array when the flow model 
    # is instantiated because it is entered as an auxiliary array.  Easiest way
    # to accomplish this is with a function that can be called anytime.  The 
    # auxiliary array will set the concentration of recharge occurring around 
    # the perimeter of the model. 
    #
    # For initial concentrations, start with a back-ground salt (TDS) 
    # concentration typical of a salinized ag system to provide some variation, 
    # setup a gradient of increasing concentrations from lowerleft to 
    # upperright, then mirror, and finally splice together the upper and lower 
    # halves of the two arrays, respectively.
    
    Lx = (ncol - 1) * delr
    Ly = (nrow - 1) * delc
    Ls = np.sqrt(Lx ** 2 + Ly ** 2) - 40000
    grad = 0.05
    c1 = grad * Ls
    a = -1
    b = -1
    c = 1
    x = gwf.modelgrid.xcellcenters
    y = gwf.modelgrid.ycellcenters
    d = abs(a*x + b*y + c) / np.sqrt(2)
    conc_grad_lower = c1 + d / Ls * c1
    conc_grad_upper = np.flipud(conc_grad_lower).copy()
    strt_conc = conc_grad_upper.copy()
    strt_conc[32:63, :] = conc_grad_lower[32:63, :].copy()
    strta = np.array([strt_conc, strt_conc, strt_conc, strt_conc])
    
    return strta


def starting_conc(gwf):
    rch_conc = np.zeros((1, nrow, ncol))
    strta = generate_rcha_starting_concentrations(gwf)
    rch_conc[0, :, :] = strta[3, :, :].copy() * 0.5
    rch_conc[0, tr_rch == 0] = 0.
    return strta, rch_conc


def build_gwf_model(sim, i, elev_adj, silent=False):
    # Instantiating MODFLOW 6 groundwater flow model
    gwfname = 'gwf_' + example_name + '_' + str(i + 1)
    gwf = flopy.mf6.ModflowGwf(sim,
                               modelname=gwfname,
                               save_flows=True,
                               newtonoptions=True,
                               model_nam_file='{}.nam'.format(gwfname)
    )
    
    # Instantiating MODFLOW 6 solver for flow model
    imsgwf = flopy.mf6.ModflowIms(sim,
                                  print_option="summary",
                                  outer_dvclose=hclose,
                                  outer_maximum=2000,
                                  under_relaxation="cooley",
                                  linear_acceleration="BICGSTAB",
                                  under_relaxation_theta=0.3,
                                  under_relaxation_kappa=0.08,
                                  under_relaxation_gamma=0.08,
                                  under_relaxation_momentum=0.01,
                                  inner_dvclose=1.0e-3,
                                  rcloserecord=[0.0001, "relative_rclose"],
                                  inner_maximum=100,
                                  relaxation_factor=0.0,
                                  number_orthogonalizations=2,
                                  preconditioner_levels=8,
                                  preconditioner_drop_tolerance=0.001,
                                  filename='{}.ims'.format(gwfname)
    )
    sim.register_ims_package(imsgwf, [gwf.name])
    
    # Instantiating MODFLOW 6 discretization package
    flopy.mf6.ModflowGwfdis(gwf,
                            length_units=length_units,
                            nlay=nlay,
                            nrow=nrow,
                            ncol=ncol,
                            delr=delr,
                            delc=delc,
                            top=top_2x - elev_adj,
                            botm=botm_np - elev_adj,
                            idomain=ibnda,
                            filename='{}.dis'.format(gwfname)
    )
    
    # Instantiating MODFLOW 6 node-property flow package
    flopy.mf6.ModflowGwfnpf(gwf,
                            save_flows=False,
                            icelltype=icelltype, 
                            k=k11, 
                            k33=k33,
                            save_specific_discharge=True,
                            filename='{}.npf'.format(gwfname)
    )
    
    # Instantiating MODFLOW 6 initial conditions package for flow model
    flopy.mf6.ModflowGwfic(gwf, 
                           strt=strt - elev_adj,
                           filename='{}.ic'.format(gwfname)
    )
    
    # Instantiate MODFLOW 6 storage package 
    flopy.mf6.ModflowGwfsto(gwf, 
                            ss=Ss, 
                            sy=sy,
                            iconvert=iconvert,
                            steady_state={0: True},
                            transient={1: True},
                            filename='{}.sto'.format(gwfname)
    )
    
    # Instantiating MODFLOW 6 output control package for flow model
    flopy.mf6.ModflowGwfoc(gwf,
                           head_filerecord="{}.hds".format(gwfname),
                           budget_filerecord="{}.bud".format(gwfname),
                           headprintrecord=[
                                            ('COLUMNS', 10, 'WIDTH', 15,
                                             'DIGITS', 6, 'GENERAL')],
                           saverecord=[('HEAD', 'LAST'),
                                       ('BUDGET', 'LAST')],
                           printrecord=[('HEAD', 'LAST'),
                                        ('BUDGET', 'LAST')]
    )
    
    
    # Instantiating MODFLOW 6 array-based recharge package (with a concentration auxiliary array above)
    # The RCH package is intended to simulate a small regional groundwater inflow from surrounding lands. Because the modeled area is patterned after an irrigated river valley, surrounding non-irrigated associated with semi-arid to arid regions are not explicitly represented by the model, though they may contribute small amounts of water along the northern and southern perimeter of active model domain. The concentration associated with this regional groundwater inflow has abitrarily been set to a low background concentration.

    # Initialize a concentration array by grabbing the starting concentration array values
    # When passing the RCHA constructor, the auxiliary concentration array needs to be 3d w/ 1 layer.
    strta, rch_conc = starting_conc(gwf)
    irch = {0: 3, 1: 3}
    recharge = {0: ss_rch, 1:tr_rch}
    rchc = {0: rch_conc, 1: rch_conc}
    
    flopy.mf6.ModflowGwfrcha(gwf, 
                             readasarrays=True,
                             auxiliary='CONCENTRATION',
                             pname='RCH-1', 
                             fixed_cell=False, 
                             print_input=False, 
                             irch=irch, 
                             recharge=recharge,
                             aux=rchc,
                             filename='{}.rcha'.format(gwfname)
    )
    
    # Instantiating MODFLOW 6 DRN package with a concentration auxiliary 
    # variable.  Here, the drain (DRN) package is used to simulate groundwater 
    # discharge to land surface to keep this water separate from rejected 
    # infiltrated simulated by the UZF package. From a transport perspective, 
    # these two sources of water can have very different concentrations, and 
    # therefore very different contributions of mass to nearby surface waters, 
    # whether streams, open drains simulated using the stream package, or lakes
    # and ponds simulated with the LAK package. Movement of both rejected 
    # infiltration and groundwater discharge to land surface to the surface 
    # water system will be handled by the MVR package. The auxiliary variable,
    # which is concentration, is entered as 0.0 and is present for the sake of
    # model input mechanics.  Because groundwater is discharging, the MODFLOW 6 
    # code will assign the calculated concentration of the groundwater to the 
    # groundwater discharge.
    #
    # Rejected infiltration: uzf -> mvr -> sfr/lak
    # Groundwater discharge to land surface: drn -> mvr -> sfr/lak
    #
    # Need to cycle through all land surface cells and create a drain for 
    # handling groundwater discharge to land surface
    
    drn_spd  = []
    drn_dict = {}
    cond     = 10000  # Use an arbitrarily high conductance term to avoid impeding groundwater discharge
    ddrn     = -0.5   # See definition of auxdepthname in drain package documentation to learn more about this parameter
    idrnno   = 0
    for i in np.arange(0, top_2x.shape[0]):
        for j in np.arange(0, top_2x.shape[1]):
            if ibnda[0, i, j]:
                drn_spd.append([(0, i, j), top_2x[i, j], cond, ddrn, 0.0])  #  last value is the concentration
                # append dictionary of drain indices
                drn_dict.update({(i, j): idrnno})
                idrnno += 1
    
    maxbound = len(drn_spd)   # The total number 
    spd = {0: drn_spd}
    flopy.mf6.ModflowGwfdrn(gwf, pname='DRN-1',
                            auxiliary=['ddrn', 'CONCENTRATION'],
                            auxdepthname='ddrn',
                            print_input=False, 
                            print_flows=False,
                            maxbound=maxbound,
                            mover=True,
                            stress_period_data=spd,   # wel_spd established in the MVR setup
                            boundnames=False, 
                            save_flows=True,
                            filename='{}.drn'.format(gwfname)
    )
    
    # Start of Advanced Package Instantiation
    # * Unsaturated Zone Flow Package
    # * Streamflow Routing Package
    # * Lake Package
    # * Mover Package
    #
    # Instantiating MODFLOW 6 unsatruated-zone flow package
    # The datasets relied upon to create the UZF input are from the original UZF input file for MF-NWT. As such, 1 year's worth of pET arrays were peeled out of the original UZF input file and stored as 2D arrays, 1 file per day. The pET information varies cell-by-cell and is based on an annually varying cropping pattern, with 5 different crop types. I believe the original publication contains more information about the pET values. Precipitation data also was stored in the original UZF input file, and these values will be entered here as well.
    # Fill 3D numpy array with every cell's total uz storage (used for post-processing later)
    uzMaxStor = np.zeros_like(ibnda)
    cell_area = delr * delc
    uz_voids  = (thts - thtr)
    for k in np.arange(uzMaxStor.shape[0]):
        for i in np.arange(uzMaxStor.shape[1]):
            for j in np.arange(uzMaxStor.shape[2]):
                if ibnda[k, i, j] > 0:
                    if k==0:
                        cel_thkness = (top_2x[i, j] - elev_adj) - (botm_np[0, i, j] - elev_adj)
                    else:
                        cel_thkness = (botm_np[k-1, i, j] - elev_adj) - (botm_np[k, i, j] - elev_adj)
                    uzMaxStor[k, i, j] = cell_area * uz_voids * cel_thkness
    
    # UZF Boundname support 
    # ----------------------
    fl = os.path.join('..', 'data', 'modsim_data', 'sfr_support')
    sfr2uzf = pd.read_csv(os.path.join(fl,'sfr_2_uzf_conns.csv'), header=0)
    
    uzf_packagedata = []
    pd0             = []
    iuzno           = 0
    surfdep         = 0.5
    # Set up the UZF static variables
    for k in range(nlay):
        for i in range(0, iuzfbnd.shape[0]):
            for j in range(0,iuzfbnd.shape[1]):
                if iuzfbnd[i, j] != 0:                           # even though ibnd = 1 below the reservoir, need to ignore
                    if k == 0:                                   # all cells below a lake, hence ibnda[0,i,j]
                        lflag = 1
                        iuzno_cell_dict.update({(i, j): iuzno})  # establish new dictionary entry for current cell 
                                                                 # addresses & iuzno connections are both 0-based
                        iuzno_dict_rev.update({iuzno: (i, j)})   # For post-processing the mvr output, need a dict with iuzno as key
                    else:
                        lflag = 0
                        surfdep = 0.0
                    
                    # Set the vertical connection, which is the cell below
                    ivertcon =  iuzno + int(iuzfbnd.sum())
                    if k == nlay - 1: ivertcon = -1       # adjust if on bottom layer (no underlying conn.)
                                                          # Keep in mind 0-based adjustment (so ivertcon==-1 -> 0)
                    
                    # Set the boundname for the land surface cells
                    if k == 0:
                        cmd_area = [getattr(itm,'Ag') for itm in sfr2uzf.itertuples() if getattr(itm,"Row")-1==i and getattr(itm, "Col")-1==j]
                        try:
                            cmd_area = cmd_area[0]
                        except:
                            cmd_area = []
                        
                        if cmd_area==1:
                            bndnm = 'CA1'
                        elif cmd_area==2:
                            bndnm = 'CA2'
                        elif cmd_area==3:
                            bndnm = 'CA3'
                        elif cmd_area==4:
                            bndnm = 'CA4'
                        else:
                            bndnm = 'NatVeg'
                    else:
                        bndnm='deepCell'
            
                    # <iuzno> <cellid(ncelldim)> <landflag> <ivertcon> <surfdep> <vks> <thtr> <thts> <thti> <eps> [<boundname>]
                    uz = [iuzno,      (k, i, j),     lflag,  ivertcon,  surfdep,  vks,  thtr,  thts,  thti,  eps,   bndnm]
                    uzf_packagedata.append(uz)
                
                    # steady-state values can be set here
                    if lflag:
                        pd0.append((iuzno, finf_ss, pet_ss, extdp_ss, extwc_ss, ha, hroot, rootact))
                    
                    iuzno += 1
    
    # Store the steady state uzf stresses in dictionary
    uzf_perioddata = {0: pd0}
    
    # Generate boundary conditions only for land surface cells
    # A couple of notes:
    #  o FINF represents precipitation and is constant for every land surface cell
    #  o Irrigation amendments will be made by the mover package from 1 of two sources:
    #    - Surface-water irrigation from ditches
    #    - Supplemental pumping water
    #  o pET varies by stress period and by cell, since difference crops were assumed for different cells
    #  o Steady state values for FINF, pET, extdp, and extwc set above
    #  o In the transient stress periods:
    #    - FINF comes from stored file
    #    - pET comes from stored arrays
    #    - extdp is constant for the entire simulation
    #    - extwc is constant for the entire simulation
    #  o Set the uzf boundaries for 1 yr w/ daily stress periods
    
    finf_in = os.path.join('..', 'data', 'modsim_data', 'uzf_support', 'finf_list.txt')
    finf = np.loadtxt(finf_in)
    
    for idx, dt in enumerate(rng):
        print('working on UZF sp: ' + dt.strftime('%Y-%m-%d'))
        # Get current stress period finf value
        finfx = finf[idx]
        # Get current stress period pET value (all arrays externally reduced from 133 columns to 125)
        fl = os.path.join('..','data','modsim_data','uzf_support','pET_' + dt.strftime('%Y-%m-%d') + '.txt')
        pETx = np.loadtxt(fl)
        
        # For each active uzf cell, set the boundary values
        iuzno = 0
        pdx   = []
        for i in range(0, iuzfbnd.shape[0]):
            for j in range(0,iuzfbnd.shape[1]):
                if iuzfbnd[i,j] != 0:
                    pdx.append((iuzno, finfx, pETx[i,j], extdp, extwc, ha, hroot, rootact))
                    iuzno += 1
    
        # Append the period data to the dictionary that is passed into the constructor
        uzf_perioddata.update({idx+1: pdx})
    
    nuzfcells = len(uzf_packagedata)
    
    # Observations
    uzf_obs = {'{}.uzfobs'.format(gwfname): [('CA1_gwd',        'uzf-gwd',        'CA1'),  # Relies on boundnames
                                             ('CA2_gwd',        'uzf-gwd',        'CA2'),
                                             ('CA3_gwd',        'uzf-gwd',        'CA3'),
                                             ('CA4_gwd',        'uzf-gwd',        'CA4'),
                                             ('CA1_gwd',        'uzf-gwd',        'CA1'),
                                             ('CA1_gwd2mvr',    'uzf-gwd-to-mvr', 'CA1'), 
                                             ('CA2_gwd2mvr',    'uzf-gwd-to-mvr', 'CA2'),
                                             ('CA3_gwd2mvr',    'uzf-gwd-to-mvr', 'CA3'),
                                             ('CA4_gwd2mvr',    'uzf-gwd-to-mvr', 'CA4'),
                                             ('CA1_rinf2mvr',   'rej-inf-to-mvr', 'CA1'), 
                                             ('CA2_rinf2mvr',   'rej-inf-to-mvr', 'CA2'),
                                             ('CA3_rinf2mvr',   'rej-inf-to-mvr', 'CA3'),
                                             ('CA4_rinf2mvr',   'rej-inf-to-mvr', 'CA4'),
                                             ('CA1_gwet',       'uzf-gwet',       'CA1'), 
                                             ('CA2_gwet',       'uzf-gwet',       'CA2'),
                                             ('CA3_gwet',       'uzf-gwet',       'CA3'),
                                             ('CA4_gwet',       'uzf-gwet',       'CA4'),
                                             ('CA1_uzet',       'uzet',           'CA1'), 
                                             ('CA2_uzet',       'uzet',           'CA2'),
                                             ('CA3_uzet',       'uzet',           'CA3'),
                                             ('CA4_uzet',       'uzet',           'CA4'),
                                             ('CA1_frommvr',    'from-mvr',       'CA1'), 
                                             ('CA2_frommvr',    'from-mvr',       'CA2'),
                                             ('CA3_frommvr',    'from-mvr',       'CA3'),
                                             ('CA4_frommvr',    'from-mvr',       'CA4'),
                                             ('NatVeg_rch',     'uzf-gwrch',      'NatVeg'),
                                             ('NatVeg_gwd2mvr', 'uzf-gwd-to-mvr', 'NatVeg'),
                                             ('NatVeg_rinf2mvr','rej-inf-to-mvr', 'NatVeg')]}
    
    flopy.mf6.ModflowGwfuzf(gwf, nuzfcells=nuzfcells, 
                            boundnames=True,
                            ntrailwaves=15, 
                            nwavesets=200, 
                            print_flows=False,
                            save_flows=True,
                            simulate_et=True, 
                            linear_gwet=True,
                            mover=True,
                            observations=uzf_obs,
                            packagedata=uzf_packagedata, 
                            perioddata=uzf_perioddata,
                            budget_filerecord='{}.uzf.bud'.format(gwfname),
                            pname='UZF-1',
                            filename='{}.uzf'.format(gwfname)
    )
    
    
    # Instantiating MODFLOW 6 streamflow routing package 
    # Use of a few externally defined functions aid this part of the model generation. The motivation for using an external script is to limit the amount of script appearing here, since these auxillary functions contains SFR information (roughly the first ~800 lines of the imported script) from the original MF-NWT formatted SFR input file and some hacky non-flopy related script that simply help transfer data from the original SFR input file to the new MF6 format.
    # 
    # SFR-to-SFR diversions area handled within the SFR package, and not with MVR, since this water is staying within the package.  In order to test out the different diversion types, all 4 cprior options are exercised as noted in the comments below.  _cprior_ options include:
    # * FRACTION - the amount of the diversion is computed as a fraction of the streamflow leaving reach RNO
    # * EXCESS - a diversion is made only if QDS for reach RNO exceeds the value of DIVFLOW. If this occurs, then the quantity of water diverted is the excess flow (Qds-DIVFLOW) and Qds from reach RNO is set equal to DIVFLOW
    # * THRESHOLD - if QDS in reach RNO is less than the specified diversion flow (DIVFLOW), no water is diverted from reach RNO. If QDS in reach RNO is greater than or equal to (DIVFLOW), (DIVFLOW) is diverted and QDS is set to the remainder (QDS DIVFLOW)).
    # * UPTO - if QDS in reach RNO is greater than or equal to the specified diversion flow (DIVFLOW), QDS is reduced by DIVFLOW. If QDS in reach RNO is less than (DIVFLOW), DIVFLOW is set to QDS and there will be no flow available for reaches connected to downstream end of reach RNO.
    # 
    # Continuous time series of TDS concentrations were pulled from two sites in Colorado's Arkansas River Valley for plugging into SFT package below. These concentrations are for a site on the mainstem of the Arkansas River located in SE Colorado and one of its tributaries.  A USGS report by [Miller et al. (2010); Table 3](https://pubs.usgs.gov/sir/2010/5069/pdf/SIR10-5069.pdf) provides site-specific equations to convert the continuously values in units of microsiemens per centimeter at 25 degrees Celsius ($\mu$S/cm) to mg/L.  The site specific equations are for the Arkansas River at Avondale and Purgatoire River near Las Animas sites, and respectively represent:
    # * Main Model Inflow: $DS = 0.793SC - 89.256$
    # * Tributary Inflow: $DS = 1.033SC - 420.487$
    # where SC is the recorded concentration in $\mu$S/cm and DS is the concentration in mg/L.
    
    # Call functions defined within the imported script
    conns        = modsimBld.gen_mf6_sfr_connections()  
    ndvs, ustrf  = modsimBld.tally_ndvs(conns)
    divs, cprior = modsimBld.define_divs_dat(ndvs)
    pkdat        = modsimBld.gen_sfrpkdata(conns, ndvs, ustrf) 
    
    # For the stress period data, refer to the time series
    # reachs with inflow are 1, the main inflow, and 556, the tributary inflow
    sfrspd = {0: [[0, 'INFLOW', 'main'], 
    	          [555, 'INFLOW', 'trib'],
    	          [20, 'DIVERSION', 0, 50001],
    	          [135, 'DIVERSION', 0, 50001],
    	          [289, 'DIVERSION', 0, 50001],
    	          [429, 'DIVERSION', 0, 50001]]}
    
    # Need a generic "on" (start of irrigation season)
    sfrspd_irrstrt = {'irrstrt': [[ 20, 'DIVERSION', 1, 304465.0],   # CA1 (MODSIM overrides this value)
                                  [ 45, 'DIVERSION', 1,  28000.0],   # CA1 - lateral 1
                                  [ 70, 'DIVERSION', 1,  28000.0],
                                  [ 87, 'DIVERSION', 1,  28000.0],
                                  [103, 'DIVERSION', 1,      1.0],   # CA1 - last lateral
                                  [135, 'DIVERSION', 1, 164465.0],   # CA2 (MODSIM overrides this value)
                                  [175, 'DIVERSION', 1,  28000.0],   # CA2 - lateral 1   
                                  [190, 'DIVERSION', 1,  28000.0],                       
                                  [209, 'DIVERSION', 1,  28000.0],                       
                                  [231, 'DIVERSION', 1,      1.0],   # CA2 - last lateral
                                  [289, 'DIVERSION', 1, 140000.0],   # CA3 (MODSIM overrides this value)
                                  [319, 'DIVERSION', 1,  28000.0],   # CA3 - lateral 1   
                                  [340, 'DIVERSION', 1,  28000.0],                       
                                  [356, 'DIVERSION', 1,  28000.0],                       
                                  [388, 'DIVERSION', 1,      1.0],   # CA3 - last lateral
                                  [429, 'DIVERSION', 1,  24465.0],   # CA4 (MODSIM overrides this value)
                                  [464, 'DIVERSION', 1,  28000.0],   # CA4 - lateral 1   
                                  [476, 'DIVERSION', 1,  28000.0],                       
                                  [515, 'DIVERSION', 1,  28000.0],                       
                                  [553, 'DIVERSION', 1,      1.0]]}  # CA4 - last lateral]]}
    # Need a generic "off" (end of irrigation season)
    sfrspd_irrstop = {'irrstop': [[ 20, 'DIVERSION', 1, 500000.0],   # CA1 (B/c 'EXCESS', set value hi to turn off)
                                  [ 45, 'DIVERSION', 1,  28000.0],   # CA1 - lateral 1
                                  [ 70, 'DIVERSION', 1,  28000.0],
                                  [ 87, 'DIVERSION', 1,  28000.0],
                                  [103, 'DIVERSION', 1,      1.0],   # CA1 - last lateral
                                  [135, 'DIVERSION', 1, 500000.0],   # CA2 (MODSIM overrides this value)
                                  [175, 'DIVERSION', 1,  28000.0],   # CA2 - lateral 1   
                                  [190, 'DIVERSION', 1,  28000.0],                       
                                  [209, 'DIVERSION', 1,  28000.0],                       
                                  [231, 'DIVERSION', 1,      1.0],   # CA2 - last lateral
                                  [289, 'DIVERSION', 1, 500000.0],   # CA3 (MODSIM overrides this value)
                                  [319, 'DIVERSION', 1,  28000.0],   # CA3 - lateral 1   
                                  [340, 'DIVERSION', 1,  28000.0],                       
                                  [356, 'DIVERSION', 1,  28000.0],                       
                                  [388, 'DIVERSION', 1,      1.0],   # CA3 - last lateral
                                  [429, 'DIVERSION', 1, 500000.0],   # CA4 (MODSIM overrides this value)
                                  [464, 'DIVERSION', 1,  28000.0],   # CA4 - lateral 1   
                                  [476, 'DIVERSION', 1,  28000.0],                       
                                  [515, 'DIVERSION', 1,  28000.0],                       
                                  [553, 'DIVERSION', 1,      1.0]]}
    
    # TODO: Need a way of amending the SFR stress period data (sfrspd), 
    # which controls the 'lateral' diversions, based on the date.  When 
    # April 1st, turn on with the sfrspd_irrstrt above, and when Oct. 31st,
    # turn off with the sfr_spd_irrstop above.  In so doing, will need to 
    # replace the keys 'irrstrt' and 'irrstop' with the appropriate 0-based
    # stress period number.
    
    # Set the diversion scheme for the 4 main diversions off of the mainstem river 
    # as well as for the 4 lateral diverting water out of diversions.  Diversions 
    # start in stress period 184 (Apr. 1st), the beginning of the irrigation season
    # and remain as-is until the end of the simulation.  From mf6io.pdf:
    #  "The information specified in the PERIOD block will continue to apply for 
    #   all subsequent stress periods, unless the program encounters another PERIOD
    #   block."
    sfrspd.update({183: [[ 20, 'DIVERSION', 0,220190.72],   # Command Area (CA) 1 uses EXCESS  (Will be overidden by MODSIM)
                         [ 45, 'DIVERSION', 0,     0.2 ],   # CA1, lateral #1 uses THRESHOLD
                         [ 70, 'DIVERSION', 0,     0.28],   # CA1, lateral #2 uses THRESHOLD
                         [ 87, 'DIVERSION', 0,     0.45],   # CA1, lateral #3 uses THRESHOLD
                         [103, 'DIVERSION', 0,     1.0 ],   # CA1, lateral #4 uses FRACTION (divert any remaining flow)
                         [135, 'DIVERSION', 0,     0.32],   # CA2, main diversion uses EXCESS (Will be overidden by MODSIM)
                         [175, 'DIVERSION', 0,     0.2 ],   # CA2, lateral #1 uses THRESHOLD
                         [190, 'DIVERSION', 0,     0.28],   # CA2, lateral #2 uses THRESHOLD
                         [209, 'DIVERSION', 0,     0.45],   # CA2, lateral #3 uses THRESHOLD
                         [231, 'DIVERSION', 0,     1.0 ],   # CA2, lateral #4 uses FRACTION (divert any remaining flow)
                         [289, 'DIVERSION', 0,220190.72],   # CA3, main diversion uses EXCESS (Will be overidden by MODSIM)
                         [319, 'DIVERSION', 0, 41591.4 ],   # CA3, lateral #1 uses THRESHOLD
                         [340, 'DIVERSION', 0, 41591.4 ],   # CA3, lateral #2 uses THRESHOLD
                         [356, 'DIVERSION', 0, 41591.4 ],   # CA3, lateral #3 uses THRESHOLD
                         [388, 'DIVERSION', 0,     1.0 ],   # CA3, lateral #4 uses FRACTION (divert any remaining flow)
                         [429, 'DIVERSION', 0, 24465.5 ],   # CA4, main diversion uses EXCESS (Will be overidden by MODSIM)
                         [464, 'DIVERSION', 0, 97862.1 ],   # CA4, lateral #1 uses THRESHOLD
                         [476, 'DIVERSION', 0, 97862.1 ],   # CA4, lateral #1 uses THRESHOLD
                         [515, 'DIVERSION', 0, 97862.1 ],   # CA4, lateral #1 uses THRESHOLD
                         [553, 'DIVERSION', 0,     1.0 ]]}) # CA4, lateral #1 uses FRACTION (divert any remaining flow)
    
    # Observations
    sfr_obs = {'{}.sfrobs'.format(gwfname): [('main_in',         'downstream-flow',  1),  # For now, these need to be 1-based
                                             ('res_in',          'downstream-flow', 20),
                                             ('resRelease',      'upstream-flow',   21),
                                             ('resSpill',        'downstream-flow',743),
                                             ('priorDiv1',       'sfr',             21),
                                             ('AgArea1_div',     'upstream-flow',   22),
                                             ('AgArea1_lat1',    'upstream-flow',   47),
                                             ('AgArea1_lat2',    'upstream-flow',   72),
                                             ('AgArea1_lat3',    'upstream-flow',   89),
                                             ('AgArea1_lat4',    'upstream-flow',  105),
                                             ('priorDiv2',       'sfr',            136),
                                             ('AgArea2_div',     'upstream-flow',  137),
                                             ('AgArea2_lat1',    'upstream-flow',  177),
                                             ('AgArea2_lat2',    'upstream-flow',  192),
                                             ('AgArea2_lat3',    'upstream-flow',  211),
                                             ('AgArea2_lat4',    'upstream-flow',  233),
                                             ('priorDiv3',       'sfr',            290), 
                                             ('AgArea3_div',     'upstream-flow',  291),
                                             ('AgArea3_lat1',    'upstream-flow',  321),
                                             ('AgArea3_lat2',    'upstream-flow',  342),
                                             ('AgArea3_lat3',    'upstream-flow',  358),
                                             ('AgArea3_lat4',    'upstream-flow',  390),
                                             ('priorDiv4',       'sfr',            430),
                                             ('AgArea4_div',     'upstream-flow',  431),
                                             ('AgArea4_lat1',    'upstream-flow',  466),
                                             ('AgArea4_lat2',    'upstream-flow',  478),
                                             ('AgArea4_lat3',    'upstream-flow',  517),
                                             ('AgArea4_lat4',    'upstream-flow',  555),
                                             ('postag4_lat4',    'upstream-flow',  719),
                                             ('minInstrmQ',      'upstream-flow',  677),
                                             ('modeloutlet',     'downstream-flow',742), 
                                             ('AgArea3_preLat1', 'upstream-flow',  320),
                                             ('AgArea3_preLat2', 'upstream-flow',  341),
                                             ('AgArea3_preLat3', 'upstream-flow',  357),
                                             ('AgArea3_preLat4', 'upstream-flow',  389)],
               '{}_detailed.sfrobs'.format(gwfname): [('mainstem1_1',     'ext-inflow',  1),
                                                      ('mainstem1_2',  'upstream-flow',  1),
                                                      ('mainstem1_3',         'inflow',  1),
                                                      ('mainstem1_4',       'from-mvr',  1),
                                                      ('mainstem1_5',            'sfr',  1),
                                                      ('mainstem1_6','downstream-flow',  1),
                                                      ('mainstem1_7',        'outflow',  1),
                                                      ('mainstem2_1',     'ext-inflow',  2),
                                                      ('mainstem2_2',  'upstream-flow',  2),
                                                      ('mainstem2_3',         'inflow',  2),
                                                      ('mainstem2_4',       'from-mvr',  2),
                                                      ('mainstem2_5',            'sfr',  2),
                                                      ('mainstem2_6','downstream-flow',  2),
                                                      ('mainstem2_7',        'outflow',  2),
                                                      ('mainstem3_1',     'ext-inflow',  3),
                                                      ('mainstem3_2',  'upstream-flow',  3),
                                                      ('mainstem3_3',         'inflow',  3),
                                                      ('mainstem3_4',       'from-mvr',  3),
                                                      ('mainstem3_5',            'sfr',  3),
                                                      ('mainstem3_6','downstream-flow',  3),
                                                      ('mainstem3_7',        'outflow',  3),
                                                      ('mainstem4_1',     'ext-inflow',  4),
                                                      ('mainstem4_2',  'upstream-flow',  4),
                                                      ('mainstem4_3',         'inflow',  4),
                                                      ('mainstem4_4',       'from-mvr',  4),
                                                      ('mainstem4_5',            'sfr',  4),
                                                      ('mainstem4_6','downstream-flow',  4),
                                                      ('mainstem4_7',        'outflow',  4),
                                                      ('mainstem5_1',     'ext-inflow',  5),
                                                      ('mainstem5_2',  'upstream-flow',  5),
                                                      ('mainstem5_3',         'inflow',  5),
                                                      ('mainstem5_4',       'from-mvr',  5),
                                                      ('mainstem5_5',            'sfr',  5),
                                                      ('mainstem5_6','downstream-flow',  5),
                                                      ('mainstem5_7',        'outflow',  5),
                                                      ('mainstem6_1',     'ext-inflow',  6),
                                                      ('mainstem6_2',  'upstream-flow',  6),
                                                      ('mainstem6_3',         'inflow',  6),
                                                      ('mainstem6_4',       'from-mvr',  6),
                                                      ('mainstem6_5',            'sfr',  6),
                                                      ('mainstem6_6','downstream-flow',  6),
                                                      ('mainstem6_7',        'outflow',  6),
                                                      ('mainstem7_1',     'ext-inflow',  7),
                                                      ('mainstem7_2',  'upstream-flow',  7),
                                                      ('mainstem7_3',         'inflow',  7),
                                                      ('mainstem7_4',       'from-mvr',  7),
                                                      ('mainstem7_5',            'sfr',  7),
                                                      ('mainstem7_6','downstream-flow',  7),
                                                      ('mainstem7_7',        'outflow',  7),
                                                      ('mainstem8_1',     'ext-inflow',  8),
                                                      ('mainstem8_2',  'upstream-flow',  8),
                                                      ('mainstem8_3',         'inflow',  8),
                                                      ('mainstem8_4',       'from-mvr',  8),
                                                      ('mainstem8_5',            'sfr',  8),
                                                      ('mainstem8_6','downstream-flow',  8),
                                                      ('mainstem8_7',        'outflow',  8),
                                                      ('mainstem284_1',     'ext-inflow', 284),
                                                      ('mainstem284_2',  'upstream-flow', 284),
                                                      ('mainstem284_3',         'inflow', 284),
                                                      ('mainstem284_4',       'from-mvr', 284),
                                                      ('mainstem284_5',            'sfr', 284),
                                                      ('mainstem284_6','downstream-flow', 284),
                                                      ('mainstem284_7',        'outflow', 284),
                                                      ('mainstem285_1',     'ext-inflow', 285),
                                                      ('mainstem285_2',  'upstream-flow', 285),
                                                      ('mainstem285_3',         'inflow', 285),
                                                      ('mainstem285_4',       'from-mvr', 285),
                                                      ('mainstem285_5',            'sfr', 285),
                                                      ('mainstem285_6','downstream-flow', 285),
                                                      ('mainstem285_7',        'outflow', 285),
                                                      ('mainstem286_1',     'ext-inflow', 286),
                                                      ('mainstem286_2',  'upstream-flow', 286),
                                                      ('mainstem286_3',         'inflow', 286),
                                                      ('mainstem286_4',       'from-mvr', 286),
                                                      ('mainstem286_5',            'sfr', 286),
                                                      ('mainstem286_6','downstream-flow', 286),
                                                      ('mainstem286_7',        'outflow', 286),
                                                      ('mainstem287_1',     'ext-inflow', 287),
                                                      ('mainstem287_2',  'upstream-flow', 287),
                                                      ('mainstem287_3',         'inflow', 287),
                                                      ('mainstem287_4',       'from-mvr', 287),
                                                      ('mainstem287_5',            'sfr', 287),
                                                      ('mainstem287_6','downstream-flow', 287),
                                                      ('mainstem287_7',        'outflow', 287),
                                                      ('mainstem288_1',     'ext-inflow', 288),
                                                      ('mainstem288_2',  'upstream-flow', 288),
                                                      ('mainstem288_3',         'inflow', 288),
                                                      ('mainstem288_4',       'from-mvr', 288),
                                                      ('mainstem288_5',            'sfr', 288),
                                                      ('mainstem288_6','downstream-flow', 288),
                                                      ('mainstem288_7',        'outflow', 288),
                                                      ('mainstem289_1',     'ext-inflow', 289),
                                                      ('mainstem289_2',  'upstream-flow', 289),
                                                      ('mainstem289_3',         'inflow', 289),
                                                      ('mainstem289_4',       'from-mvr', 289),
                                                      ('mainstem289_5',            'sfr', 289),
                                                      ('mainstem289_6','downstream-flow', 289),
                                                      ('mainstem289_7',        'outflow', 289),
                                                      ('mainstem290_1',     'ext-inflow', 290),
                                                      ('mainstem290_2',  'upstream-flow', 290),
                                                      ('mainstem290_3',         'inflow', 290),
                                                      ('mainstem290_4',       'from-mvr', 290),
                                                      ('mainstem290_5',            'sfr', 290),
                                                      ('mainstem290_6','downstream-flow', 290),
                                                      ('mainstem290_7',        'outflow', 290)]}
    
    # Time series input are initialized through sfr.ts.initialize after package instantiation
    sfr = flopy.mf6.ModflowGwfsfr(gwf, 
                                  print_stage=False,
                                  print_flows=False,
                                  print_input=True,
                                  budget_filerecord=gwfname + '.sfr.bud', 
                                  save_flows=True,
                                  mover=True, 
                                  pname='SFR-1',
                                  unit_conversion=86400.00, 
                                  boundnames=True, 
                                  nreaches=len(conns),
                                  packagedata=pkdat, 
                                  connectiondata=conns,
                                  diversions=divs,
                                  perioddata=sfrspd,
                                  observations=sfr_obs,   
                                  auxiliary=['modsimdiv'],
                                  filename='{}.sfr'.format(gwfname)
    )
    
    # Pull in the time series of flow entering the model on the mainstem
    fname1 = os.path.join('..','data','modsim_data','sfr_support','Model_Inflow_Q_wConc.txt')
    fname2 = os.path.join('..','data','modsim_data','sfr_support','Tributary_Q_wConc.txt')
    

    # This grabs two time series simultaneously
    sfr_ts_in = []
    sft_ts_in = []
    with open(fname1, 'r') as flmain, open(fname2, 'r') as fltrib:
        for main in flmain:      # Read line from the original main inflow tabfile
            trib = next(fltrib)  # Read corresponding line from the original tributary inflow tabfile
            main_arr = main.strip().split()
            trib_arr = trib.strip().split()
            if '#' in main_arr[0]:
                pass  # This is a comment line
            else:
                sfr_ts_in.append((main_arr[0], main_arr[1], trib_arr[1]))
                sft_ts_in.append((main_arr[0], convert_mainConc2gperm3(float(main_arr[2])), 
                                               convert_tribConc2gperm3(float(trib_arr[2]))))
    
    sfr.ts.initialize(filename=gwfname + '_sfr_inflows.ts', 
                      timeseries=sfr_ts_in,
                      time_series_namerecord=['main','trib'],
                      interpolation_methodrecord=['stepwise','stepwise'])


    # Instantiating MODFLOW 6 lake package
    # Many blocks to be defined:
    # * options
    # * dimensions
    # * package data
    # * observation data
    # * connection data
    # * outlets
    # * period data
    # * time series data
    # * table input (stage - volume - surface area relationships)
    # Import the original MF-NWT LAK file's ibound array
    
    # Fill the lake's connection data
    fname = os.path.join('..', 'data', 'modsim_data', 'lak_support', 'lakibnd_2xmodel.txt')
    lakibd = np.loadtxt(fname, dtype=int)
    lakeconnectiondata = []
    nlakecon = [0]            # Expand this to [0, 0, ...] for each additional lake
    lak_leakance = 0.01
    for i in range(nrow):
        for j in range(ncol):
            if lakibd[i, j] == 0:
                continue
            else:
                ilak = lakibd[i, j] - 1
                # back
                if i > 0:
                    if lakibd[i - 1, j] == 0 and ibnda[0, i - 1, j] == 1:
                        # <lakeno>     <iconn> <cellid(ncelldim)>   <claktype>      <bedleak> <belev> <telev> <connlen> <connwidth>
                        h = [ilak, nlakecon[ilak], (0, i - 1, j), 'horizontal', lak_leakance,    0.0,    0.0, delc / 2., delr]
                        nlakecon[ilak] += 1
                        lakeconnectiondata.append(h)
                # left
                if j > 0:
                    if lakibd[i, j - 1] == 0 and ibnda[0, i, j - 1] == 1:
                        h = [ilak, nlakecon[ilak], (0, i, j - 1), 'horizontal', lak_leakance,    0.0,    0.0, delr / 2., delc]
                        nlakecon[ilak] += 1
                        lakeconnectiondata.append(h)
                # right
                if j < ncol - 1:
                    if lakibd[i, j + 1] == 0 and ibnda[0, i, j + 1] == 1:
                        h = [ilak, nlakecon[ilak], (0, i, j + 1), 'horizontal', lak_leakance,    0.0,    0.0, delr / 2., delc]
                        nlakecon[ilak] += 1
                        lakeconnectiondata.append(h)
                # front
                if i < nrow - 1:
                    if lakibd[i + 1, j] == 0 and ibnda[0, i + 1, j] == 1:
                        h = [ilak, nlakecon[ilak], (0, i + 1, j), 'horizontal', lak_leakance,    0.0,    0.0, delc / 2., delr]
                        nlakecon[ilak] += 1
                        lakeconnectiondata.append(h)
                # vertical
                v = [ilak, nlakecon[ilak], (1, i, j), 'vertical', lak_leakance, 0.0, 0.0, 0.0, 0.0]
                nlakecon[ilak] += 1
                lakeconnectiondata.append(v)
    
    strtStg = 1390.
    lakpackagedata = [[0, strtStg, nlakecon[0], 'reservoir1']]
    lak_pkdat_dict = {'filename': "lak_pakdata.in", 'data': lakpackagedata}
    
    # There are two outlets from the reservoir:
    #  * Normal release outlet located near the bottom of the dam
    #  * Spillway at top of day to ensure that lake level doesn't exceed certain level
    #
    # Using a function defined in an external file to retrieve the outlet information from the original SFR input file
    outlets = modsimBld.get_lak_outletInfo()
    
    # Period data for the reservoir, including things like precip and evap, will be entered using a timeseries file below
    lakeperioddata = {0:   [(0, 'RAINFALL',    'rain'),
                            (0, 'EVAPORATION', 'evap'),
                            (0, 'RUNOFF',      'runoff'),
                            (0, 'WITHDRAWAL',  'withdrawal'),
                            (0, 'RATE',  -24465.525)],         #  -10 cfs =  -24,465.525 cmd
                      183: [(0, 'RATE', -856293.360)]}         # -350 cfs = -856,293.000 cmd
    
    tab6_filename = '{}.laktab'.format(gwfname)
    
    # note: for specifying lake number, use fortran indexing!
    lak_obs = {'{}.lakobs'.format(gwfname): [('lakestage',  'stage',       'reservoir1'),
                                             ('lakevol',    'volume',      'reservoir1'),
                                             ('lakrain',    'rainfall',    'reservoir1'), 
                                             ('lakevap',    'evaporation', 'reservoir1'),
                                             ('lakrnf',     'runoff',      'reservoir1'),
                                             ('lakexit',    'ext-outflow', 'reservoir1'),
                                             ('lakmvr_in',  'from-mvr',    'reservoir1'),
                                             ('lakmvr_out', 'to-mvr',      'reservoir1'),
                                             ('gwexchng',   'lak',         'reservoir1')]}

    lak = flopy.mf6.ModflowGwflak(gwf, 
                                  time_conversion=86400.0,      # Set time_conversion for use with Manning's eqn.
                                  print_stage=True, 
                                  print_flows=False,
                                  budget_filerecord=gwfname+'.lak.bud',
                                  length_conversion=1.0,
                                  mover=True, 
                                  pname='RES-1',
                                  boundnames=True,
                                  nlakes=len(lakpackagedata), 
                                  noutlets=len(outlets),
                                  outlets=outlets,
                                  packagedata=lak_pkdat_dict, 
                                  connectiondata=lakeconnectiondata,
                                  perioddata=lakeperioddata,
                                  ntables=1,
                                  tables=[0, tab6_filename],
                                  observations=lak_obs,
                                  filename='{}.lak'.format(gwfname)
    )
    
    # Pull in the tabfile defining the lake stage, volume, surface area relationship and put into flopy/mf6 format
    fname = os.path.join('..', 'data', 'modsim_data', 'lak_support', 'Stage_Vol_Area_TABFILE.txt')
    tabinput = []
    with open(fname, 'r') as f:
        for line in f:
            m_arr = line.strip().split()
            #                 <stage>, <volume>,  <sarea>, [<barea>]
            tabinput.append([float(m_arr[0]) - elev_adj, m_arr[1], m_arr[2]])
    
    laktab = flopy.mf6.ModflowUtllaktab(gwf, 
                                        nrow=len(tabinput), 
                                        ncol=3,
                                        table=tabinput,
                                        filename=tab6_filename,
                                        pname='LAK_tab',
                                        parent_file=lak)
    
    fname = os.path.join('..', 'data', 'modsim_data', 'lak_support', 'Res_Release_TABFILE.txt')
    ts_data_resRel = []
    with open(fname, 'r') as tseries2:
        for line in tseries2:
            m_arr = line.strip().split()
            ts_data_resRel.append((m_arr[0], m_arr[1]))
    
    lak.ts.initialize(filename=gwfname + '_res_release.ts', 
                      timeseries=ts_data_resRel,
                      time_series_namerecord='resrel',
                      interpolation_methodrecord='stepwise')
    
    # Use data from original MF-NWT lake package input file to define precipitation and evaportation from the lake
    fname = os.path.join('..', 'data', 'modsim_data', 'lak_support', 'res_fluxes.txt')
    resfluxes = []
    with open(fname, 'r') as fl:
        for i, line in enumerate(fl):
            m_arr = line.strip().split()
            #                 tm,     rain,     evap,   runoff, withdrawal
            resfluxes.append((i, m_arr[0], m_arr[1], m_arr[2], m_arr[3]))
    
    lak.ts.append_package(filename=gwfname + '_res_fluxes.ts', 
                          timeseries=resfluxes,
                          time_series_namerecord=['rain', 'evap', 'runoff', 'withdrawal'],
                          interpolation_methodrecord=['stepwise', 'stepwise', 'stepwise', 'stepwise'])
    
    
    # ## Instantiate MVR package
    # __Pathways to be defined by MVR:__
    # * sfr -> lak
    # * lak -> sfr (reservoir release)
    # * sfr -> uzf (sw irrig)
    # * uzf -> sfr (runoff)
    # * uzf -> lak (runoff in the vicinity of the lake)
    # * wel -> uzf (gw irrig)
    # 
    # __Package names as defined in the cells above include:__
    # * WEL-1 (Still need to add this one above)
    # * UZF-1
    # * SFR-1
    # * RES-1 # "RES" to signify that the lake package is used to represent a reservoir with managed releases and distinct storage accounts (MODSIM will manage the individual storage accounts when that integration happens)
    pth = os.path.join('..', 'data', 'modsim_data')
    
    mvrpack = [['UZF-1'], ['SFR-1'], ['RES-1'], ['WEL-1'], ['DRN-1']]
    maxpackages = len(mvrpack)
    
    # Set up static (used for the entire simulation) SFR -> LAK and LAK -> SFR connections
    static_mvrperioddata = [     # don't forget to use 0-based
                            ('SFR-1', 19, 'RES-1',   0, 'FACTOR',  1.), # Connection is fixed and won't change (reservoir inflow)
                            ('RES-1',  0, 'SFR-1',  20, 'FACTOR',  1.), # Managed release outlet (1 of 2)
                            ('RES-1',  1, 'SFR-1', 742, 'FACTOR',  1.), # Spillway outlet (2 of 2)
                           ]
    
    # Set up all potential UZF -> SFR and UZF -> LAK connections (runoff)
    
    # This function uses the top elevation array and SFR locations to calculate the irunbnd array from the UZF1 package.
    # Of course in MF6, MVR now handles these runoff connections
    irunbnd = modsimBld.determine_runoff_conns_4mvr(pth)  # at this point, irunbnd is 1-based, compensate below
    
    iuzno = 0
    k     = 0             # Hard-wire the layer no.
    for i in range(0, iuzfbnd.shape[0]):
        for j in range(0,iuzfbnd.shape[1]):
            if (i, j) in iuzno_cell_dict:
                iuzno = iuzno_cell_dict[(i, j)]
            if irunbnd[i, j] > 0 and iuzfbnd[i, j] != 0:           # This is a uzf -> sfr connection
                static_mvrperioddata.append(('UZF-1', iuzno, 'SFR-1', irunbnd[i, j] - 1, 'FACTOR', 1.))
            elif irunbnd[i, j] < 0 and iuzfbnd[i, j] != 0:        # This is a uzf -> lak connection
                static_mvrperioddata.append(('UZF-1', iuzno, 'RES-1', 0, 'FACTOR', 1.))
            if ibnda[0, i, j] > 0 and not irunbnd[i, j] < 0:
                drn_idx = drn_dict[(i,j)]  # Everything in drn_dict is 0-based (both the key and value)
                static_mvrperioddata.append(('DRN-1', drn_idx, 'SFR-1', irunbnd[i, j] - 1, 'FACTOR', 1.))
            elif ibnda[0, i, j] > 0 and irunbnd[i, j] < 0:
                drn_idx = drn_dict[(i,j)]
                static_mvrperioddata.append(('DRN-1', drn_idx, 'RES-1', 0, 'FACTOR', 1.))
    
    mvrspd = {0: static_mvrperioddata}
    
    # Set up all potential SFR -> UZF connections (for simulating sw irrig)
    #
    # In addition, setup supplemental pumping (gw irrig) connections: WEL -> UZF
    #
    # These connections need to be established after stress period 184, the
    # start of the irrigation season.  The functions that set these connections
    # up via MVR are contained within an imported script for keeping the script
    # appearing here to a minimum.
    
    mvr_irrig, wel_spd = modsimBld.generate_irrig_events(delr, delc, iuzno_cell_dict, static_mvrperioddata.copy())
    mvrspd.update(mvr_irrig)
    
    maxmvr = 10000    # I don't know, something high
    flopy.mf6.ModflowGwfmvr(gwf, 
                            maxmvr=maxmvr, 
                            print_flows=False,
                            maxpackages=maxpackages, 
                            packages=mvrpack, 
                            perioddata=mvrspd,
                            budget_filerecord=gwfname + '.mvr.bud',
                            filename='{}.mvr'.format(gwfname)
    )
    
    # Instantiating the MODFLOW 6 basic well package (with a concentration auxiliary variable)
    # The WEL package uses the MVR package to deliver pumped water to UZF for irrigation, making this model representative of a conjunctive use system. Thus, pumping represents supplemental groundwater pumping where surface-water delivery shortfalls occur.  Instatiation of the WEL package comes after MVR, since it is the code associated with MVR that establishes how much water to pump.  In other words, only after running the script associated with the MVR connections do we know how much to pump from each well for each daily stress period. 
    #
    # The concentration of the pumped water will be determined by the model once it calculates the groundwater concentrations.  Nevertheless, entering a dummy time series for concentration that consists of zeros.
    # Need to cycle through the WEL data returned by the mvr preparation script and get it into a form acceptable to ModflowGwfwel class
    stress_period_data = {}
    for key, values in wel_spd.items():
        cur_period = flopy.mf6.ModflowGwfwel.stress_period_data.empty(gwf, 
                                                                      maxbound=len(values),
                                                                      boundnames=True,
                                                                      aux_vars=['CONCENTRATION'],
                                                                      timeseries=False)
        for i, wel_x in enumerate(values):
            #                  (       <lay>,       <row>,        <col>,      <Q>, <conc>, <boundname>)
            cur_period[0][i] = ((wel_x[0][0], wel_x[0][1], wel_x[0][2]), wel_x[1],    0.0,    wel_x[2])
        # upload to the dictionary
        stress_period_data[key] = cur_period[0]
    
    maxbound = 250   # The total number 
    flopy.mf6.ModflowGwfwel(gwf, 
                            pname='WEL-1',
                            auxiliary='CONCENTRATION',
                            print_input=True, 
                            print_flows=False,
                            maxbound=maxbound,
                            mover=True,
                            auto_flow_reduce=0.1,
                            stress_period_data=stress_period_data,   # wel_spd established in the MVR setup
                            boundnames=True, 
                            save_flows=True,
                            filename='{}.wel'.format(gwfname)
    )

    objs4gwt = [sft_ts_in, uzf_packagedata]
    return gwfname, objs4gwt


def build_gwt_model(sim, i, objs4gwt, elev_adj, silent=False):
    # ----------------------
    # TRANSPORT MODEL SETUP
    # ----------------------

    # unpack objects that were prepared by the gwf instantiation
    sft_ts_in = objs4gwt[0]
    uzf_packagedata = objs4gwt[1]
    
    # Instantiating MODFLOW 6 groundwater transport package
    gwtname = 'gwt_' + example_name + '_' + str(i + 1)
    gwt = flopy.mf6.MFModel(sim,
                            model_type='gwt6',
                            modelname=gwtname,
                            model_nam_file='{}.nam'.format(gwtname))
    gwt.name_file.save_flows = True
    
    # create iterative model solution and register the gwt model with it  
    imsgwt = flopy.mf6.ModflowIms(sim,
                                  print_option='SUMMARY',
                                  outer_dvclose=hclose,
                                  outer_maximum=nouter,
                                  under_relaxation='NONE',
                                  inner_maximum=ninner,
                                  inner_dvclose=hclose, 
                                  rcloserecord=rclose,
                                  linear_acceleration='BICGSTAB',
                                  scaling_method='NONE',
                                  reordering_method='NONE',
                                  relaxation_factor=relax,
                                  filename='{}.ims'.format(gwtname)
    )
    sim.register_ims_package(imsgwt, [gwt.name])
    
    # Instantiating MODFLOW 6 transport discretization package
    flopy.mf6.ModflowGwtdis(gwt, 
                            nlay=nlay,
                            nrow=nrow,
                            ncol=ncol,
                            delr=delr, 
                            delc=delc,
                            top=top_2x - elev_adj,
                            botm=botm_np - elev_adj,
                            idomain=ibnda,
                            filename='{}.dis'.format(gwtname)
    )
    
    # Instantiating MODFLOW 6 transport initial concentrations
    strta, rch_conc = starting_conc(gwt)
    flopy.mf6.ModflowGwtic(gwt, 
                           strt=strta,
                           filename='{}.ic'.format(gwtname)
    )
    
    # Instantiating MODFLOW 6 transport advection package
    if mixelm == 0:
        scheme = 'UPSTREAM'
    elif mixelm == -1:
        scheme = 'TVD'
    else:
        raise Exception()
    flopy.mf6.ModflowGwtadv(gwt, 
                            scheme=scheme,
                            filename='{}.adv'.format(gwtname)
    )
    
    # Instantiating MODFLOW 6 transport dispersion package
    if al != 0:
        flopy.mf6.ModflowGwtdsp(gwt,
                                alh=al,
                                ath1=ath1,
                                atv=atv,
                                filename='{}.dsp'.format(gwtname)
    )
    
    # Instantiating MODFLOW 6 transport mass storage package (formerly "reaction" package in MT3DMS)
    flopy.mf6.ModflowGwtmst(gwt, 
                            porosity=prsity,
                            first_order_decay=False,
                            decay=None, 
                            decay_sorbed=None,
                            sorption='linear', 
                            bulk_density=rhob, 
                            distcoef=Kd,
                            filename='{}.mst'.format(gwtname)
    )
    
    # Instantiating MODFLOW 6 transport source-sink mixing package
    sourcerecarray = [('DRN-1', 'AUX', 'CONCENTRATION'),
                      ('RCH-1', 'AUX', 'CONCENTRATION'),
                      ('WEL-1', 'AUX', 'CONCENTRATION')]
    flopy.mf6.ModflowGwtssm(gwt, 
                            sources=sourcerecarray,
                            filename='{}.ssm'.format(gwtname)
    )
    
    # Instantiating MODFLOW 6 transport output control package
    flopy.mf6.ModflowGwtoc(gwt,
                           budget_filerecord='{}.cbc'.format(gwtname),
                           concentration_filerecord='{}.ucn'.format(
                               gwtname),
                           concentrationprintrecord=[
                               ('COLUMNS', 10, 'WIDTH', 15,
                                'DIGITS', 6, 'GENERAL')],
                           saverecord=[('CONCENTRATION', 'LAST'),
                                       ('BUDGET', 'LAST')],
                           printrecord=[('CONCENTRATION', 'LAST'),
                                        ('BUDGET', 'LAST')]
    )

    # Start of Advanced Transport Package Instantiations
    # * Unsaturated Zone Flow Package
    # * Streamflow Routing Package
    # * Lake Package
    # * Mover Package
    #
    # Instantiating MODFLOW 6 lake transport (lkt) package
    lktpackagedata = [(0, 800., 'reservoir1')]
    
    lktperioddata = {0: [(0, 'STATUS',      'active'),
                         (0, 'RAINFALL',     0.0),
                         (0, 'EVAPORATION',  0.0),
                         (0, 'RUNOFF',       0.0)]}
    
    # note: for specifying lake number, use fortran indexing!
    lkt_obs = {'{}.lakobs'.format(gwtname): [('resConc',         'concentration', 1),
                                             ('resGwMassExchng', 'lkt',           'reservoir1')]}
    
    flopy.mf6.ModflowGwtlkt(gwt,        # Set time_conversion for use with Manning's eqn.
                            flow_package_name='RES-1',
                            budget_filerecord=gwtname + '.lkt.bud',
                            boundnames=True,
                            save_flows=True,
                            print_input=True,
                            print_flows=False,
                            print_concentration=True,
                            packagedata=lktpackagedata, 
                            lakeperioddata=lktperioddata,
                            observations=lkt_obs, 
                            pname='LKT-1',
                            filename='{}.lkt'.format(gwtname)
    )
    
    # Instantiating MODFLOW 6 streamflow transport (SFT) package
    # gather the pkdat info
    conns = modsimBld.gen_mf6_sfr_connections()
    ndvs, ustrf = modsimBld.tally_ndvs(conns)
    pkdat = modsimBld.gen_sfrpkdata(conns, ndvs, ustrf)

    sftpkdat = []
    for irno in range(len(pkdat['data'])):
        t = (irno, sft_ts_in[0][1], pkdat['data'][irno][12])
        sftpkdat.append(t)
    
    sftspd = {0: [[0, 'INFLOW', 'mainConc'], [555, 'INFLOW', 'tribConc']]}
    
    sft_obs = {(gwtname + '.sftobs', ): [('main_in',         'concentration',   1),  # For now, these need to be 1-based
                                         ('res_in',          'concentration',  20),
                                         ('resRelease',      'concentration',  21),
                                         ('resSpill',        'concentration', 743),
                                         ('priorDiv1',       'concentration',  21),
                                         ('AgArea1_div',     'concentration',  22),
                                         ('AgArea1_lat1',    'concentration',  47),
                                         ('AgArea1_lat2',    'concentration',  72),
                                         ('AgArea1_lat3',    'concentration',  89),
                                         ('AgArea1_lat4',    'concentration', 105),
                                         ('priorDiv2',       'concentration', 136),
                                         ('AgArea2_div',     'concentration', 137),
                                         ('AgArea2_lat1',    'concentration', 177),
                                         ('AgArea2_lat2',    'concentration', 192),
                                         ('AgArea2_lat3',    'concentration', 211),
                                         ('AgArea2_lat4',    'concentration', 233),
                                         ('priorDiv3',       'concentration', 290), 
                                         ('AgArea3_div',     'concentration', 291),
                                         ('AgArea3_lat1',    'concentration', 321),
                                         ('AgArea3_lat2',    'concentration', 342),
                                         ('AgArea3_lat3',    'concentration', 358),
                                         ('AgArea3_lat4',    'concentration', 390),
                                         ('priorDiv4',       'concentration', 430),
                                         ('AgArea4_div',     'concentration', 431),
                                         ('AgArea4_lat1',    'concentration', 466),
                                         ('AgArea4_lat2',    'concentration', 478),
                                         ('AgArea4_lat3',    'concentration', 517),
                                         ('AgArea4_lat4',    'concentration', 555),
                                         ('postag4_lat4',    'concentration', 719),
                                         ('minInstrmQ',      'concentration', 677),
                                         ('modeloutlet',     'concentration', 742), 
                                         ('AgArea3_preLat1', 'concentration', 320),
                                         ('AgArea3_preLat2', 'concentration', 341),
                                         ('AgArea3_preLat3', 'concentration', 357),
                                         ('AgArea3_preLat4', 'concentration', 389)]}
    
    sft = flopy.mf6.modflow.ModflowGwtsft(gwt,
                                          boundnames=True,
                                          flow_package_name='SFR-1',
                                          print_concentration=True,
                                          save_flows=True,
                                          concentration_filerecord=gwtname + '.sft.bin',
                                          budget_filerecord=gwtname + '.sft.bud',
                                          packagedata=sftpkdat,
                                          reachperioddata=sftspd,
                                          observations=sft_obs,
                                          pname='SFT-1',
                                          filename='{}.sft'.format(gwtname)
    )
    
    sft.ts.initialize(filename=gwtname + '_sft_boundary_conc.ts',
                      timeseries=sft_ts_in,
                      time_series_namerecord=['mainConc', 'tribConc'],
                      interpolation_methodrecord=['stepwise', 'stepwise'])
    
    # Instantiating MODFLOW 6 unsaturated zone transport (UZT) package
    uzt_pkdat = []
    uzt_perioddata = []
    uzt_sconc = 1000.
    for i, iuz in enumerate(uzf_packagedata):
        #                <iuzno>, <strtconc>, <boundname>
        uzt_pkdat.append([iuz[0],  uzt_sconc,  iuz[10]])
        
        if iuz[1][0] == 0:  # This tests if a surface cell
            #                     <iuzno>,   <uztsetting>, <conc>
            uzt_perioddata.append([iuz[0], 'INFILTRATION',   0.])  # Sets the concentration associated with precip
    
    uzt_spd = {0: uzt_perioddata}
    
    flopy.mf6.modflow.ModflowGwtuzt(gwt,
                                    flow_package_name='UZF-1',
                                    boundnames=True,
                                    save_flows=True,
                                    print_input=False,
                                    print_flows=False,
                                    print_concentration=True,
                                    concentration_filerecord=gwtname + '.uzt.bin',
                                    budget_filerecord=gwtname + '.uzt.bud',
                                    packagedata=uzt_pkdat,
                                    uztperioddata=uzt_spd,
                                    #observations=uzt_obs,
                                    pname='UZT-1',
                                    filename='{}.uzt'.format(gwtname)
    )
    
    # Instantiating MODFLOW 6 mover transport (MVT) package
    flopy.mf6.modflow.ModflowGwtmvt(gwt, 
                                    save_flows=True, 
                                    budget_filerecord=gwtname + '.mvt.bud',
                                    filename='{}.mvt'.format(gwtname)
    )
    
    # Instantiating MODFLOW 6 flow-model interface (FMI) package
    flopy.mf6.modflow.ModflowGwtfmi(gwt,
                                    flow_imbalance_correction=False,
                                    filename='{}.fmi'.format(gwtname)
    )

    return gwtname


def setup_gwfgwt_exchng(sim, gwfname, gwtname, i):
    # Instantiating MODFLOW 6 flow-transport exchange mechanism
    flopy.mf6.ModflowGwfgwt(sim,
                            exgtype='GWF6-GWT6',
                            exgmnamea=gwfname,
                            exgmnameb=gwtname,
                            filename='{}.gwfgwt'.format(gwfname)
                            )


# ## Function to build models
# MODFLOW 6 flopy simulation object (sim) is returned if building the model
def build_simulation(sim_name, elev_adj, silent=False):
    if config.buildModel:
        
        # MODFLOW 6
        sim_ws = os.path.join(ws, sim_name)
        sim = flopy.mf6.MFSimulation(sim_name=sim_name,
                                     version='mf6',
                                     sim_ws=sim_ws, 
                                     exe_name=mf6exe,
                                     memory_print_option="ALL",
                                     continue_=False
        )
        
        # Instantiating MODFLOW 6 time discretization
        tdis_rc = []
        for i in range(len(rng)):
            tdis_rc.append((perlen[i], nstp[i], tsmult[i]))
        flopy.mf6.ModflowTdis(sim, 
                              nper=nper, 
                              perioddata=tdis_rc, 
                              time_units=time_units
        )
        
        # Setting up two versions of the same model, one downstream of the other
        gwf_model_names = []
        gwt_model_names = []
        for i, itm in enumerate(elev_adj):
            gwfname, objs4gwt = build_gwf_model(sim, i, elev_adj=itm, silent=False)
            gwtname = build_gwt_model(sim, i, objs4gwt, elev_adj=itm, silent=False)
            # Establish flow-transport exchange
            setup_gwfgwt_exchng(sim, gwfname, gwtname, i)

            # Establish flow-flow exchange: Collect the gwf names in a list
            if gwfname not in gwf_model_names:
                gwf_model_names.append(gwfname)

            # Establish transport-transport exchange: Collect the gwt names in a list
            if gwtname not in gwt_model_names:
                gwt_model_names.append(gwtname)
        
        # After all the models are created, connect gwf to gwf and gwt to gwt
        # Start by setting up exchange data
        gwf_model1 = sim.get_model(gwf_model_names[0])
        gwf_model2 = sim.get_model(gwf_model_names[1])
        idom1 = gwf_model1.dis.idomain
        idom2 = gwf_model2.dis.idomain
        idom1_dat = idom1.get_data()
        idom2_dat = idom2.get_data()

        # Matchup the downstream (right) edge of the upstream model with the
        # upstream (left) edge of the downstream model
        j=0
        exchng_conn = []
        ihc = 1
        cl1 = 200
        cl2 = 200
        hwva = 400
        for k in range(idom1_dat.shape[0]):
            for i in range(idom1_dat.shape[1]):
                #        upstream model                downstream model
                #        --------------                ----------------
                if idom1_dat[k, i, j - 1] != 0 and idom2_dat[k, i, -j] != 0:
                    exchng_conn.append(((k, i, ncol - 1), (k, i, j), ihc, cl1, cl2, hwva))

        gwfgwf = flopy.mf6.ModflowGwfgwf(sim,
                                         exgtype='GWF6-GWF6',
                                         print_flows=True,
                                         print_input=True,
                                         exgmnamea=gwf_model_names[0],
                                         exgmnameb=gwf_model_names[1],
                                         nexg=len(exchng_conn),
                                         exchangedata=exchng_conn,
                                         mvr_filerecord='{}.mvr'.format(sim_name),
                                         pname='EXG-1',
                                         filename='{}.exg'.format(sim_name))

        # Instantiate model-to-model MVR package (for SFR conn)
        mvrpack = [[gwf_model_names[0], 'SFR-1'], [gwf_model_names[1], 'SFR-1']]
        maxpackages = len(mvrpack)

        # Set up static SFR-to-SFR connections that remains fixed for entire simulation
        static_mvrperioddata = [  # don't forget to use 0-based values
            [gwf_model_names[0], 'SFR-1', 731, gwf_model_names[1], 'SFR-1', 0, 'FACTOR', 1.]
        ]

        mvrspd = {0: static_mvrperioddata}
        maxmvr = 1
        mvr = flopy.mf6.ModflowMvr(sim,
                                   modelnames=True,
                                   maxmvr=maxmvr,
                                   print_flows=True,
                                   maxpackages=maxpackages,
                                   packages=mvrpack,
                                   perioddata=mvrspd,
                                   filename='{}.mvr'.format(sim_name))

        return sim
    return None


# ## Function to write model files
def write_simulation(sim, silent=False):
    if config.writeModel:
        sim.write_simulation(silent=silent)


# ## Function to run the model
# _True_ is returned if the model runs successfully
def run_model(sim, silent=False):
    success = True
    if config.runModel:
        success, buff = sim.run_simulation(silent=False)
        if not success:
            print(buff)
    return success


# ## Function to plot the model results
def plot_results(mf6, idx, ax=None):
    if config.plotModel:
        mf6_out_path = mf6.simulation_data.mfpath.get_sim_path()
        mf6.simulation_data.mfpath.get_sim_path()
        
        # Get the MF6 concentration output
        fname_mf6 = os.path.join(mf6_out_path, list(mf6.model_names)[1] + '.ucn')
        ucnobj_mf6 = flopy.utils.HeadFile(fname_mf6, precision='double',
                                          text='CONCENTRATION')
        
        times_mf6 = ucnobj_mf6.get_times() 
        conc_mf6 = ucnobj_mf6.get_alldata()
        
        # Read LAK observation file
        lakobs = pd.read_csv(os.path.join(mf6_out_path,'{}.lakobs'.format(list(mf6.model_names)[0])),header=0)
        lakobs['Date'] = rng
        lakobs.set_index('Date', inplace=True)
        
        # Read SFR observation file
        sfrobs = pd.read_csv(os.path.join(mf6_out_path,'{}.sfrobs'.format(list(mf6.model_names)[0])),header=0)
        sfrobs['Date'] = rng
        sfrobs.set_index('Date', inplace=True)
        
        # Upload model budget
        modobj = bf.CellBudgetFile(os.path.join(mf6_out_path,'{}.bud'.format(list(mf6.model_names)[0])), precision='double')
        
        # Create figure for scenario 
        fs = USGSFigure(figure_type="graph", verbose=False)
        sim_name = mf6.name
        plt.rcParams['lines.dashed_pattern'] = [5.0, 5.0]
        
        # Convert time series to familiar units (ac*ft & cfs)
        lakobs['LAKEVOL_acft'] = lakobs['LAKEVOL'] * 35.315 / 43560
        lakobs['LAKEXIT_cfs'] = lakobs['LAKEXIT'] * 35.315 / 86400
        lakobs['LAKMVR_IN_cfs'] = lakobs['LAKMVR_IN'] * 35.315 / 86400
        lakobs['LAKMVR_OUT_cfs'] = lakobs['LAKMVR_OUT'] * 35.315 / 86400
        
        # ---------------------------------
        # Plot 1 - Reservoir water budget
        # ---------------------------------
        fig = plt.figure(figsize=figure_size)
        ax = lakobs.LAKEVOL_acft.plot(color='b', label='Reservoir Storage')
        ax2 = lakobs.LAKMVR_IN_cfs.plot(color='g', label='Lake Inflow', secondary_y=True)
        ax2 = lakobs.LAKMVR_OUT_cfs.plot(color='m', label='Lake Outflow', secondary_y=True)
        
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], loc='upper right')
        
        ax2.axhline(y=0.0, color='k', linestyle='-') # add horizontal line = 0 to secondary y-axis
        ax.set_ylim(0, 85000)
        ax.set_xlabel('Date')
        ax.set_ylabel('Storage, in ac-ft')
        ax2.set_ylabel('Flow, in cfs')
        ax2.set_ylim(-410,410)
        title = 'Reservoir Water Budget'
        
        letter = chr(ord("@") + idx + 1)
        fs.heading(letter=letter, heading=title)
        
        # save figure
        if config.plotSave:
            fpth = os.path.join(
                "..", "figures", "{}{}".format(sim_name + "res_budget", config.figure_ext)
            )
            fig.savefig(fpth)
        
        # ----------------------------------
        # Plot 2 - Reservoir inflow/outflow
        # ----------------------------------
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(sfrobs['time'], abs(sfrobs['MAIN_IN']) * 35.315 / 86400, color='k', label='Primary Inflow')
        ax.plot(sfrobs['time'], abs(sfrobs['RESSPILL']) * 35.315 / 86400, '.', color='b', label='Spillway Flow')
        ax.plot(sfrobs['time'], abs(sfrobs['RESRELEASE']) * 35.315 / 86400, '--', color='r', label='Managed Release')
        
        ax.set_ylim(0, 550)
        ax.set_xlim(0, 370)
        ax.set_xlabel('Time, in days')
        ax.set_ylabel('Flow, in cfs')
        title = "Simulated Flows"
        ax.legend()   
        
        letter = chr(ord("@") + idx + 2)        
        fs.heading(letter=letter, heading=title)
        
        # save figure
        if config.plotSave:
            fpth = os.path.join(
                "..", "figures", "{}{}".format(sim_name + "res_inOut", config.figure_ext)
            )
            fig.savefig(fpth)
        
        # ----------------------------------
        # Plot 3 - Main Diversions
        # ----------------------------------
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(abs(sfrobs['AGAREA1_DIV']) * 35.315 / 86400, color='k', label='Command Area 1')
        ax.plot(abs(sfrobs['AGAREA2_DIV']) * 35.315 / 86400, '.', color='b', label='Command Area 2')
        ax.plot(abs(sfrobs['AGAREA3_DIV']) * 35.315 / 86400, '--', color='r', label='Command Area 3')
        ax.plot(abs(sfrobs['AGAREA4_DIV']) * 35.315 / 86400, '--', color='m', label='Command Area 4')
        ax.plot(abs(sfrobs['MININSTRMQ']) * 35.315 / 86400, '--', color='g', label='Minimum Instream Flow Requirement')
        ax.set_ylim(0, 200)
        ax.set_xlabel('Date')
        ax.set_ylabel('Flow, in cfs')
        title = "Simulated Diversions from Mainstem River"
        ax.legend()
        
        letter = chr(ord("@") + idx + 3)
        fs.heading(letter=letter, heading=title)
        
        # save figure
        if config.plotSave:
            fpth = os.path.join(
                "..", "figures", "{}{}".format(sim_name + "main_divs", config.figure_ext)
            )
            fig.savefig(fpth)
        
        # ----------------------------------
        # Plot 4 - CA1 Lateral Diversions
        # ----------------------------------
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(abs(sfrobs['AGAREA1_DIV']) * 35.315 / 86400, color='k', label='Command Area 1')
        ax.plot(abs(sfrobs['AGAREA1_LAT1']) * 35.315 / 86400, '--', color='b', label='Lateral 1')
        ax.plot(abs(sfrobs['AGAREA1_LAT2']) * 35.315 / 86400, '--', color='r', label='Lateral 2')
        ax.plot(abs(sfrobs['AGAREA1_LAT3']) * 35.315 / 86400, '^', color='m', label='Lateral 3')
        ax.plot(abs(sfrobs['AGAREA1_LAT3']) * 35.315 / 86400, '--', color='g', label='Lateral 4',linewidth=2.0)
        
        ax.set_ylim(0, 100)
        ax.set_xlabel('Date')
        ax.set_ylabel('Flow, in cfs')
        title = "Command Area 1 Lateral Delivery"
        ax.legend()
        
        letter = chr(ord("@") + idx + 4)
        fs.heading(letter=letter, heading=title)
        
        # save figure
        if config.plotSave:
            fpth = os.path.join(
                "..", "figures", "{}{}".format(sim_name + "CA1_latDiv", config.figure_ext)
            )
            fig.savefig(fpth)
        
        # ----------------------------------
        # Plot 5 - CA2 Lateral Diversions
        # ----------------------------------
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(abs(sfrobs['AGAREA2_DIV']) * 35.315 / 86400, color='k', label='Command Area 2')
        ax.plot(abs(sfrobs['AGAREA2_LAT1']) * 35.315 / 86400, '--', color='b', label='Lateral 1')
        ax.plot(abs(sfrobs['AGAREA2_LAT2']) * 35.315 / 86400, '--', color='r', label='Lateral 2')
        ax.plot(abs(sfrobs['AGAREA2_LAT3']) * 35.315 / 86400, '--', color='m', label='Lateral 3')
        ax.plot(abs(sfrobs['AGAREA2_LAT4']) * 35.315 / 86400, '--', color='g', label='Lateral 4')
        
        ax.set_ylim(0, 100)
        ax.set_xlabel('Date')
        ax.set_ylabel('Flow, in cfs')
        title = "Command Area 2 Lateral Delivery"
        ax.legend()

        letter = chr(ord("@") + idx + 5)
        fs.heading(letter=letter, heading=title)
        
        # save figure
        if config.plotSave:
            fpth = os.path.join(
                "..", "figures", "{}{}".format(sim_name + "CA2_latDiv", config.figure_ext)
            )
            fig.savefig(fpth)
        
        # ----------------------------------
        # Plot 6 - CA3 Lateral Diversions
        # ----------------------------------
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(abs(sfrobs['AGAREA3_DIV']) * 35.315 / 86400, color='k', label='Command Area 3')
        ax.plot(abs(sfrobs['AGAREA3_LAT1']) * 35.315 / 86400, '-', color='b', label='Lateral 1')
        ax.plot(abs(sfrobs['AGAREA3_LAT2']) * 35.315 / 86400, '--', color='r', label='Lateral 2')
        ax.plot(abs(sfrobs['AGAREA3_LAT3']) * 35.315 / 86400, '--', color='m', label='Lateral 3')
        ax.plot(abs(sfrobs['AGAREA3_LAT4']) * 35.315 / 86400, '--', color='g', label='Lateral 4')
        
        ax.set_ylim(0, 300)
        ax.set_xlabel('Date')
        ax.set_ylabel('Flow, in cfs')
        title = "Command Area 3 Lateral Delivery"
        ax.legend(loc='upper left')
        
        letter = chr(ord("@") + idx + 6)
        fs.heading(letter=letter, heading=title)
        
        # save figure
        if config.plotSave:
            fpth = os.path.join(
                "..", "figures", "{}{}".format(sim_name + "CA3_latDiv", config.figure_ext)
            )
            fig.savefig(fpth)
        
        # ----------------------------------
        # Plot 7 - CA4 Lateral Diversions
        # ----------------------------------
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(abs(sfrobs['AGAREA4_DIV']) * 35.315 / 86400, color='k', label='Command Area 4')
        ax.plot(abs(sfrobs['AGAREA4_LAT1']) * 35.315 / 86400, '--', color='b', label='Lateral 1')
        ax.plot(abs(sfrobs['AGAREA4_LAT2']) * 35.315 / 86400, '--', color='r', label='Lateral 2')
        ax.plot(abs(sfrobs['AGAREA4_LAT3']) * 35.315 / 86400, '--', color='m', label='Lateral 3')
        ax.plot(abs(sfrobs['AGAREA4_LAT4']) * 35.315 / 86400, '--', color='g', label='Lateral 4')
        
        ax.set_ylim(0, 200)
        ax.set_xlabel('Date')
        ax.set_ylabel('Flow, in cfs')
        title = "Command Area 4 Lateral Delivery"
        ax.legend()
        
        letter = chr(ord("@") + idx + 7)
        fs.heading(letter=letter, heading=title)
        
        # save figure
        if config.plotSave:
            fpth = os.path.join(
                "..", "figures", "{}{}".format(sim_name + "CA4_latDiv", config.figure_ext)
            )
            fig.savefig(fpth)
        
        # -----------------------------------------
        # Plot 8 - Concentrations in the reservoir
        # -----------------------------------------
        # For the time being, need to screen the concentrations where there is no flow
        # Read SFT observation file
        sftobs = pd.read_csv(os.path.join(mf6_out_path,'{}.sftobs'.format(list(mf6.model_names)[1])),header=0)
        sftobs['Date'] = rng
        sftobs.set_index('Date', inplace=True)
        
        # Read LKT observation file
        lktobs = pd.read_csv(os.path.join(mf6_out_path,'{}.lakobs'.format(list(mf6.model_names)[1])),header=0)
        lktobs['Date'] = rng
        lktobs.set_index('Date', inplace=True)
        
        fig = plt.figure(figsize=(14,6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(sftobs['RES_IN'], color='k', label='Concentration of water entering the reservoir')
        ax.plot(lktobs['RESCONC'], color='b', label='Concentration of instantaneously mixed reservoir')
        ax.plot(sftobs['RESRELEASE'], '--', color='r', label='Concentration of released water')
        ax.plot(sftobs['MAIN_IN'], color='g', label='Specified Concentration of water entering the model')
        
        ax.set_ylim(0, 1500)
        ax.set_xlabel('Date')
        ax.set_ylabel('Concentration, in mg/L')
        title = "Simulated Concentrations"
        ax.legend()
        
        letter = chr(ord("@") + idx + 8)
        fs.heading(letter=letter, heading=title)
        
        # save figure
        if config.plotSave:
            fpth = os.path.join(
                "..", "figures", "{}{}".format(sim_name + "resConcs", config.figure_ext)
            )
            fig.savefig(fpth)
        
        # ------------------------------------------------
        # Plot 9 - Check concentrations of diverted water
        # ------------------------------------------------
        # Adjust some of the output for clean presentation
        sftobs['AGAREA1_DIV'].loc[sfrobs['AGAREA1_DIV'] == 0] = np.nan
        
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(sftobs['PRIORDIV1'], color='k', label='Concentration of river upstream of Diversion CA1')
        ax.plot(sftobs['AGAREA1_DIV'],   color='r', linewidth=3.0, label='Concentration of diverted water')
        #ax.plot(sftobs['RESRELEASE'],'--', color='r', label='Concentration of released water')
        ax.set_ylim(0, 1500)
        ax.set_xlabel('Date')
        ax.set_ylabel('Concentration, in mg/L')
        title = "Simulated Concentrations"
        ax.legend()
        
        letter = chr(ord("@") + idx + 9)
        fs.heading(letter=letter, heading=title)
        
        # save figure
        if config.plotSave:
            fpth = os.path.join(
                "..", "figures", "{}{}".format(sim_name + "divConcCA1", config.figure_ext)
            )
            fig.savefig(fpth)
        
        # ---------------------------------------------------------------
        # Plot 10 - Plot up concentrations in river for snapshot in time
        # ---------------------------------------------------------------
        # First, need to retrieve some output       
        
        # Attempt a plot of river (mainstem) concentrations for each reach of 
        # the river. X-axis would be the increasing reach number with gaps 
        # removed. Perhaps add line plots for two different times. And perhaps 
        # show the gw/sw exchange on the plot as well (2nd y-axis). At this 
        # point in MF6-transport's development, it is necessary to screen 
        # concentrations where there are no flows. To do this, we need to read 
        # in the binary sfr budget file and process it for flows. Running the 
        # following command: sfr_flow.get_unique_record_names() will yield:
        # * FLOW-JA-FACE
        # * GWF
        # * RAINFALL
        # * EVAPORATION
        # * RUNOFF
        # * EXT-INFLOW
        # * EXT-OUTFLOW
        # * STORAGE
        # * FROM-MVR
        # * TO-MVR
        # Use 'FLOW-JA-FACE' to approximate the flow at the midpoint. The data 
        # returned by .get_data(..., text='FLOW-JA-FACE') will be a list. Each 
        # item of the list is from node, to node, Q, and flow area.
        
        # cycle through all sfr/sft stream cell listings and pick out the mainstem.  
        # Create a logical array where mainstem==1
        
        # The script below leverages pkdat and because it doesn't get passed in, 
        # need to regenerate it
        conns        = modsimBld.gen_mf6_sfr_connections()  
        ndvs, ustrf  = modsimBld.tally_ndvs(conns)
        pkdat        = modsimBld.gen_sfrpkdata(conns, ndvs, ustrf)
        
        main_riv = np.zeros((len(pkdat['data'])))
        for i, itm in enumerate(pkdat['data']):
            if('Mainstem' in itm[12]):
                main_riv[i] = 1
        
        # Create the object pointing the binary file with all of the SFR flows
        sfr_flow_all = 'gwf_modsim.sfr.bud'
        sfr_flow = bf.CellBudgetFile(os.path.join(mf6_out_path, sfr_flow_all), precision='double')  
        
        ckstpkper = sfr_flow.get_kstpkper()
        strm_Q = []
        for i, kstpkper in enumerate(ckstpkper):
            strm_Q_step = sfr_flow.get_data(kstpkper=kstpkper, text='    FLOW-JA-FACE')
            strm_Q.append(strm_Q_step)
        
        # Loop for each time step, then for each reach on the main river
        main_riv_flowing = np.zeros((len(strm_Q), len(pkdat['data'])))
        strm_Q_vals = np.zeros((len(strm_Q), len(pkdat['data'])))
        for tm in range(len(strm_Q)):
            cur_stp = strm_Q[tm][0]
            # remember that entries in cur_stp will be indexed by fortran and therefore 1-based
            itms = [entry for entry in cur_stp if (main_riv[(entry[0]-1)] == 1 and main_riv[(entry[1]-1)] == 1 and entry[1] > entry[0])]
            for i, itm in enumerate(itms):
                if abs(itm[2]) > 0:        # This condition would mean the stream is flowing
                    main_riv_flowing[tm, (itm[0]-1)] = 1
                    strm_Q_vals[tm, (itm[0]-1)] = abs(itm[2])
        
        sft_conc_file = 'gwt_modsim.sft.bin'
        sft_conc = bf.HeadFile(os.path.join(mf6_out_path,sft_conc_file), text='CONCENTRATION', precision='double')
        
        ckstpkper = sft_conc.get_kstpkper()
        strm_conc = []
        for i, kstpkper in enumerate(ckstpkper):
            strm_conc_step = sft_conc.get_data(kstpkper = kstpkper)
            strm_conc.append(strm_conc_step)
        
        strm_conc = np.array(strm_conc)
        # Apply mask of np.nan when/where river flow is 0
        for tm in range(strm_conc.shape[0]):
            for ipos in range(strm_conc.shape[3]):
                if main_riv_flowing[tm, ipos] == 0:
                    strm_conc[tm,0,0,ipos] = np.nan
        
        print(strm_conc.shape)
        
        tm1 = 184
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(1, 1, 1)
        ln1 = ax.plot(strm_conc[tm1,0,0,main_riv==1], '.', color='r', label='Concentration at River Cell Position i')
        ax.set_xlabel('River Cell Position (upstream to downstream)')
        ax.set_ylabel('River Concentration (mg/L)')
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 4000)
        title = "River Conditions at Stess Period " + str(tm1)
        
        # Mark the location of the reservoir on the working plot
        ax.axvline(x=19)
        ax.text(16, 3250, "Reservoir", ha='left', rotation=90)
        
        # Mark the location of where CA2 diverts out of the river
        ax.axvline(x=24)
        ax.text(25, 3150, "CA2 Diversion", ha='left', rotation=90)
        
        # Mark the location of where Drain 1 dumping into the river
        ax.axvline(x=47)
        ax.text(44, 3150, "Drain #1 conf.", ha='left', rotation=90)
        
        # Mark the location of where CA3 diverts out of the river
        ax.axvline(x=53)
        ax.text(50, 3150, "CA3 Diversion", ha='left', rotation=90)
        
        # Mark the location of where CA3 diverts out of the river
        ax.axvline(x=65)
        ax.text(62, 3150, "Drain #2 conf.", ha='left', rotation=90)
        
        # Mark the location of where CA4 diverts out of the river
        ax.axvline(x=82)
        ax.text(79, 3150, "CA4 Diversion", ha='left', rotation=90)
        
        # Mark the location of where CA2 returns to the river
        ax.axvline(x=101)
        ax.text(97, 3250, "CA2 Return", ha='left', rotation=90)
        
        # Mark the location of where CA3 returns to the river
        ax.axvline(x=114)
        ax.text(111, 3250, "CA3 Return", ha='left', rotation=90)
        
        # Mark the location of the confluence with the tributary
        ax.axvline(x=121)
        ax.text(118, 3000, "Tributary Inflow", ha='left', rotation=90)
        
        # Mark the location of where CA4 returns to the river
        ax.axvline(x=162)
        ax.text(159, 3250, "CA4 Return", ha='left', rotation=90)
        
        ax2 = ax.twinx()
        
        ln2 = ax2.plot(strm_Q_vals[tm1, main_riv==1] * 35.315 / 86400, '.', color='b', label='Flow at River Cell Position i')
        ax2.set_ylim(0, 400)
        ax2.set_ylabel('River Flow (cfs)')
        
        # Put a legend to the right of the current axis
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
        ax.legend(lns, labs, loc='upper right', bbox_to_anchor=(1, 1.15))
        
        letter = chr(ord("@") + idx + 10)
        fs.heading(letter=letter, heading=title)
        
        # save figure
        if config.plotSave:
            fpth = os.path.join(
                "..", "figures", "{}{}".format(sim_name + "rivConc", config.figure_ext)
            )
            fig.savefig(fpth)
        
        # ------------------------------------------------
        # Plot 11 - Plot depth of saturated zone
        # ------------------------------------------------
        # Working up a 2D plot of groundwater and unsaturated zone concentrations
        hed_file = 'gwf_modsim.hds'
        hds = bf.HeadFile(os.path.join(mf6_out_path, hed_file), precision='double')
        
        ckstpkper = hds.get_kstpkper()
        hd = []
        for i, kstpkper in enumerate(ckstpkper):
            heads_step = hds.get_data(kstpkper=kstpkper)
            hd.append(heads_step)
        
        hd = np.array(hd)
        
        depth = np.zeros((hd.shape[0], hd.shape[2], hd.shape[3]))
        for tm, kstpkper in enumerate(ckstpkper):
            depth[tm, :, :] = top - hd[tm,1,:,:]
        
        depth[depth<=0] = np.nan
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(depth[200,:,:], cmap='jet')
        title = "Depth to water table"
        cbar = plt.colorbar(shrink=0.7)
        cbar.ax.set_title('Depth, m', pad=20)
        
        letter = chr(ord("@") + idx + 11)
        fs.heading(letter=letter, heading=title)
        
        # save figure
        if config.plotSave:
            fpth = os.path.join(
                "..", "figures", "{}{}".format(sim_name + "gwDepth", config.figure_ext)
            )
            fig.savefig(fpth)
        
        # ---------------------------------------------------------
        # Plot 12 - Plot concentration of saturated zone (layer 2)
        # ---------------------------------------------------------
        # Get the concentrations of the saturated zone
        ucn_file = 'gwt_modsim.ucn'
        gw_conc = bf.HeadFile(os.path.join(mf6_out_path, ucn_file), text='CONCENTRATION', precision='double')
        sat_conc = []
        for i, kstpkper in enumerate(ckstpkper):
            conc_step = gw_conc.get_data(kstpkper = kstpkper)
            sat_conc.append(conc_step)
        
        sat_conc = np.array(sat_conc)
        
        uzt_conc_file = 'gwt_modsim.uzt.bin'
        uz_conc = bf.HeadFile(os.path.join(mf6_out_path, uzt_conc_file), text='CONCENTRATION', precision='double')
        unsat_conc = []
        for i, kstpkper in enumerate(ckstpkper):
            uzconc_step = uz_conc.get_data(kstpkper = kstpkper)
            unsat_conc.append(uzconc_step)
        
        unsat_conc = np.array(unsat_conc)
        
        # For plotting up the saturated zone concentrations, use the following masking routine:
        # (check for heads below cell bottoms)
        for i, kstpkper in enumerate(ckstpkper):
            sat_conc[i,np.logical_or(ibnda==0, hd[i,:,:,:] < botm)] = np.nan
        
        map_layer = 1  # Remember, 0-based
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(sat_conc[360, map_layer, :, :], cmap='jet')
        title = 'Layer ' + str(map_layer + 1) + ' Saturated Concentrations'
        cbar = plt.colorbar(shrink=0.7)
        cbar.ax.set_title('TDS, mg/L', pad=20)
        
        letter = chr(ord("@") + idx + 12)
        fs.heading(letter=letter, heading=title)
        
        # save figure
        if config.plotSave:
            fpth = os.path.join(
                "..", "figures", "{}{}".format(sim_name + "concLay2", config.figure_ext)
            )
            fig.savefig(fpth)
        
        # ---------------------------------------------------------
        # Plot 13 - Plot concentration of saturated zone (layer 1)
        # ---------------------------------------------------------
        map_layer = 0  # Remember, 0-based
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(sat_conc[360, map_layer, :, :], cmap='jet')
        title = "Layer " + str(map_layer + 1) + " Saturated Concentrations"
        cbar = plt.colorbar(shrink=0.7)
        cbar.ax.set_title('TDS, mg/L', pad=20)
        plt.scatter(54, 10, s=100, c='blue', marker='x')
        
        letter = chr(ord("@") + idx + 13)
        fs.heading(letter=letter, heading=title)
        
        # save figure
        if config.plotSave:
            fpth = os.path.join(
                "..", "figures", "{}{}".format(sim_name + "concLay1", config.figure_ext)
            )
            fig.savefig(fpth)
        
        # -----------------------------------------------------------
        # Plot 14 - Plot concentration of unsaturated zone (layer 1)
        # -----------------------------------------------------------
        iuzfbnd = np.loadtxt(os.path.join('..','data','modsim_data','uzf_support','iuzfbnd.txt'))
        uzt_conc = np.zeros_like(sat_conc)
        for tm in range(unsat_conc.shape[0]):
            idx_tmp = 0
            for k in range(ibnda.shape[0]):
                for i in range(iuzfbnd.shape[0]):
                    for j in range(iuzfbnd.shape[1]):
                        if iuzfbnd[i, j] == 1:
                            uzt_conc[tm, k, i, j] = unsat_conc[tm,0,0,idx_tmp]
                            idx_tmp += 1
        
        uzt_conc[uzt_conc==0] = np.nan
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(uzt_conc[360,0,:,:], cmap='jet')
        title = "Layer 1 Unsaturated Concentrations"
        cbar = plt.colorbar(shrink=0.7)
        cbar.ax.set_title('TDS, mg/L', pad=20)
        #            x   y
        #           --  --
        plt.scatter(54, 10, s=100, c='blue', marker='x')
        plt.scatter(58, 10, s=100, c='blue', marker='x')
        plt.scatter(62, 10, s=100, c='blue', marker='x')
        
        letter = chr(ord("@") + idx + 14)
        fs.heading(letter=letter, heading = title)
        
        # save figure
        if config.plotSave:
            fpth = os.path.join(
                "..", "figures", "{}{}".format(sim_name + "uzConcLay1", config.figure_ext)
            )
            fig.savefig(fpth)
        
        # -----------------------------------------------------------
        # Plot 15 - Plot delivered surface water
        # -----------------------------------------------------------
        # Post-processing the MVR budget file (.mvr.bud)
        # Would be cool to try and plot up a 3D barplot for a cluster of 
        # irrigated fields that receive irrigation water. Would be even 
        # cooler to use stacked bars to delineate the delivery differences 
        # between groundwater and surface-water.
        # Get MVR values
        mvr_output = 'gwf_modsim.mvr.bud'
        verbose=False
        if verbose:
            # The following, which has 'verbose=True' provides a clue as to what is in the CellBudget object
            mvr_Q = bf.CellBudgetFile(os.path.join(mf6_out_path, mvr_output), verbose=True, precision='double')
        else:
            # Otherwise, to keep the output more succinct, use:
            mvr_Q = bf.CellBudgetFile(os.path.join(mf6_out_path, mvr_output), verbose=False, precision='double')
        
        # This retrieves all of the MOVER fluxes:
        # ---------------------------------------
        mvr_all = mvr_Q.get_data(text='      MOVER-FLOW')
        
        # In addition to using the verbose option above, could use the following to see the contents of 
        # the recarray that tells us what's in each entry. Can access specific columns with 
        #mvr_Q.list_records()
        # Also have something like the following to get a true/false 1D array for extracting from the recarray
        #mvr_Q.recordarray['paknam'] == b'UZF-1           '
        # For more, see below, e.g., np.where... etc.
        
        ckstpkper = mvr_Q.get_kstpkper()
        sw_irr_arr = np.zeros((nper, nrow, ncol))        # surface water irrigation
        gw_irr_arr = np.zeros((nper, nrow, ncol))        # groundwater irrigation
        precip_arr = np.zeros((nper, nrow, ncol))        # rainfall
        uzet_arr   = np.zeros((nper, nlay, nrow, ncol))  # uzet
        mvr_uzf_rej_inf_vals = []
        mvr_drn_gw_disQ      = []
        
        
        # Get all rejected infiltration:
        provider  = b'UZF-1           '
        receiver1 = b'SFR-1           '
        receiver2 = b'RES-1           '
        rej_inf_spat = []
        for i, kstpkper in enumerate(ckstpkper):
            rej_inf_spat_tmp = np.zeros((nrow, ncol))
            mvr_rejinf = 0  # Initialize
            # The following gets the actual indexes of where the conditions are satisfied, which is what the recarray needs
            mvr_idxs = np.where(np.all((np.array(mvr_Q.recordarray['kper'] == i), 
                                        np.array(mvr_Q.recordarray['paknam'] == provider),        # Provider
                                        np.logical_or(mvr_Q.recordarray['paknam2'] == receiver1,  # Receiver
                                                      mvr_Q.recordarray['paknam2'] == receiver2)), axis=0))
            # For each index (there will be two in this case because there are two receivers)
            for j in range(len(mvr_idxs[0])):
                mvr_uzf2sw = mvr_all[mvr_idxs[0][j]]
                # Loop through each row of the mvr_uzf2sw for tallying flow totals
                for k, itm in enumerate(mvr_uzf2sw):
                    rw, cl = iuzno_dict_rev[itm[0]]             
                    rej_inf_spat_tmp[rw, cl] += float(itm[2])
                    mvr_rejinf += itm[2]
            
            rej_inf_spat.append(rej_inf_spat_tmp)  
            # Tally the result
            mvr_uzf_rej_inf_vals.append(mvr_rejinf)
        
        rej_inf_spat = np.array(rej_inf_spat)    
        print('Finished tallying rejected infiltration')
        
        # Get all groundwater discharge:
        provider  = b'DRN-1           '
        receiver1 = b'SFR-1           '
        receiver2 = b'RES-1           '
        gwd_spat = []
        for i, kstpkper in enumerate(ckstpkper):
            gwd_spat_tmp = np.zeros((nrow, ncol))
            mvr_gwdisQ = 0  # Initialize
            # The following gets the actual indexes of where the conditions are satisfied, which is what the recarray needs
            mvr_idxs = np.where(np.all((np.array(mvr_Q.recordarray['kper'] == i), 
                                        np.array(mvr_Q.recordarray['paknam'] == provider),        # Provider
                                        np.logical_or(mvr_Q.recordarray['paknam2'] == receiver1,  # Receiver
                                                      mvr_Q.recordarray['paknam2'] == receiver2)), axis=0))
            # For each index (there will be two in this case because there are two receivers)
            for j in range(len(mvr_idxs[0])):
                mvr_drn2sw = mvr_all[mvr_idxs[0][j]]
                # Loop through each row of the mvr_uzf2sw for tallying flow totals
                for k, itm in enumerate(mvr_drn2sw):
                    rw, cl = iuzno_dict_rev[itm[0]]
                    gwd_spat_tmp[rw, cl] += float(itm[2])
                    mvr_gwdisQ += itm[2]
            
            gwd_spat.append(gwd_spat_tmp)
            # Tally the result
            mvr_drn_gw_disQ.append(mvr_gwdisQ)
        
        gwd_spat = np.array(gwd_spat)
        
        print('Finished tallying groundwater discharge')
        
        # Get all surface-water derived irrigation events (store as 3D array: [time, row, col]):
        provider  = b'SFR-1           '
        receiver1 = b'UZF-1           '
        for i, kstpkper in enumerate(ckstpkper):
            # The following gets the actual indexes of where the conditions are satisfied, which is what the recarray needs
            mvr_idxs = np.where(np.all((np.array(mvr_Q.recordarray['kper'] == i), 
                                        np.array(mvr_Q.recordarray['paknam']  == provider),            # Provider
                                        np.array(mvr_Q.recordarray['paknam2'] == receiver1)), axis=0)) # Receiver
        
            if len(mvr_idxs[0]) > 0:
                mvr_sfrirrig = mvr_all[mvr_idxs[0][0]]
                # Loop through each row of the mvr_sfrirrig recarray for tallying sw irrigation events on a cell-by-cell basis
                for k, itm in enumerate(mvr_sfrirrig):
                    iuz_row_col_tup = iuzno_dict_rev[itm[1]]   # itm[1]: the receiver identifier, in this case iuzno, need to relate to row/col address
                    sw_irr_arr[i, iuz_row_col_tup[0], iuz_row_col_tup[1]] += itm[2]
        
        print('Finished tallying surface water irrigation')
        
        # Get all groundwater derived irrigation events (store as 3D array: [time, row, col]):
        provider  = b'WEL-1           '
        receiver1 = b'UZF-1           '
        for i, kstpkper in enumerate(ckstpkper):
            # The following gets the actual indexes of where the conditions are satisfied, which is what the recarray needs
            mvr_idxs = np.where(np.all((np.array(mvr_Q.recordarray['kper'] == i), 
                                        np.array(mvr_Q.recordarray['paknam']  == provider),            # Provider
                                        np.array(mvr_Q.recordarray['paknam2'] == receiver1)), axis=0)) # Receiver
        
            if len(mvr_idxs[0]) > 0:
                mvr_welirrig = mvr_all[mvr_idxs[0][0]]
                # Loop through each row of the mvr_sfrirrig recarray for tallying sw irrigation events on a cell-by-cell basis
                for k, itm in enumerate(mvr_welirrig):
                    iuz_row_col_tup = iuzno_dict_rev[itm[1]]   # itm[1]: the receiver identifier, in this case iuzno, need to relate to row/col address
                    gw_irr_arr[i, iuz_row_col_tup[0], iuz_row_col_tup[1]] += itm[2]
        
        print('Finished tallying groundwater irrigation')
        
        # Get the precipitation amounts on a cell-by-cell basis
        uzf_bud_file = 'gwf_modsim.uzf.bud'
        uzfobj = bf.CellBudgetFile(os.path.join(mf6_out_path, uzf_bud_file), precision='double')
        for i, kstpkper in enumerate(ckstpkper):
            infil = uzfobj.get_data(kstpkper=kstpkper, text=b'    INFILTRATION')
            for j, itm in enumerate(infil[0]):
                if itm[0] - 1 >= len(iuzno_dict_rev):
                    break
                iuz_row_col_tup = iuzno_dict_rev[itm[0] - 1]  # '-1' for converting mf 1-based output to 0-base
                prcp_amt = itm[2]
                precip_arr[i, iuz_row_col_tup[0], iuz_row_col_tup[1]] += prcp_amt
                
        print('Finished tallying precip')
        

        # Get UZET amounts on a cell-by-cell basis
        # For this model, UZET can only occur from layer 1 since the extinction depth doesn't extend beyond layer 1
        # However, writing the code for 3D processing in case others want to use it (including me!)
        for i, kstpkper in enumerate(ckstpkper):
            uzet = uzfobj.get_data(kstpkper=kstpkper, text=b'            UZET')
            for j, itm in enumerate(uzet[0]):
                if itm[0] - 1 < len(iuzno_dict_rev):
                    k = 0
                    iuz_row_col_tup = iuzno_dict_rev[itm[0] - 1]  # '-1' for converting mf 1-based output to 0-base
                else:  # Code enters here when processing a deeper layer
                    # To figure out the layer:
                    k = (itm[0] - 1) // len(iuzno_dict_rev)  # This k will be 0-based
                    # This will get the row/col (this code will not work if spatial extent of uzf objects varies among layers):
                    iuz_row_col_tup = iuzno_dict_rev[(itm[0] - 1) - (k * len(iuzno_dict_rev))]
                
                uzet_arr[i, k, iuz_row_col_tup[0], iuz_row_col_tup[1]] = itm[2]
                
        print('Finished tallying uzet')    
        
        uzet_arr = np.zeros((nper, nlay, nrow, ncol))
        for i, kstpkper in enumerate(ckstpkper):
            uzet = uzfobj.get_data(kstpkper=kstpkper, text=b'            UZET')
            for j, itm in enumerate(uzet[0]):
                if itm[0] - 1 < len(iuzno_dict_rev):
                    k = 0
                    iuz_row_col_tup = iuzno_dict_rev[itm[0] - 1]  # '-1' for converting mf 1-based output to 0-base
                else:  # Code enters here when processing a deeper layer
                    # To figure out the layer:
                    k = (itm[0] - 1) // len(iuzno_dict_rev)  # This k will be 0-based
                    # This will get the row/col (this code will not work if spatial extent of uzf objects varies among layers):
                    iuz_row_col_tup = iuzno_dict_rev[(itm[0] - 1) - (k * len(iuzno_dict_rev))]
                
                uzet_arr[i, k, iuz_row_col_tup[0], iuz_row_col_tup[1]] = itm[2]
        
        gwet_arr = np.zeros((nper, nlay, nrow, ncol))
        for i, kstpkper in enumerate(ckstpkper):
            gwet = modobj.get_data(kstpkper=kstpkper, text=b'        UZF-GWET')
            for j, itm in enumerate(gwet[0]):
                if itm[0] - 1 < len(iuzno_dict_rev):
                    k = 0
                    iuz_row_col_tup = iuzno_dict_rev[itm[1] - 1]  # '-1' for converting mf 1-based output to 0-base
                                                                  # use second entry in itm for getting iuzno 
                else:  # Code enters here when processing a deeper layer, but for now, since extinction depth doesn't extend 
                       # beyond layer 1, can ignore this
                    pass
                    # # To figure out the layer:
                    # k = (itm[1] - 1) // len(iuzno_dict_rev)  # This k will be 0-based
                    # # This will get the row/col (this code will not work if spatial extent of uzf objects varies among layers):
                    # iuz_row_col_tup = iuzno_dict_rev[(itm[1] - 1) - (k * len(iuzno_dict_rev))]
                
                gwet_arr[i, k, iuz_row_col_tup[0], iuz_row_col_tup[1]] = itm[2]
        
        
        #sw_irr_arr[185, 14:19, 50:55].sum(axis=0)
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(1, 1, 1)
        sw_del = sw_irr_arr[:,:,:].sum(axis=0)
        sw_del[sw_del==0.] = np.nan
        sw_del = abs(sw_del) / 400**2
        sw_del[sw_del==0.] = np.nan
        sw = ax.imshow(sw_del, cmap='terrain_r')
        ax.axvline(x=64)
        ax.axvline(x=70)
        ax.axhline(y=23)
        ax.axhline(y=29)
        
        # Create a Rectangle patch (but the following isn't working, so sticking with the lines)
        plt.gca().add_patch(patches.Rectangle((29, 64), 6, 6, linewidth=2, edgecolor='r', facecolor='none'))
        #rect = patches.Rectangle((29, 64), 6, 6, linewidth=2, edgecolor='r', facecolor='none')
        
        # Add the patch to the Axes
        #ax.add_patch(rect)
        
        cbar = plt.colorbar(sw, shrink=0.85)
        cbar.ax.set_title('Depth, in m', pad=20)
        title = "Annual Sum of Delivered Surface Water"
        
        letter = chr(ord("@") + idx + 15)
        fs.heading(letter=letter, heading=title)
        
        # save figure
        if config.plotSave:
            fpth = os.path.join(
                "..", "figures", "{}{}".format(sim_name + "spatSwDeliv", config.figure_ext)
            )
            fig.savefig(fpth)
        
        # -----------------------------------------------------------
        # Plot 16 - Plot delivered groundwater
        # -----------------------------------------------------------
        #sw_irr_arr[185, 14:19, 50:55].sum(axis=0)
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(1, 1, 1)
        gw_del = gw_irr_arr[:,:,:].sum(axis=0)
        gw_del[gw_del==0.] = np.nan
        gw_del = abs(gw_del) / 400**2
        gw_del[gw_del==0.] = np.nan
        
        gw = ax.imshow(gw_del, cmap='terrain_r')
        ax.axvline(x=64)
        ax.axvline(x=70)
        ax.axhline(y=23)
        ax.axhline(y=29)
        #ax.patches.Rectangle(xy, width, height
        
        cbar = plt.colorbar(gw, shrink=0.85)
        cbar.ax.set_title('Depth, in m', pad=20)
        title = "Annual Sum of Delivered Groundwater"
        
        letter = chr(ord("@") + idx + 16)
        fs.heading(letter=letter, heading=title)
        
        # save figure
        if config.plotSave:
            fpth = os.path.join(
                "..", "figures", "{}{}".format(sim_name + "spatGwDeliv", config.figure_ext)
            )
            fig.savefig(fpth)
        
        # -----------------------------------------------------------
        # Plot 17 - 3D barplot of total deliveries
        # -----------------------------------------------------------
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(111, projection = "3d")
        ax.set_yticks([0,4,8,12,16,20], minor=False)
        ax.yaxis.grid(True, which='major')
        ax.yaxis.set_ticklabels([64,65,66,67,68,69])
        ax.set_xticks([0.5,3.5,6.5,9.5,12.5,15.5], minor=False)
        ax.xaxis.grid(True, which='major')
        ax.xaxis.set_ticklabels([23,24,25,26,27,28])
        ax.view_init(azim=-10)
        
        ax.set_xlabel("Model Columns")
        ax.set_ylabel("Model Rows") 
        ax.set_zlabel("Total Depth, in m")
        ax.set_xlim3d(0,18.5)
        ax.set_ylim3d(0,23)
        ax.set_zlim3d(0,2)
        
        xpos = [2,5,8,11,14,17,2,5,8,11,14,17,2,5,8,11,14,17,2,5,8,11,14,17,2,5,8,11,14,17,2,5,8,11,14,17]
        ypos = [1,1,1,1,1,1,5,5,5,5,5,5,9,9,9,9,9,9,13,13,13,13,13,13,17,17,17,17,17,17,21,21,21,21,21,21]
        zpos = np.zeros(36)
        
        dx = np.ones(36)
        dy = np.ones(36)
        # the heights of the 3 bar sets
        dz = [[abs(number)/400**2 for number in precip_arr[:,23:29,64:70].sum(axis=0).flatten().tolist()],
              [abs(number)/400**2 for number in sw_irr_arr[:,23:29,64:70].sum(axis=0).flatten().tolist()], 
              [abs(number)/400**2 for number in gw_irr_arr[:,23:29,64:70].sum(axis=0).flatten().tolist()]] 
        
        _zpos = zpos   # the starting zpos for each bar
        colors = ['b', 'skyblue', 'mediumseagreen']
        for i in range(3):
            ax.bar3d(xpos, ypos, _zpos, dx, dy, dz[i], color=colors[i], shade=False, edgecolor = 'grey', linewidth=0.5)
            _zpos += dz[i]    # add the height of each bar to know where to start the next
        
        xtick_loc = np.arange(0.5, 18, 3)
        xticks_lab = [64,65,66,67,68,69]
        ytick_loc = np.arange(-0.5, 22,4)
        yticks_lab = [23,24,25,26,27,28]
        plt.xticks(xtick_loc, xticks_lab)
        plt.yticks(ytick_loc, yticks_lab)
        light = LightSource(azdeg=130, altdeg=33)
        plt.gca().invert_xaxis()
        
        precip_proxy = plt.Rectangle((0, 0), 1, 1, fc="b")
        sw_proxy = plt.Rectangle((0, 0), 1, 1, fc="skyblue")
        gw_proxy = plt.Rectangle((0, 0), 1, 1, fc="mediumseagreen")
        ax.legend([precip_proxy, sw_proxy, gw_proxy],['Precip','SW Delivery','GW Delivery'])
        ax.view_init(elev=33, azim=130)
        
        title = "Annual Sum of Irrigation Deliveries"

        letter = chr(ord("@") + idx + 17)
        #fs.heading(letter=letter, heading=title)
        
        # save figure
        if config.plotSave:
            fpth = os.path.join(
                "..", "figures", "{}{}".format(sim_name + "3dDeliv", config.figure_ext)
            )
            fig.savefig(fpth)
        
        # ------------------------------------------------------------
        # Plot 18 - Time series plot of concentrations for unsat cell
        # ------------------------------------------------------------
        # Post-process the MVT (.mvt.bud) mass fluxes
        mvt_output = 'gwt_modsim.mvt.bud'
        verbose=False
        if verbose:
            # The following, which has 'verbose=True' provides a clue as to what is in the CellBudget object
            mvt_Flx = bf.CellBudgetFile(os.path.join(mf6_out_path, mvt_output), verbose=True, precision='double')
        else:
            # Otherwise, to keep the output more succinct, use:
            mvt_Flx = bf.CellBudgetFile(os.path.join(mf6_out_path, mvt_output), verbose=False, precision='double')
        
        # This retrieves all of the MOVER fluxes:
        # ---------------------------------------
        mvt_all = mvt_Flx.get_data(text='        MVT-FLOW')
        
        ckstpkper = mvt_Flx.get_kstpkper()
        sw_irr_massFlx_arr = np.zeros((nper, nrow, ncol))  # surface water irrigation
        gw_irr_massFlx_arr = np.zeros((nper, nrow, ncol))  # groundwater irrigation
        precip_massFlx_arr = np.zeros((nper, nrow, ncol))  # rainfall
        mvt_uzf_rej_inf_massFlx_vals = []
        mvt_drn_gw_massFlx_disQ      = []
        
        # Get all mass flux associated with rejected infiltration:
        # ---------------------------------------------------------
        provider  = b'UZF-1           '
        receiver1 = b'SFR-1           '
        receiver2 = b'RES-1           '
        for i, kstpkper in enumerate(ckstpkper):
            mvt_rejinf_massFlx = 0  # Initialize
            # The following gets the actual indexes of where the conditions are satisfied, which is what the recarray needs
            mvt_idxs = np.where(np.all((np.array(mvt_Flx.recordarray['kper'] == i), 
                                        np.array(mvt_Flx.recordarray['paknam'] == provider),        # Provider
                                        np.logical_or(mvt_Flx.recordarray['paknam2'] == receiver1,  # Receiver
                                                      mvt_Flx.recordarray['paknam2'] == receiver2)), axis=0))
            # For each index (there will be two in this case because there are two receivers)
            for j in range(len(mvt_idxs[0])):
                mvt_uzf2sw = mvt_all[mvt_idxs[0][j]]
                # Loop through each row of the mvr_uzf2sw for tallying flow totals
                for k, itm in enumerate(mvt_uzf2sw):
                    mvt_rejinf_massFlx += itm[2]
            # Tally the result
            mvt_uzf_rej_inf_massFlx_vals.append(mvt_rejinf_massFlx)
            
        print('Finished tallying mass flux associated with rejected infiltration')
        
        # Get all groundwater discharge:
        # ---------------------------------------
        provider  = b'DRN-1           '
        receiver1 = b'SFR-1           '
        receiver2 = b'RES-1           '
        for i, kstpkper in enumerate(ckstpkper):
            mvt_gwdisQ_massFlx = 0  # Initialize
            # The following gets the actual indexes of where the conditions are satisfied, which is what the recarray needs
            mvt_idxs = np.where(np.all((np.array(mvt_Flx.recordarray['kper'] == i), 
                                        np.array(mvt_Flx.recordarray['paknam'] == provider),        # Provider
                                        np.logical_or(mvt_Flx.recordarray['paknam2'] == receiver1,  # Receiver
                                                      mvt_Flx.recordarray['paknam2'] == receiver2)), axis=0))
            # For each index (there will be two in this case because there are two receivers)
            for j in range(len(mvt_idxs[0])):
                mvt_drn2sw = mvt_all[mvt_idxs[0][j]]
                # Loop through each row of the mvr_uzf2sw for tallying flow totals
                for k, itm in enumerate(mvt_drn2sw):
                    mvt_gwdisQ_massFlx += itm[2]
            # Tally the result
            mvt_drn_gw_massFlx_disQ.append(mvt_gwdisQ_massFlx)
            
        print('Finished tallying mass flux associated with groundwater discharge')
        
        # Get all surface-water derived irrigation events (store as 3D array: [time, row, col]):
        # ---------------------------------------------------------------------------------------
        provider  = b'SFR-1           '
        receiver1 = b'UZF-1           '
        for i, kstpkper in enumerate(ckstpkper):
            # The following gets the actual indexes of where the conditions are satisfied, which is what the recarray needs
            mvt_idxs = np.where(np.all((np.array(mvt_Flx.recordarray['kper'] == i), 
                                        np.array(mvt_Flx.recordarray['paknam']  == provider),            # Provider
                                        np.array(mvt_Flx.recordarray['paknam2'] == receiver1)), axis=0)) # Receiver
        
            if len(mvt_idxs[0]) > 0:
                mvt_sfrirrig_massFlx = mvt_all[mvt_idxs[0][0]]
                # Loop through each row of the mvt_sfrirrig_massFlx recarray for tallying sw irrigation events on a cell-by-cell basis
                for k, itm in enumerate(mvt_sfrirrig_massFlx):
                    iuz_row_col_tup = iuzno_dict_rev[itm[1]]   # itm[1]: the receiver identifier, in this case iuzno, need to relate to row/col address
                    sw_irr_massFlx_arr[i, iuz_row_col_tup[0], iuz_row_col_tup[1]] += itm[2]
        
        print('Finished tallying mass flux associated with surface water irrigation')
        
        # Get all groundwater derived irrigation events (store as 3D array: [time, row, col]):
        # -------------------------------------------------------------------------------------
        provider  = b'WEL-1           '
        receiver1 = b'UZF-1           '
        for i, kstpkper in enumerate(ckstpkper):
            # The following gets the actual indexes of where the conditions are satisfied, which is what the recarray needs
            mvt_idxs = np.where(np.all((np.array(mvt_Flx.recordarray['kper'] == i), 
                                        np.array(mvt_Flx.recordarray['paknam']  == provider),            # Provider
                                        np.array(mvt_Flx.recordarray['paknam2'] == receiver1)), axis=0)) # Receiver
        
            if len(mvt_idxs[0]) > 0:
                mvt_welirrig_massFlx = mvt_all[mvt_idxs[0][0]]
                # Loop through each row of the mvr_sfrirrig recarray for tallying sw irrigation events on a cell-by-cell basis
                for k, itm in enumerate(mvt_welirrig_massFlx):
                    iuz_row_col_tup = iuzno_dict_rev[itm[1]]   # itm[1]: the receiver identifier, in this case iuzno, need to relate to row/col address
                    gw_irr_massFlx_arr[i, iuz_row_col_tup[0], iuz_row_col_tup[1]] += itm[2]
        
        print('Finished tallying mass flux associated with groundwater irrigation')
        
        # Get the precipitation amounts on a cell-by-cell basis
        # ------------------------------------------------------
        uzt_bud_file = 'gwt_modsim.uzt.bud'
        uztobj = bf.CellBudgetFile(os.path.join(mf6_out_path, uzt_bud_file), precision='double')
        for i, kstpkper in enumerate(ckstpkper):
            infil_massFlx = uzfobj.get_data(kstpkper=kstpkper, text=b'    INFILTRATION')
            for j, itm in enumerate(infil_massFlx[0]):
                if itm[0] - 1 >= len(iuzno_dict_rev):
                    break
                iuz_row_col_tup = iuzno_dict_rev[itm[0] - 1]  # '-1' for converting mf 1-based output to 0-base
                prcp_flx = itm[2]
                precip_massFlx_arr[i, iuz_row_col_tup[0], iuz_row_col_tup[1]] += prcp_flx
                
        print('Finished tallying mass flux associated with precip (Should always be zero, this serves as a check)')
        
        # Need to work up a plot of a UZT cell's concentration that receives 
        # water from mover. Track the concentration (or mass) of the irrigation 
        # water remembering that it can be sourced from either groundwater or 
        # surface-water.
        # Calculate saturation (dedicate this calc to its own cell so it doesn't have to be rerun, its slow)
        sat = np.zeros_like(hd)  # Dimensions are time, lay, row, col
        for i, kstpkper in enumerate(ckstpkper):
            stor = uzfobj.get_data(kstpkper=kstpkper, text=b'         STORAGE')
            for j, itm in enumerate(stor[0]):  # For each row (uzf obj) in the current stress period
                if itm[0] - 1 < len(iuzno_dict_rev):
                    k = 0
                    iuz_row_col_tup = iuzno_dict_rev[itm[0] - 1]  # '-1' for converting mf 1-based output to 0-base
                else:  # Code enters here when processing a deeper layer
                    # To figure out the layer:
                    k = (itm[0] - 1) // len(iuzno_dict_rev)  # This k will be 0-based
                    # This will get the row/col:
                    iuz_row_col_tup = iuzno_dict_rev[(itm[0] - 1) - (k * len(iuzno_dict_rev))]
                
                sat[i, k, iuz_row_col_tup[0], iuz_row_col_tup[1]] = itm[3] / uzMaxStor[k, iuz_row_col_tup[0], iuz_row_col_tup[1]]
        
        def make_patch_spines_invisible(ax):
            ax.set_frame_on(True)
            ax.patch.set_visible(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
        
        # Set row/col indices for plotting information
        rw, cl = 10, 54
        
        precBars = abs(precip_arr[:, rw, cl]) / 400**2
        swIrrBars = abs(sw_irr_arr[:, rw, cl]) / 400**2
        gwIrrBars = abs(gw_irr_arr[:, rw, cl]) / 400**2
        
        # The position of the bars on the x-axis
        r = np.arange(366).tolist()
        
        # Heights of precBars + swIrrBars
        top2bars = np.add(precBars, swIrrBars).tolist()
        barWidth = 2
        
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(1,1,1)
        
        w1 = plt.bar(r, precBars, color='b', edgecolor='white', width=barWidth, label='Precip.')
        w2 = plt.bar(r, swIrrBars, color='m', edgecolor='white', width=barWidth, bottom=precBars, label='SW Deliv.')
        w3 = plt.bar(r, gwIrrBars, color='c', edgecolor='white', width=barWidth, bottom=top2bars, label='GW Deliv.')
        ax.set_ylim(0, 0.15)
        ax.set_ylabel('Irrigation/precipitation Depth, in m')
        
        secax = ax.twinx()
        secax.spines["left"].set_position(("axes", -0.1))
        make_patch_spines_invisible(secax)
        secax.spines["left"].set_visible(True)
        secax.yaxis.set_label_position('left')
        secax.yaxis.set_ticks_position('left')
        totet = secax.plot((abs(gwet_arr[1:,0, rw, cl]) + abs(uzet_arr[1:, 0, rw, cl])) / 400**2 * 1000,'--', color='k', label='Total ET')
        secax.set_ylim(0,7)
        secax.set_ylabel('Total ET (UZET + GWET), in mm')
        
        ax2 = ax.twinx()
        ln1 = ax2.plot(uzt_conc[:,0, rw, cl], '.', color='r', label='Soil Moisture Concentration')
        ax2.set_ylim(0, 3500)
        ax2.set_ylabel('Concentration, in mg/L', rotation=270, labelpad=15)
        
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('axes', 1.1))
        ax3.set_ylim(1358,1365)
        ln2 = ax3.plot(hd[:, 1, rw, cl], '-', color='b', label='Head, row: ' + str(rw) + ' col: ' + str(cl))
        ax3.set_ylabel('Groundwater Altitude, in m', rotation=270, labelpad=23)
        
        ax4 = ax.twinx()
        ax4.spines['right'].set_position(('axes', 1.2))
        ln3 = ax4.plot(sat[:, 0, rw, cl], '-', color='g', label='Saturation, row: ' + str(rw) + ' col: ' + str(cl))
        ax4.set_ylim(0.0, 0.60)
        ax4.set_ylabel('Saturation', rotation=270, labelpad=30)
        
        # Put a legend to the right of the current axis
        lns = [w1] + [w2] + [w3] + totet + ln1 + ln2 + ln3
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper left')           
        
        title = 'Focused Look at Unsat Cell: row 10, col 54'
        letter = chr(ord("@") + idx + 18)
        fs.heading(letter=letter, heading=title)
        
        # save figure
        if config.plotSave:
            fpth = os.path.join(
                "..", "figures", "{}{}".format(sim_name + "_r10_c54", config.figure_ext)
            )
            fig.savefig(fpth)
        
        # ------------------------------------------------------------
        # Plot 19 - Unsat Zn conc for 3 cells through time
        # ------------------------------------------------------------
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(1, 1, 1)
        ln1 = ax.plot(uzt_conc[:,0,10,54], '.', color='r', label='Concentration in Layer 1 Cell')
        ln1 = ax.plot(uzt_conc[:,0,10,58], '.', color='b', label='Concentration in Layer 1 Cell')
        ln1 = ax.plot(uzt_conc[:,0,10,62], '.', color='g', label='Concentration in Layer 1 Cell')
        ax.set_xlabel('Stress Period')
        ax.set_ylabel('Unsaturated Zone Concentration (mg/L)')
        ax.set_xlim(0, 366)
        ax.set_ylim(0, 3200)
        title = "Unsaturated zone concentration for 3 cells through time"
        ax.yaxis.grid('on')
        
        letter = chr(ord("@") + idx + 19)
        fs.heading(letter=letter, heading=title)
        
        # save figure
        if config.plotSave:
            fpth = os.path.join(
                "..", "figures", "{}{}".format(sim_name + "_uzConc_3cells", config.figure_ext)
            )
            fig.savefig(fpth)
        
# ## Function that wraps all of the steps for each scenario
#
# 1. build_simulation
# 2. write_model
# 3. run_model
# 4. plot_results
#
def scenario(idx):
    # Remember that the same base model is used twice and is linked in series
    # Items that are different between the upstream and downstream repititions 
    # of the model include:
    #   o top elevation
    #   o bottom elevations
    #   o lake stage-volume-surface area table
    #   o starting concentrations
    
    # Two models connected in series, the main difference being the elevation adjustment factor
    elev_adj = [0, 151.04]
    
    # Build "upstream" version of model
    sim = build_simulation(example_name, elev_adj)
    
    if success:
        plot_results(sim, idx)


# nosetest - exclude block from this nosetest to the next nosetest
def test_01():
    scenario(0)

# nosetest end

if __name__ == "__main__":
    # ## Scenario 1
    # Two-dimensional transport in a uniform flow field
    scenario(0)
