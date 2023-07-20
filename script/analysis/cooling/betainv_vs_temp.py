######################################################################
#
# Simple analysis and plotting script to process problem output
# Plots mhdmodes, bondi and torus problem (2D and 3D)
#
######################################################################

# python3 ./script/analysis/cooling/betainv_vs_temp.py -p ./script/analysis/cooling/betainv_vs_temp.dat
#this assumes that both runs have equal grid files

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os,psutil,sys
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprocessing as mp

# Parallelize analysis by spawning several processes using multiprocessing's Pool object
def run_parallel(function,dlist,nthreads):
    pool = mp.Pool(nthreads)
    pool.map_async(function,dlist).get(720000)
    pool.close()
    pool.join()

# Initialize global variables
globalvars_keys = ['PROB','NDIMS','DUMPSDIR_heat','DUMPSDIR_heat_and_cool','PLOTSDIR']
globalvars = {}
grid ={}


# Function to generate poloidal (x,z) slice
# Argument must be variable, patch pole (to have x coordinate plotted correctly), averaging in phi option
def xz_slice(var, thmaxind, thminind, rmaxind, rminind, patch_pole=False, average=False):
	xz_var = np.zeros((2*rmaxind-rminind,thmaxind-thminind))
	if average:
		var = np.mean(var,axis=2)
		for i in range(rmaxind-rminind):
			xz_var[i,:] = var[rmaxind-rminind-1-i,:]
			xz_var[i+rmaxind-rminind,:] = var[i,:]
	else:
		angle = 0.; ind = 0
		for i in range(rmaxind-rminind):
			xz_var[i,:] = var[rmaxind-rminind-1-i,:,ind+grid['n3']//2]
			xz_var[i+rmaxind-rminind,:] = var[i,:,ind]
	if patch_pole:
		xz_var[:,0] = xz_var[:,-1] = 0
	return xz_var


# Function to generate poloidal (y,z) slice
# Argument must be variable, patch pole (to have y coordinate plotted correctly), averaging in phi option
# Not really called but can include a function call 
def yz_slice(var, thmaxind, thminind, rmaxind, rminind, patch_pole=False, average=False):
	yz_var = np.zeros((2*rmaxind-rminind,thmaxind-thminind))
	if average:
		var = np.mean(var,axis=2)
		for i in range(rmaxind-rminind):
			yz_var[i,:] = var[rmaxind-rminind-1-i,:]
			yz_var[i+rmaxind-rminind,:] = var[i,:]
	else:
		angle = np.pi/2; ind = np.argmin(abs(grid['phi'][0,0,:]-angle))
		for i in range(rmaxind-rminind):
			yz_var[i,:] = var[rmaxind-rminind-1-i,:,ind+grid['n3']//2]
			yz_var[i+rmaxind-rminind,:] = var[i,:,ind]
	if patch_pole:
		yz_var[:,0] = yz_var[:,-1] = 0
	return yz_var


# Function to generate toroidal (x,y) slice
# Argument must be variable, averaging in theta option
def xy_slice(var, average=False, patch_phi=False):
    if average:
        xy_var = np.mean(var,axis=1)
    else:
        xy_var = var[:,grid['n2']//2,:]
    #xy_var = np.vstack((xy_var.transpose(),xy_var.transpose()[0])).transpose()
    if patch_phi:
        xy_var[:,0] = xy_var[:,-1] = 0
    return xy_var

def analysis_torus2d(dumpval, cmap='jet', vmin=-13, vmax=-9, domain = [-50,0,-50,50], bh=True, shading='gouraud'):
    plt.clf()
    print("Analyzing {0:04d} dump".format(dumpval))
    thmid = np.pi/2.
    thmidind = np.argmin(abs(grid['th'][-1,:,0]-thmid))
    thminind = thmidind - 2
    thmaxind = thmidind + 2
    rmin = grid['rEH']
    rmax = 6
    rminind = np.argmin(abs(grid['r'][:,0,0]-rmin))
    rmaxind = np.argmin(abs(grid['r'][:,0,0]-rmax))

    dfile = h5py.File(os.path.join(globalvars['DUMPSDIR_heat'],'dump_0000{0:04d}.h5'.format(dumpval)),'r')
    rho = dfile['prims'][rminind:rmaxind,thminind:thmaxind,:][Ellipsis,0]
    uu = np.array(dfile['prims'][rminind:rmaxind,thminind:thmaxind,:][Ellipsis,1])
    u = np.array(dfile['prims'][rminind:rmaxind,thminind:thmaxind,:][Ellipsis,2:5])
    B = np.array(dfile['prims'][rminind:rmaxind,thminind:thmaxind,:][Ellipsis,5:8])
    gam = np.array(dfile['header/gam'][()])
    game = np.array(dfile['header/gam_e'][()])
    KEL0 = np.array(dfile['prims'][rminind:rmaxind,thminind:thmaxind,:][Ellipsis,9])
    KEL1 = np.array(dfile['prims'][rminind:rmaxind,thminind:thmaxind,:][Ellipsis,10])
    KEL2 = np.array(dfile['prims'][rminind:rmaxind,thminind:thmaxind,:][Ellipsis,11])
    KEL3 = np.array(dfile['prims'][rminind:rmaxind,thminind:thmaxind,:][Ellipsis,12])
    uel0 = rho**game*KEL0/(game-1.)
    uel1 = rho**game*KEL1/(game-1.)
    uel2 = rho**game*KEL2/(game-1.)
    uel3 = rho**game*KEL3/(game-1.)
    ME = 9.1093826e-28 #Electron mass
    MP = 1.67262171e-24 #Proton mass
    CL = 2.99792458e10 # Speed of light
    GNEWT = 6.6742e-8 # Gravitational constant
    MSUN = 1.989e33 # grams per solar mass
    Kbol = 1.380649e-16 # boltzmann constant
    M_bh = 6.5e9 #mass of M87* in solar masses
    M_unit = 1.e28 #arbitrary
    M_bh_cgs = M_bh * MSUN
    L_unit = GNEWT*M_bh_cgs/pow(CL, 2.)
    RHO_unit = M_unit*pow(L_unit, -3.)
    Ne_unit = RHO_unit/(MP + ME)
    U_unit = RHO_unit*CL*CL
    theta0 = np.log10((game-1.)*uel0*U_unit/(rho*Ne_unit*Kbol))
    theta1 = np.log10((game-1.)*uel1*U_unit/(rho*Ne_unit*Kbol))
    theta2 = np.log10((game-1.)*uel2*U_unit/(rho*Ne_unit*Kbol))
    theta3 = np.log10((game-1.)*uel3*U_unit/(rho*Ne_unit*Kbol))
    t = dfile['t'][()]
    dfile.close()
    t = "{:.3f}".format(t)
    pg = (gam-1)*uu
    gti = grid['gcon'][rminind:rmaxind,thminind:thmaxind,:][Ellipsis,0,1:4]
    gij = grid['gcov'][rminind:rmaxind,thminind:thmaxind,:][Ellipsis,1:4,1:4]
    beta_i = np.einsum('ijks,ijk->ijks',gti,grid['lapse'][rminind:rmaxind,thminind:thmaxind,:]**2)
    qsq = np.einsum('ijky,ijky->ijk',np.einsum('ijkxy,ijkx->ijky',gij,u),u)
    gamma = np.sqrt(1+qsq)
    ui = u-np.einsum('ijks,ijk->ijks',beta_i,gamma/grid['lapse'][rminind:rmaxind,thminind:thmaxind,:])
    ut = gamma/grid['lapse'][rminind:rmaxind,thminind:thmaxind,:]
    ucon = np.append(ut[Ellipsis,None],ui,axis=3)
    ucov = np.einsum('ijkmn,ijkn->ijkm',grid['gcov'][rminind:rmaxind,thminind:thmaxind,:],ucon)
    bt = np.einsum('ijkm,ijkm->ijk',np.einsum('ijksm,ijks->ijkm',grid['gcov'][rminind:rmaxind,thminind:thmaxind,:][Ellipsis,1:4,:],B),ucon)
    bi = (B+np.einsum('ijks,ijk->ijks',ui,bt))/ut[Ellipsis,None]
    bcon = np.append(bt[Ellipsis,None],bi,axis=3)
    bcov = np.einsum('ijkmn,ijkn->ijkm',grid['gcov'][rminind:rmaxind,thminind:thmaxind,:],bcon)
    bsq = np.einsum('ijkm,ijkm->ijk',bcon,bcov)
    beta = pg/(0.5*bsq)
    logbeta = np.log10(beta)

    theta0p = xz_slice(theta0, thmaxind, thminind, rmaxind, rminind)
    theta1p = xz_slice(theta1, thmaxind, thminind, rmaxind, rminind)
    theta2p = xz_slice(theta2, thmaxind, thminind, rmaxind, rminind)
    theta3p = xz_slice(theta3, thmaxind, thminind, rmaxind, rminind)
    betap = xz_slice(logbeta, thmaxind, thminind, rmaxind, rminind)

    #**********************************************************************************************************************

    dfile2 = h5py.File(os.path.join(globalvars['DUMPSDIR_heat_and_cool'],'dump_0000{0:04d}.h5'.format(dumpval)),'r')
    rho2 = dfile2['prims'][rminind:rmaxind,thminind:thmaxind,:][Ellipsis,0]
    uu2 = np.array(dfile2['prims'][rminind:rmaxind,thminind:thmaxind,:][Ellipsis,1])
    u2 = np.array(dfile2['prims'][rminind:rmaxind,thminind:thmaxind,:][Ellipsis,2:5])
    B2 = np.array(dfile2['prims'][rminind:rmaxind,thminind:thmaxind,:][Ellipsis,5:8])
    gam2 = np.array(dfile2['header/gam'][()])
    game2 = np.array(dfile2['header/gam_e'][()])
    KEL02 = np.array(dfile2['prims'][rminind:rmaxind,thminind:thmaxind,:][Ellipsis,9])
    KEL12 = np.array(dfile2['prims'][rminind:rmaxind,thminind:thmaxind,:][Ellipsis,10])
    KEL22 = np.array(dfile2['prims'][rminind:rmaxind,thminind:thmaxind,:][Ellipsis,11])
    KEL32 = np.array(dfile2['prims'][rminind:rmaxind,thminind:thmaxind,:][Ellipsis,12])
    uel02 = rho2**game2*KEL02/(game2-1.)
    uel12 = rho2**game2*KEL12/(game2-1.)
    uel22 = rho2**game2*KEL22/(game2-1.)
    uel32 = rho2**game2*KEL32/(game2-1.)
    theta02 = np.log10((game2-1.)*uel02*U_unit/(rho2*Ne_unit*Kbol))
    theta12 = np.log10((game2-1.)*uel12*U_unit/(rho2*Ne_unit*Kbol))
    theta22 = np.log10((game2-1.)*uel22*U_unit/(rho2*Ne_unit*Kbol))
    theta32 = np.log10((game2-1.)*uel32*U_unit/(rho2*Ne_unit*Kbol))
    t2 = dfile2['t'][()]
    dfile2.close()
    t2 = "{:.3f}".format(t2)
    pg2 = (gam2-1)*uu2
    gti2 = grid['gcon'][rminind:rmaxind,thminind:thmaxind,:][Ellipsis,0,1:4]
    gij2 = grid['gcov'][rminind:rmaxind,thminind:thmaxind,:][Ellipsis,1:4,1:4]
    beta_i2 = np.einsum('ijks,ijk->ijks',gti2,grid['lapse'][rminind:rmaxind,thminind:thmaxind,:]**2)
    qsq2 = np.einsum('ijky,ijky->ijk',np.einsum('ijkxy,ijkx->ijky',gij2,u2),u2)
    gamma2 = np.sqrt(1+qsq2)
    ui2 = u2-np.einsum('ijks,ijk->ijks',beta_i2,gamma2/grid['lapse'][rminind:rmaxind,thminind:thmaxind,:])
    ut2 = gamma2/grid['lapse'][rminind:rmaxind,thminind:thmaxind,:]
    ucon2 = np.append(ut2[Ellipsis,None],ui2,axis=3)
    ucov2 = np.einsum('ijkmn,ijkn->ijkm',grid['gcov'][rminind:rmaxind,thminind:thmaxind,:],ucon2)
    bt2 = np.einsum('ijkm,ijkm->ijk',np.einsum('ijksm,ijks->ijkm',grid['gcov'][rminind:rmaxind,thminind:thmaxind,:][Ellipsis,1:4,:],B2),ucon2)
    bi2 = (B2+np.einsum('ijks,ijk->ijks',ui2,bt2))/ut2[Ellipsis,None]
    bcon2 = np.append(bt2[Ellipsis,None],bi2,axis=3)
    bcov2 = np.einsum('ijkmn,ijkn->ijkm',grid['gcov'][rminind:rmaxind,thminind:thmaxind,:],bcon2)
    bsq2 = np.einsum('ijkm,ijkm->ijk',bcon2,bcov2)
    beta2 = pg2/(0.5*bsq2)
    logbeta2 = np.log10(beta2)

    theta0p2 = xz_slice(theta02, thmaxind, thminind, rmaxind, rminind)
    theta1p2 = xz_slice(theta12, thmaxind, thminind, rmaxind, rminind)
    theta2p2 = xz_slice(theta22, thmaxind, thminind, rmaxind, rminind)
    theta3p2 = xz_slice(theta32, thmaxind, thminind, rmaxind, rminind)
    betap2 = xz_slice(logbeta2, thmaxind, thminind, rmaxind, rminind)

    fig = plt.figure(figsize=(16,9))
    heights = [0.25,0.1,5]
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=heights, figure=fig)

    ax0 = fig.add_subplot(gs[0,:])
    ax0.annotate('using 5 zones in the theta direction near the midplane and EH < r < 6M',xy=(0.5,0.5),xycoords='axes fraction',va='center',ha='center',fontsize='x-large')
    ax0.axis("off")
    ax3 = fig.add_subplot(gs[1,:])
    ax3.annotate('t= '+str(t),xy=(0.5,0.5),xycoords='axes fraction',va='center',ha='center',fontsize='x-large')
    ax3.axis("off")

    #plotting heating:
    ax1 = fig.add_subplot(gs[2,0])
    ax1.scatter(betap, theta0p, color = 'r', marker = '.', label = 'Kawazura')
    ax1.scatter(betap, theta1p, color = 'b', marker = '.', label = 'Werner')
    ax1.scatter(betap, theta2p, color = 'g', marker = '.', label = 'Rowan')
    ax1.scatter(betap, theta3p, color = 'y', marker = '.', label = 'Sharma')
    ax1.set_xlim(-5, 10)
    ax1.set_ylim(2, 13)
    ax1.set_xlabel('log(beta)')
    ax1.set_ylabel('log(temperature in K)')
    ax1.set_title('Only With HEATING Enabled',fontsize='large')
    ax1.legend(loc='lower right')

    #plotting Heating and Cooling:
    ax2 = fig.add_subplot(gs[2,1])
    ax2.scatter(betap2, theta0p2, color = 'r', marker = '.', label = 'Kawazura')
    ax2.scatter(betap2, theta1p2, color = 'b', marker = '.', label = 'Werner')
    ax2.scatter(betap2, theta2p2, color = 'g', marker = '.', label = 'Rowan')
    ax2.scatter(betap2, theta3p2, color = 'y', marker = '.', label = 'Sharma')
    ax2.set_xlim(-5, 10)
    ax2.set_ylim(2, 13)
    ax2.set_xlabel('log(beta)')
    ax2.set_ylabel('log(temperature in K)')
    ax2.set_title('With Both HEATING and COOLING Enabled',fontsize='large')
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(globalvars['PLOTSDIR'],'{}_theta_plot_{:04d}.png'.format(globalvars['PROB'],dumpval)))
    plt.close()

# main(): Reads param file, writes grid dict and calls analysis function
if __name__=="__main__":
    if len(sys.argv) > 1 and sys.argv[1]=='-p':
        fparams_name = sys.argv[2]
    else:
        sys.exit('No param file provided')

    # Reading the param file
    with open(fparams_name,'r') as fparams:
        lines = fparams.readlines()
        for line in lines:
            if line[0]=='#' or line.isspace(): pass
            elif line.split()[0] in globalvars_keys: globalvars[line.split()[0]]=line.split()[-1]

    # Creating the output directory if it doesn't exist
    if not os.path.exists(globalvars['PLOTSDIR']):
        os.makedirs(globalvars['PLOTSDIR'])

    # Calculating total dump files
    dstart = int(sorted(os.listdir(globalvars['DUMPSDIR_heat']))[0][-7:-3])
    dend = int(sorted(list(filter(lambda dump: 'dump' in dump,os.listdir(globalvars['DUMPSDIR_heat']))))[-1][-7:-3])
    dlist = range(dstart,dend+1)
    Ndumps = dend-dstart+1

    # Setting grid dict
    gfile = h5py.File(os.path.join(globalvars['DUMPSDIR_heat'],'grid.h5'),'r')
    dfile = h5py.File(os.path.join(globalvars['DUMPSDIR_heat'],'dump_0000{0:04d}.h5'.format(dstart)),'r')
    grid['n1'] = dfile['/header/n1'][()]; grid['n2'] = dfile['/header/n2'][()]; grid['n3'] = dfile['/header/n3'][()]
    grid['dx1'] = dfile['/header/geom/dx1'][()]; grid['dx2'] = dfile['/header/geom/dx2'][()]; grid['dx3'] = dfile['/header/geom/dx3'][()]
    grid['startx1'] = dfile['header/geom/startx1'][()]; grid['startx2'] = dfile['header/geom/startx2'][()]; grid['startx3'] = dfile['header/geom/startx3'][()]
    grid['metric'] = dfile['header/metric'][()].decode('UTF-8')
    if grid['metric']=='MKS' or grid['metric']=='MMKS':
        try:
            grid['a'] = dfile['header/geom/mks/a'][()]
        except KeyError:
            grid['a'] = dfile['header/geom/mmks/a'][()]
        try:
            grid['rEH'] = dfile['header/geom/mks/Reh'][()]
        except KeyError:
            pass
        try:
            grid['rEH'] = dfile['header/geom/mks/r_eh'][()]
        except KeyError:
            pass
        try:
            grid['rEH'] = dfile['header/geom/mmks/Reh'][()]
        except KeyError:
            pass
        try:
            grid['rEH'] = dfile['header/geom/mmks/r_eh'][()]
        except KeyError:
            pass
        try:
            grid['hslope'] = dfile['header/geom/mks/hslope'][()]
        except KeyError:
            grid['hslope'] = dfile['header/geom/mmks/hslope'][()]
    if grid['metric']=='MMKS':
        grid['mks_smooth'] = dfile['header/geom/mmks/mks_smooth'][()]
        grid['poly_alpha'] = dfile['header/geom/mmks/poly_alpha'][()]
        grid['poly_xt'] = dfile['header/geom/mmks/poly_xt'][()]
        grid['D'] = (np.pi*grid['poly_xt']**grid['poly_alpha'])/(2*grid['poly_xt']**grid['poly_alpha']+(2/(1+grid['poly_alpha'])))
    grid['x1'] = gfile['X1'][()]; grid['x2'] = gfile['X2'][()]; grid['x3'] = gfile['X3'][()]
    grid['r'] = gfile['r'][()]; grid['th'] = gfile['th'][()]; grid['phi'] = gfile['phi'][()]
    grid['x'] = gfile['X'][()]; grid['y'] = gfile['Y'][()]; grid['z'] = gfile['Z'][()]
    grid['gcov'] = gfile['gcov'][()]; grid['gcon'] = gfile['gcon'][()]
    grid['gdet'] = gfile['gdet'][()]
    grid['lapse'] = gfile['lapse'][()]
    dfile.close()
    gfile.close()
    ncores = psutil.cpu_count(logical=True)
    pad = 0.25
    nthreads = int(ncores*pad); print("Number of threads: {0:03d}".format(nthreads))

    # Calling analysis function for torus2d
    run_parallel(analysis_torus2d,dlist,nthreads)
