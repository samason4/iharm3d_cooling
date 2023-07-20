######################################################################
#
# Simple analysis and plotting script to process problem output
# Plots mhdmodes, bondi and torus problem (2D and 3D)
#
######################################################################

# python3 ./script/analysis/cooling/temp_plots.py -p ./script/analysis/cooling/temp_params.dat
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
def xz_slice(var, patch_pole=False, average=False):
	xz_var = np.zeros((2*grid['n1'],grid['n2']))
	if average:
		var = np.mean(var,axis=2)
		for i in range(grid['n1']):
			xz_var[i,:] = var[grid['n1']-1-i,:]
			xz_var[i+grid['n1'],:] = var[i,:]
	else:
		angle = 0.; ind = 0
		for i in range(grid['n1']):
			xz_var[i,:] = var[grid['n1']-1-i,:,ind+grid['n3']//2]
			xz_var[i+grid['n1'],:] = var[i,:,ind]
	if patch_pole:
		xz_var[:,0] = xz_var[:,-1] = 0
	return xz_var


# Function to generate poloidal (y,z) slice
# Argument must be variable, patch pole (to have y coordinate plotted correctly), averaging in phi option
# Not really called but can include a function call 
def yz_slice(var, patch_pole=False, average=False):
	yz_var = np.zeros((2*grid['n1'],grid['n2']))
	if average:
		var = np.mean(var,axis=2)
		for i in range(grid['n1']):
			yz_var[i,:] = var[grid['n1']-1-i,:]
			yz_var[i+grid['n1'],:] = var[i,:]
	else:
		angle = np.pi/2; ind = np.argmin(abs(grid['phi'][0,0,:]-angle))
		for i in range(grid['n1']):
			yz_var[i,:] = var[grid['n1']-1-i,:,ind+grid['n3']//2]
			yz_var[i+grid['n1'],:] = var[i,:,ind]
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

def analysis_torus2d(dumpval, cmap='jet', vmin=5, vmax=12, domain = [-50,0,-50,50], bh=True, shading='gouraud'):
    plt.clf()
    print("Analyzing {0:04d} dump".format(dumpval))
    dfile = h5py.File(os.path.join(globalvars['DUMPSDIR_heat'],'dump_0000{0:04d}.h5'.format(dumpval)),'r')
    rho = dfile['prims'][()][Ellipsis,0]
    game = np.array(dfile['header/gam_e'][()])
    KEL0 = np.array(dfile['prims'][()][Ellipsis,9])
    KEL1 = np.array(dfile['prims'][()][Ellipsis,10])
    KEL2 = np.array(dfile['prims'][()][Ellipsis,11])
    KEL3 = np.array(dfile['prims'][()][Ellipsis,12])
    uel0 = rho**game*KEL0/(game-1.)
    uel1 = rho**game*KEL1/(game-1.)
    uel2 = rho**game*KEL2/(game-1.)
    uel3 = rho**game*KEL3/(game-1.)
    #I would define these at Tel (temperature of the electrons), but at first I tried plotting the normalized temperature, theta,
    #which ended up being way too small, and I didn't want to go through and change theta to Tel, just for the sake of time
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

    xp = xz_slice(grid['x'], patch_pole=True)
    zp = xz_slice(grid['z'])
    theta0p = xz_slice(theta0)
    theta1p = xz_slice(theta1)
    theta2p = xz_slice(theta2)
    theta3p = xz_slice(theta3)

#**********************************************************************************************************************

    dfile2 = h5py.File(os.path.join(globalvars['DUMPSDIR_heat_and_cool'],'dump_0000{0:04d}.h5'.format(dumpval)),'r')
    rho2 = dfile2['prims'][()][Ellipsis,0]
    game2 = np.array(dfile2['header/gam_e'][()])
    KEL02 = np.array(dfile2['prims'][()][Ellipsis,9])
    KEL12 = np.array(dfile2['prims'][()][Ellipsis,10])
    KEL22 = np.array(dfile2['prims'][()][Ellipsis,11])
    KEL32 = np.array(dfile2['prims'][()][Ellipsis,12])
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

    xp2 = xz_slice(grid['x'], patch_pole=True)
    zp2 = xz_slice(grid['z'])
    theta0p2 = xz_slice(theta02)
    theta1p2 = xz_slice(theta12)
    theta2p2 = xz_slice(theta22)
    theta3p2 = xz_slice(theta32)

    #just to check that there is a difference:
    max0 = np.amax(theta0.reshape(16384) - theta02.reshape(16384))
    max1 = np.amax(theta1.reshape(16384) - theta12.reshape(16384))
    max2 = np.amax(theta2.reshape(16384) - theta22.reshape(16384))
    max3 = np.amax(theta3.reshape(16384) - theta32.reshape(16384))
    max0 = "{:.16f}".format(max0)
    max1 = "{:.16f}".format(max1)
    max2 = "{:.16f}".format(max2)
    max3 = "{:.16f}".format(max3)

    fig = plt.figure(figsize=(16,9))
    heights = [1,5,5]
    gs = gridspec.GridSpec(nrows=3, ncols=4, height_ratios=heights, figure=fig)

    ax0 = fig.add_subplot(gs[0,:])
    ax0.annotate('t= '+str(t)+'; max diff in: temp0: '+str(max0)+', temp1: '+str(max1)+', temp2: '+str(max2)+', temp3: '+str(max3),xy=(0.5,0.5),xycoords='axes fraction',va='center',ha='center',fontsize='x-large')
    ax0.axis("off")

    #plotting theta0p:
    ax1 = fig.add_subplot(gs[1,0])
    theta0polplot = ax1.pcolormesh(xp, zp, theta0p, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax1.set_xlabel('$x (GM/c^2)$')
    ax1.set_ylabel('$z (GM/c^2)$')
    ax1.set_xlim(domain[:2])
    ax1.set_ylim(domain[2:])
    ax1.set_title('log(temp in K) Kawazura with Heating',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax1.add_artist(circle)
    ax1.set_aspect('equal')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta0polplot, cax=cax)

    #plotting theta1p:
    ax2 = fig.add_subplot(gs[1,1])
    theta1polplot = ax2.pcolormesh(xp, zp, theta1p, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax2.set_xlabel('$x (GM/c^2)$')
    ax2.set_ylabel('$z (GM/c^2)$')
    ax2.set_xlim(domain[:2])
    ax2.set_ylim(domain[2:])
    ax2.set_title('log(temp in K) Werner with Heating',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax2.add_artist(circle)
    ax2.set_aspect('equal')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta1polplot, cax=cax)

    #plotting theta2p:
    ax3 = fig.add_subplot(gs[1,2])
    theta2polplot = ax3.pcolormesh(xp, zp, theta2p, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax3.set_xlabel('$x (GM/c^2)$')
    ax3.set_ylabel('$z (GM/c^2)$')
    ax3.set_xlim(domain[:2])
    ax3.set_ylim(domain[2:])
    ax3.set_title('log(temp in K) Rowan with Heating',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax3.add_artist(circle)
    ax3.set_aspect('equal')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta2polplot, cax=cax)

    #plotting theta3p:
    ax4 = fig.add_subplot(gs[1,3])
    theta3polplot = ax4.pcolormesh(xp, zp, theta3p, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax4.set_xlabel('$x (GM/c^2)$')
    ax4.set_ylabel('$z (GM/c^2)$')
    ax4.set_xlim(domain[:2])
    ax4.set_ylim(domain[2:])
    ax4.set_title('log(temp in K) Sharma with Heating',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax4.add_artist(circle)
    ax4.set_aspect('equal')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta3polplot, cax=cax)

    #plotting theta0p2:
    ax5 = fig.add_subplot(gs[2,0])
    theta02polplot = ax5.pcolormesh(xp, zp, theta0p2, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax5.set_label('$x (GM/c^2)$')
    ax5.set_ylabel('$z (GM/c^2)$')
    ax5.set_xlim(domain[:2])
    ax5.set_ylim(domain[2:])
    ax5.set_title('log(temp in K) Kawazura with Heating & Cooling',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax5.add_artist(circle)
    ax5.set_aspect('equal')
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta02polplot, cax=cax)

    #plotting theta1p2:
    ax6 = fig.add_subplot(gs[2,1])
    theta12polplot = ax6.pcolormesh(xp, zp, theta1p2, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax6.set_label('$x (GM/c^2)$')
    ax6.set_ylabel('$z (GM/c^2)$')
    ax6.set_xlim(domain[:2])
    ax6.set_ylim(domain[2:])
    ax6.set_title('log(temp in K) Werner with Heating & Cooling',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax6.add_artist(circle)
    ax6.set_aspect('equal')
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta12polplot, cax=cax)

    #plotting theta2p2:
    ax7 = fig.add_subplot(gs[2,2])
    theta22polplot = ax7.pcolormesh(xp, zp, theta2p2, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax7.set_label('$x (GM/c^2)$')
    ax7.set_ylabel('$z (GM/c^2)$')
    ax7.set_xlim(domain[:2])
    ax7.set_ylim(domain[2:])
    ax7.set_title('log(temp in K) Rowan with Heating & Cooling',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax7.add_artist(circle)
    ax7.set_aspect('equal')
    divider = make_axes_locatable(ax7)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta22polplot, cax=cax)

    #plotting theta3p2:
    ax8 = fig.add_subplot(gs[2,3])
    theta32polplot = ax8.pcolormesh(xp, zp, theta3p2, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax8.set_label('$x (GM/c^2)$')
    ax8.set_ylabel('$z (GM/c^2)$')
    ax8.set_xlim(domain[:2])
    ax8.set_ylim(domain[2:])
    ax8.set_title('log(temp in K) Sharma with Heating & Cooling',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax8.add_artist(circle)
    ax8.set_aspect('equal')
    divider = make_axes_locatable(ax8)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(theta32polplot, cax=cax)

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
