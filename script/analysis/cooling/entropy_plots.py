######################################################################
#
# Simple analysis and plotting script to process problem output
# Plots mhdmodes, bondi and torus problem (2D and 3D)
#
######################################################################

# python3 ./script/analysis/cooling/entropy_plots.py -p ./script/analysis/cooling/params_analysis.dat

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
globalvars_keys = ['PROB','NDIMS','DUMPSDIR','PLOTSDIR']
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

# Function to overlay field lines
# Argument must be axes object, B1, B2 and 'nlines' -> a parameter to account for density of field lines
def plotting_bfield_lines(ax,B1,B2,nlines=20):
    xp = xz_slice(grid['x'], patch_pole=True)
    zp = xz_slice(grid['z'])
    B1_phi_avg = B1.mean(axis=-1) 
    B2_phi_avg = B2.mean(axis=-1)
    AJ_phi = np.zeros([2*grid['n1'],grid['n2']]) 
    for j in range(grid['n2']):
        for i in range(grid['n1']):
            AJ_phi[grid['n1']-1-i,j] = AJ_phi[i+grid['n1'],j] = (np.trapz(grid['gdet'][:i,j,0]*B2_phi_avg[:i,j],dx=grid['dx1']) - np.trapz(grid['gdet'][i,:j,0]*B1_phi_avg[i,:j],dx=grid['dx2']))
    AJ_phi -=AJ_phi.min()
    levels = np.linspace(0,AJ_phi.max(),nlines*2)
    ax.contour(xp, zp, AJ_phi, levels=levels, colors='k')

# The actual function that computes and plots diagnostics for PROB=mhdmodes
def analysis_mhdmodes(dumpval, cmap='jet', vmin=-4e-5, vmax=4e-5, domain = [0,1,0,1], shading='gouraud'):
    plt.clf()
    print("Analyzing {0:04d} dump".format(dumpval))
    dfile = h5py.File(os.path.join(globalvars['DUMPSDIR'],'dump_0000{0:04d}.h5'.format(dumpval)),'r')
    rho = dfile['prims'][()][Ellipsis,0]
    t = dfile['t'][()]
    dfile.close()
    t = "{:.3f}".format(t)
    logrho=np.log10(rho)

    fig = plt.figure(figsize=(16,9))
    heights = [1,5]
    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=heights, figure=fig)

    ax0 = fig.add_subplot(gs[0,:])
    ax0.annotate('t= '+str(t),xy=(0.5,0.5),xycoords='axes fraction',va='center',ha='center',fontsize='x-large')
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[1,0])
    rhoxzplot = ax1.pcolormesh(grid['x'][:,0,:], grid['z'][:,0,:], logrho[:,0,:], cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$z$')
    ax1.set_xlim(domain[:2])
    ax1.set_ylim(domain[2:])
    ax1.set_title('Log($\\rho$)',fontsize='large')
    ax1.set_aspect('equal')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(rhoxzplot, cax=cax)

    ax2 = fig.add_subplot(gs[1,1])
    rhoxyplot = ax2.pcolormesh(grid['x'][:,:,0], grid['y'][:,:,0], logrho[:,:,0], cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')
    ax2.set_xlim(domain[:2])
    ax2.set_ylim(domain[2:])
    ax2.set_title('Log($\\rho$)',fontsize='large')
    ax2.set_aspect('equal')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(rhoxyplot, cax=cax)

    plt.tight_layout()
    plt.savefig(os.path.join(globalvars['PLOTSDIR'],'{}_basic_plot_{:04d}.png'.format(globalvars['PROB'],dumpval)))
    plt.close()


# The actual function that computes and plots diagnostics for PROB=bondi
def analysis_bondi(dumpval, cmap='jet', vmin=-3, vmax=-1, domain = [-20,0,-20,20], bh=True, shading='gouraud'):
    plt.clf()
    print("Analyzing {0:04d} dump".format(dumpval))
    dfile = h5py.File(os.path.join(globalvars['DUMPSDIR'],'dump_0000{0:04d}.h5'.format(dumpval)),'r')
    rho = dfile['prims'][()][Ellipsis,0]
    t = dfile['t'][()]
    dfile.close()
    t = "{:.3f}".format(t)
    logrho=np.log10(rho)

    xp = xz_slice(grid['x'], patch_pole=True)
    zp = xz_slice(grid['z'])
    rhop = xz_slice(logrho)

    fig = plt.figure(figsize=(16,9))
    heights = [1,5]
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=heights, figure=fig)

    ax0 = fig.add_subplot(gs[0,0])
    ax0.annotate('t= '+str(t),xy=(0.5,0.5),xycoords='axes fraction',va='center',ha='center',fontsize='x-large')
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[1,0])
    rhopolplot = ax1.pcolormesh(xp, zp, rhop, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax1.set_xlabel('$x (GM/c^2)$')
    ax1.set_ylabel('$z (GM/c^2)$')
    ax1.set_xlim(domain[:2])
    ax1.set_ylim(domain[2:])
    ax1.set_title('Log($\\rho$)',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax1.add_artist(circle)
    ax1.set_aspect('equal')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(rhopolplot, cax=cax)

    plt.tight_layout()
    plt.savefig(os.path.join(globalvars['PLOTSDIR'],'{}_basic_plot_{:04d}.png'.format(globalvars['PROB'],dumpval)))
    plt.close()

# The actual function that computes and plots diagnostics for PROB=torus and NDIMS=2
def analysis_torus2d(dumpval, cmap='jet', vmin=-5, vmax=0, domain = [-50,0,-50,50], bh=True, shading='gouraud'):
    plt.clf()
    print("Analyzing {0:04d} dump".format(dumpval))
    dfile = h5py.File(os.path.join(globalvars['DUMPSDIR'],'dump_0000{0:04d}.h5'.format(dumpval)),'r')
    KEL0 = np.array(dfile['prims'][()][Ellipsis,9])
    KEL1 = np.array(dfile['prims'][()][Ellipsis,10])
    KEL2 = np.array(dfile['prims'][()][Ellipsis,11])
    KEL3 = np.array(dfile['prims'][()][Ellipsis,12])
    KTOT = np.array(dfile['prims'][()][Ellipsis,8])
    rho = np.array(dfile['prims'][()][Ellipsis,0])
    t = dfile['t'][()]
    tptemin = np.array(dfile['header/tptemin'][()])
    tptemax = np.array(dfile['header/tptemax'][()])
    gam = np.array(dfile['header/gam'][()])
    game = np.array(dfile['header/gam_e'][()])
    gamp = np.array(dfile['header/gam_p'][()])
    kelmax = KTOT*rho**(gam-game)/(tptemin*(gam-1)/(gamp-1)+(gam-1)/(game-1))
    kelmin = KTOT*rho**(gam-game)/(tptemax*(gam-1)/(gamp-1)+(gam-1)/(game-1))

    dfile.close()
    t = "{:.3f}".format(t)

    xp = xz_slice(grid['x'], patch_pole=True)
    zp = xz_slice(grid['z'])
    KEL0p = xz_slice(KEL0)
    KEL1p = xz_slice(KEL1)
    KEL2p = xz_slice(KEL2)
    KEL3p = xz_slice(KEL3)

    fig = plt.figure(figsize=(16,9))
    heights = [1,5]
    gs = gridspec.GridSpec(nrows=2, ncols=4, height_ratios=heights, figure=fig)

    ax0 = fig.add_subplot(gs[0,:])
    ax0.annotate('t= '+str(t),xy=(0.5,0.5),xycoords='axes fraction',va='center',ha='center',fontsize='x-large')
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[1,0])
    KEL0plot = ax1.pcolormesh(xp, zp, KEL0p, cmap=cmap, vmin=0, vmax=0.1, shading=shading)
    ax1.set_xlabel('$x (GM/c^2)$')
    ax1.set_ylabel('$z (GM/c^2)$')
    ax1.set_xlim(domain[:2])
    ax1.set_ylim(domain[2:])
    ax1.set_title('KEL0',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax1.add_artist(circle)
    ax1.set_aspect('equal')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(KEL0plot, cax=cax)

    ax2 = fig.add_subplot(gs[1,1])
    KEL1plot = ax2.pcolormesh(xp, zp, KEL1p, cmap=cmap, vmin=0, vmax=0.1, shading=shading)
    ax2.set_xlabel('$x (GM/c^2)$')
    ax2.set_ylabel('$z (GM/c^2)$')
    ax2.set_xlim(domain[:2])
    ax2.set_ylim(domain[2:])
    ax2.set_title('KEL1',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax2.add_artist(circle)
    ax2.set_aspect('equal')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(KEL1plot, cax=cax)

    ax3 = fig.add_subplot(gs[1,2])
    KEL2plot = ax3.pcolormesh(xp, zp, KEL2p, cmap=cmap, vmin=0, vmax=0.1, shading=shading)
    ax3.set_xlabel('$x (GM/c^2)$')
    ax3.set_ylabel('$z (GM/c^2)$')
    ax3.set_xlim(domain[:2])
    ax3.set_ylim(domain[2:])
    ax3.set_title('KEL2',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax3.add_artist(circle)
    ax3.set_aspect('equal')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(KEL2plot, cax=cax)

    ax4 = fig.add_subplot(gs[1,3])
    KEL3plot = ax4.pcolormesh(xp, zp, KEL3p, cmap=cmap, vmin=0, vmax=0.1, shading=shading)
    ax4.set_xlabel('$x (GM/c^2)$')
    ax4.set_ylabel('$z (GM/c^2)$')
    ax4.set_xlim(domain[:2])
    ax4.set_ylim(domain[2:])
    ax4.set_title('KEL3',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax3.add_artist(circle)
    ax3.set_aspect('equal')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(KEL3plot, cax=cax)

    plt.tight_layout()
    plt.savefig(os.path.join(globalvars['PLOTSDIR'],'{}_KEL_plot_{:04d}.png'.format(globalvars['PROB'],dumpval)))
    plt.close()

# The actual function that computes and plots diagnostics for PROB=torus and NDIMS=3
def analysis_torus3d(dumpval, cmap='jet', vmin=-5, vmax=0, domain = [-50,50,-50,50], bh=True, shading='gouraud'):
    plt.clf()
    print("Analyzing {0:04d} dump".format(dumpval))
    dfile = h5py.File(os.path.join(globalvars['DUMPSDIR'],'dump_0000{0:04d}.h5'.format(dumpval)),'r')
    rho = dfile['prims'][()][Ellipsis,0]
    uu = np.array(dfile['prims'][()][Ellipsis,1])
    u = np.array(dfile['prims'][()][Ellipsis,2:5])
    B = np.array(dfile['prims'][()][Ellipsis,5:8])
    gam = np.array(dfile['header/gam'][()])
    t = dfile['t'][()]
    dfile.close()
    t = "{:.3f}".format(t)
    logrho=np.log10(rho)
    pg = (gam-1)*uu
    gti = grid['gcon'][Ellipsis,0,1:4]
    gij = grid['gcov'][Ellipsis,1:4,1:4]
    beta_i = np.einsum('ijks,ijk->ijks',gti,grid['lapse']**2)
    qsq = np.einsum('ijky,ijky->ijk',np.einsum('ijkxy,ijkx->ijky',gij,u),u)
    gamma = np.sqrt(1+qsq)
    ui = u-np.einsum('ijks,ijk->ijks',beta_i,gamma/grid['lapse'])
    ut = gamma/grid['lapse']
    ucon = np.append(ut[Ellipsis,None],ui,axis=3)
    ucov = np.einsum('ijkmn,ijkn->ijkm',grid['gcov'],ucon)
    bt = np.einsum('ijkm,ijkm->ijk',np.einsum('ijksm,ijks->ijkm',grid['gcov'][Ellipsis,1:4,:],B),ucon)
    bi = (B+np.einsum('ijks,ijk->ijks',ui,bt))/ut[Ellipsis,None]
    bcon = np.append(bt[Ellipsis,None],bi,axis=3)
    bcov = np.einsum('ijkmn,ijkn->ijkm',grid['gcov'],bcon)
    bsq = np.einsum('ijkm,ijkm->ijk',bcon,bcov)
    betainv = 0.5*bsq/pg
    logbetainv = np.log10(betainv)

    xp = xz_slice(grid['x'], patch_pole=True)
    zp = xz_slice(grid['z'])
    rhop = xz_slice(logrho)
    betainvp = xz_slice(logbetainv)

    xt = xy_slice(grid['x'])
    yt = xy_slice(grid['y'],patch_phi=True)
    rhot = xy_slice(logrho)
    betainvt = xy_slice(logbetainv)

    fig = plt.figure(figsize=(16,9))
    heights = [1,5,5]
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=heights, figure=fig)

    ax0 = fig.add_subplot(gs[0,:])
    ax0.annotate('t= '+str(t),xy=(0.5,0.5),xycoords='axes fraction',va='center',ha='center',fontsize='x-large')
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[1,0])
    rhopolplot = ax1.pcolormesh(xp, zp, rhop, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    plotting_bfield_lines(ax1,B[Ellipsis,0],B[Ellipsis,1],nlines=40)
    ax1.set_xlabel('$x (GM/c^2)$')
    ax1.set_ylabel('$z (GM/c^2)$')
    ax1.set_xlim(domain[:2])
    ax1.set_ylim(domain[2:])
    ax1.set_title('Log($\\rho$)',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax1.add_artist(circle)
    ax1.set_aspect('equal')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(rhopolplot, cax=cax)

    ax2 = fig.add_subplot(gs[1,1])
    rhotorplot = ax2.pcolormesh(xt, yt, rhot, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    ax2.set_xlabel('$x (GM/c^2)$')
    ax2.set_ylabel('$y (GM/c^2)$')
    ax2.set_xlim(domain[:2])
    ax2.set_ylim(domain[2:])
    ax2.set_title('Log($\\rho$)',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax2.add_artist(circle)
    ax2.set_aspect('equal')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(rhotorplot, cax=cax)

    ax3 = fig.add_subplot(gs[2,0])
    betainvpolplot = ax3.pcolormesh(xp, zp, betainvp, cmap=cmap, vmin=-3, vmax=3, shading=shading)
    ax3.set_xlabel('$x (GM/c^2)$')
    ax3.set_ylabel('$z (GM/c^2)$')
    ax3.set_xlim(domain[:2])
    ax3.set_ylim(domain[2:])
    ax3.set_title('Log($\\beta^{-1}$)',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax3.add_artist(circle)
    ax3.set_aspect('equal')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(betainvpolplot, cax=cax)

    ax4 = fig.add_subplot(gs[2,1])
    betainvtorplot = ax4.pcolormesh(xt, yt, betainvt, cmap=cmap, vmin=-3, vmax=3, shading=shading)
    ax4.set_xlabel('$x (GM/c^2)$')
    ax4.set_ylabel('$y (GM/c^2)$')
    ax4.set_xlim(domain[:2])
    ax4.set_ylim(domain[2:])
    ax4.set_title('Log($\\beta^{-1}$)',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax4.add_artist(circle)
    ax4.set_aspect('equal')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(betainvtorplot, cax=cax)

    plt.tight_layout()
    plt.savefig(os.path.join(globalvars['PLOTSDIR'],'{}_basic_plot_{:04d}.png'.format(globalvars['PROB'],dumpval)))
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
    dstart = int(sorted(os.listdir(globalvars['DUMPSDIR']))[0][-7:-3])
    dend = int(sorted(list(filter(lambda dump: 'dump' in dump,os.listdir(globalvars['DUMPSDIR']))))[-1][-7:-3])
    dlist = range(dstart,dend+1)
    Ndumps = dend-dstart+1

    # Setting grid dict
    gfile = h5py.File(os.path.join(globalvars['DUMPSDIR'],'grid.h5'),'r')
    dfile = h5py.File(os.path.join(globalvars['DUMPSDIR'],'dump_0000{0:04d}.h5'.format(dstart)),'r')
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
