######################################################################
#
# Simple analysis and plotting script to process problem output
# Plots mhdmodes, bondi and torus problem (2D and 3D)
#
######################################################################

# python3 ./script/analysis/cooling/combined_plots.py -p ./script/analysis/cooling/combined_params.dat
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
globalvars_keys = ['PROB','NDIMS','DUMPSDIR_with','DUMPSDIR_without','PLOTSDIR']
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

def analysis_torus2d(dumpval, cmap='jet', vmin=-5, vmax=0, domain = [-50,0,-50,50], bh=True, shading='gouraud'):
    plt.clf()
    print("Analyzing {0:04d} dump".format(dumpval))
    dfile = h5py.File(os.path.join(globalvars['DUMPSDIR_with'],'dump_0000{0:04d}.h5'.format(dumpval)),'r')
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

#**********************************************************************************************************************

    dfile2 = h5py.File(os.path.join(globalvars['DUMPSDIR_without'],'dump_0000{0:04d}.h5'.format(dumpval)),'r')
    rho2 = dfile2['prims'][()][Ellipsis,0]
    uu2 = np.array(dfile2['prims'][()][Ellipsis,1])
    u2 = np.array(dfile2['prims'][()][Ellipsis,2:5])
    B2 = np.array(dfile2['prims'][()][Ellipsis,5:8])
    gam2 = np.array(dfile2['header/gam'][()])
    t2 = dfile2['t'][()]
    dfile2.close()
    t2 = "{:.3f}".format(t2)
    logrho2=np.log10(rho2)
    pg2 = (gam2-1)*uu2
    gti2 = grid['gcon'][Ellipsis,0,1:4]
    gij2 = grid['gcov'][Ellipsis,1:4,1:4]
    beta_i2 = np.einsum('ijks,ijk->ijks',gti2,grid['lapse']**2)
    qsq2 = np.einsum('ijky,ijky->ijk',np.einsum('ijkxy,ijkx->ijky',gij2,u2),u2)
    gamma2 = np.sqrt(1+qsq2)
    ui2 = u2-np.einsum('ijks,ijk->ijks',beta_i2,gamma2/grid['lapse'])
    ut2 = gamma2/grid['lapse']
    ucon2 = np.append(ut2[Ellipsis,None],ui2,axis=3)
    ucov2 = np.einsum('ijkmn,ijkn->ijkm',grid['gcov'],ucon2)
    bt2 = np.einsum('ijkm,ijkm->ijk',np.einsum('ijksm,ijks->ijkm',grid['gcov'][Ellipsis,1:4,:],B2),ucon2)
    bi2 = (B2+np.einsum('ijks,ijk->ijks',ui2,bt2))/ut2[Ellipsis,None]
    bcon2 = np.append(bt2[Ellipsis,None],bi2,axis=3)
    bcov2 = np.einsum('ijkmn,ijkn->ijkm',grid['gcov'],bcon2)
    bsq2 = np.einsum('ijkm,ijkm->ijk',bcon2,bcov2)
    betainv2 = 0.5*bsq2/pg2
    logbetainv2 = np.log10(betainv2)

    xp2 = xz_slice(grid['x'], patch_pole=True)
    zp2 = xz_slice(grid['z'])
    rhop2 = xz_slice(logrho2)
    betainvp2 = xz_slice(logbetainv2)

    errors = np.array([])
    rho_reshaped = rho.reshape(16384)
    uu_reshaped = uu.reshape(16384)
    u_reshaped = u.reshape(49152)
    B_reshaped = B.reshape(49152)
    rho2_reshaped = rho2.reshape(16384)
    uu2_reshaped = uu2.reshape(16384)
    u2_reshaped = u2.reshape(49152)
    B2_reshaped = B2.reshape(49152)
    errors = np.append(errors, abs(rho_reshaped-rho2_reshaped))
    errors = np.append(errors, abs(uu_reshaped-uu2_reshaped))
    errors = np.append(errors, abs(u_reshaped-u2_reshaped))
    errors = np.append(errors, abs(B_reshaped-B2_reshaped))
    maxerror = np.amax(errors)
    maxerror = "{:.3f}".format(maxerror)


    fig = plt.figure(figsize=(16,9))
    heights = [1,5]
    gs = gridspec.GridSpec(nrows=2, ncols=4, height_ratios=heights, figure=fig)

    ax0 = fig.add_subplot(gs[0,:])
    ax0.annotate('t= '+str(t)+' maximum error of any prim = '+str(maxerror),xy=(0.5,0.5),xycoords='axes fraction',va='center',ha='center',fontsize='x-large')
    ax0.axis("off")

    #plotting rho with:
    ax1 = fig.add_subplot(gs[1,0])
    rhopolplot = ax1.pcolormesh(xp, zp, rhop, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    plotting_bfield_lines(ax1,B[Ellipsis,0],B[Ellipsis,1],nlines=40)
    ax1.set_xlabel('$x (GM/c^2)$')
    ax1.set_ylabel('$z (GM/c^2)$')
    ax1.set_xlim(domain[:2])
    ax1.set_ylim(domain[2:])
    ax1.set_title('Log($\\rho$) with COOLING',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax1.add_artist(circle)
    ax1.set_aspect('equal')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(rhopolplot, cax=cax)

    #plotting rho without:
    ax2 = fig.add_subplot(gs[1,1])
    rhopolplot2 = ax2.pcolormesh(xp2, zp2, rhop2, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
    plotting_bfield_lines(ax2,B2[Ellipsis,0],B2[Ellipsis,1],nlines=40)
    ax2.set_xlabel('$x (GM/c^2)$')
    ax2.set_ylabel('$z (GM/c^2)$')
    ax2.set_xlim(domain[:2])
    ax2.set_ylim(domain[2:])
    ax2.set_title('Log($\\rho$) without ELECTRONS',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax2.add_artist(circle)
    ax2.set_aspect('equal')
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(rhopolplot2, cax=cax2)

    #plotting betainv with:
    ax3 = fig.add_subplot(gs[1,2])
    betainvpolplot = ax3.pcolormesh(xp, zp, betainvp, cmap=cmap, vmin=-3, vmax=3, shading=shading)
    ax3.set_xlabel('$x (GM/c^2)$')
    ax3.set_ylabel('$z (GM/c^2)$')
    ax3.set_xlim(domain[:2])
    ax3.set_ylim(domain[2:])
    ax3.set_title('Log($\\beta^{-1}$) with COOLING',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax3.add_artist(circle)
    ax3.set_aspect('equal')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(betainvpolplot, cax=cax)

    #plotting betainv without:
    ax4 = fig.add_subplot(gs[1,3])
    betainvpolplot2 = ax4.pcolormesh(xp2, zp2, betainvp2, cmap=cmap, vmin=-3, vmax=3, shading=shading)
    ax4.set_xlabel('$x (GM/c^2)$')
    ax4.set_ylabel('$z (GM/c^2)$')
    ax4.set_xlim(domain[:2])
    ax4.set_ylim(domain[2:])
    ax4.set_title('Log($\\beta^{-1}$) without ELECTRONS',fontsize='large')
    if bh:
            circle = plt.Circle((0,0),grid['rEH'],color='k')
            ax4.add_artist(circle)
    ax4.set_aspect('equal')
    divider = make_axes_locatable(ax4)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(betainvpolplot, cax=cax2)

    plt.tight_layout()
    plt.savefig(os.path.join(globalvars['PLOTSDIR'],'{}_combined_plot_{:04d}.png'.format(globalvars['PROB'],dumpval)))
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
    dstart = int(sorted(os.listdir(globalvars['DUMPSDIR_with']))[0][-7:-3])
    dend = int(sorted(list(filter(lambda dump: 'dump' in dump,os.listdir(globalvars['DUMPSDIR_with']))))[-1][-7:-3])
    dlist = range(dstart,dend+1)
    Ndumps = dend-dstart+1

    # Setting grid dict
    gfile = h5py.File(os.path.join(globalvars['DUMPSDIR_with'],'grid.h5'),'r')
    dfile = h5py.File(os.path.join(globalvars['DUMPSDIR_with'],'dump_0000{0:04d}.h5'.format(dstart)),'r')
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
