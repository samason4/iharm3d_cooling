import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os,psutil,sys
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprocessing as mp

# Initialize global variables
globalvars_keys = ['PROB','NDIMS','DUMPSDIR9','DUMPSDIR5','DUMPSDIR2','DUMPSDIR05','PLOTSDIR']
globalvars = {}
grid ={}

# python3 prob/flat_space/conv_flat.py -p ./prob/flat_space/params_analysis.dat

# The actual function that computes and plots diagnostics for PROB=torus and NDIMS=2
def find_error(dumpsdir, i, j):
    dfile = h5py.File(os.path.join(globalvars[dumpsdir],'dump_0000{0:04d}.h5'.format(100)),'r')
    rho = dfile['prims'][()][Ellipsis,0].reshape(grid['n1'], grid['n2'], grid['n3'])[i][j][0]
    KEL0 = np.array(dfile['prims'][()][Ellipsis,9]).reshape(grid['n1'], grid['n2'], grid['n3'])[i][j][0]
    t = dfile['t'][()]
    game = 1.333333
    dfile.close()
    t = "{:.3f}".format(t)

    #numerical:
    print("numerical rho: ", rho, "and kel0: ", KEL0)
    u_num = rho**game*KEL0/(game-1)

    #analytical:
    dfile_init = h5py.File(os.path.join(globalvars[dumpsdir],'dump_0000{0:04d}.h5'.format(0000)),'r')
    rho_init = dfile_init['prims'][()][Ellipsis,0].reshape(grid['n1'], grid['n2'], grid['n3'])[i][j][0]
    KEL0_init = np.array(dfile_init['prims'][()][Ellipsis,9]).reshape(grid['n1'], grid['n2'], grid['n3'])[i][j][0]
    dfile_init.close()
    print("initial rho: ", rho_init, "and kel0: ", KEL0_init)
    u0 = rho_init**game*KEL0_init/(game-1)
    alpha = -0.2
    print("time: ", float(t))
    u_ana = u0*np.exp(alpha*float(t))

    #error:
    print("error: ", abs(u_num-u_ana))
    return (abs(u_num-u_ana))


# main(): Reads param file, writes grid dict and calls analysis function
if __name__=="__main__":
    #if there are more than one arguments and the first is -p
    if len(sys.argv) > 1 and sys.argv[1]=='-p':
        fparams_name = sys.argv[2] # the second argument needs to be the params_analysis.dat from flat_space
    else:
        sys.exit('No param file provided')

    # Reading the param file
    with open(fparams_name,'r') as fparams:
        lines = fparams.readlines()
        for line in lines:
            if line[0]=='#' or line.isspace(): pass #exclude comments and whitespace
            elif line.split()[0] in globalvars_keys: globalvars[line.split()[0]]=line.split()[-1]

    # Creating the output directory if it doesn't exist
    if not os.path.exists(globalvars['PLOTSDIR']):
        os.makedirs(globalvars['PLOTSDIR'])

    # Calculating total dump files
    dstart = int(sorted(os.listdir(globalvars['DUMPSDIR9']))[0][-7:-3])
    dend = int(sorted(list(filter(lambda dump: 'dump' in dump,os.listdir(globalvars['DUMPSDIR9']))))[-1][-7:-3])
    dlist = range(dstart,dend+1)
    Ndumps = dend-dstart+1

    # Setting grid dict
    gfile = h5py.File(os.path.join(globalvars['DUMPSDIR9'],'grid.h5'),'r')
    dfile = h5py.File(os.path.join(globalvars['DUMPSDIR9'],'dump_0000{0:04d}.h5'.format(dstart)),'r')
    grid['n1'] = dfile['/header/n1'][()]
    grid['n2'] = dfile['/header/n2'][()]
    grid['n3'] = dfile['/header/n3'][()]
    print("grid['n1']: ", grid['n1'])
    dfile.close()
    gfile.close()

    #getting errors:
    errors = []
    cour_inv = []
    errors9 = 0
    errors5 = 0
    errors2 = 0
    errors05 = 0
    for i in range(10):
        for j in range(10):
            errors9 += find_error('DUMPSDIR9', i, j)
            errors5 += find_error('DUMPSDIR5', i, j)
            errors2 += find_error('DUMPSDIR2', i, j)
            errors05 += find_error('DUMPSDIR05', i, j)
            print(i*10+j+1, "percent done")
    errors.append(errors9/100)
    cour_inv.append(1/.9)
    errors.append(errors5/100)
    cour_inv.append(1/.5)
    errors.append(errors2/100)
    cour_inv.append(1/.2)
    errors.append(errors05/100)
    cour_inv.append(1/.05)

    #this is for the comparison line:
    x = []
    res = []
    x2 = []
    res2 = []
    temp_res = 1
    temp_x = 1e-15
    for i in range(12):
        x.append(temp_x*temp_res**(-2))
        res.append(temp_res)
        temp_res += 2
    temp_res2 = 1
    temp_x2 = 1e-15
    for i in range(12):
        x2.append(temp_x2*temp_res2**(-1))
        res2.append(temp_res2)
        temp_res2 += 2

    #plotting:
    fig1, sub1 = plt.subplots()
    sub1.loglog(cour_inv, errors, color = 'b', label = 'Error of Test Cooling')
    sub1.loglog(res, x, color = 'r', label = 'Line of Slope N^-2 for Comparison')
    sub1.loglog(res2, x2, color = 'g', label = 'Line of Slope N^-1 for Comparison')
    sub1.loglog(cour_inv, errors, 'bo')
    plt.xticks([], [])
    sub1.set_xticks([])
    sub1.set_xticks([], minor=True)
    sub1.set_xticks([1, 2, 4, 8, 16, 32], ['2^0', '2^1', '2^2', '2^3', '2^4', '2^5'])
    plt.ylabel("Total Error")
    plt.xlabel("1 / The Courant Number")
    plt.title('Error vs 1/cour')
    plt.legend()
    plt.savefig('error_vs_cour.png')
    plt.close()