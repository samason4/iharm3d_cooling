################################################################################
#                                                                              #
#  PLOTS OF VARIABLES COMPUTED IN eht_analysis.py                              #
#                                                                              #
################################################################################

import matplotlib
matplotlib.use('Agg')

import util

import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

# For radials
FIGX = 14
FIGY = 10
# For flux plots; per-plot Y dim
PLOTY = 3
SIZE = 40

RADS = True
OMEGA = True
FLUXES = True
EXTRAS = True
DIAGS = True

def plot_multi(ax, iname, varname, varname_pretty, logx=False, logy=False, ylim=None, timelabels=False):
  for i,avg in enumerate(avgs):
    if varname in avg.keys():
      ax.plot(avg[iname], avg[varname], styles[i], label=labels[i])
  # Plot additions
  if logx: ax.set_xscale('log')
  if logy: ax.set_yscale('log')
  if ylim is not None: ax.set_ylim(ylim)
  ax.grid()
  ax.set_ylabel(varname_pretty)
  if iname == 't':
    ax.set_xlim([ti,tf])
    if timelabels:
      ax.set_xlabel('t/M')
    else:
      ax.set_xticklabels([])
  elif iname == 'th':
    ax.set_xlabel(r"$\theta$")
  elif iname == 'r':
    ax.set_xlabel("r")
    ax.set_xlim([0,50]) # For EHT comparison

def plot_rads():
  fig, ax = plt.subplots(2,3, figsize=(FIGX, FIGY))
  plot_multi(ax[0,0], 'r', 'rho_r', r"$<\rho>$", logy=True, ylim=[1.e-2, 1.e0])
  plot_multi(ax[0,1], 'r', 'Pg_r', r"$<P_g>$", logy=True, ylim=[1.e-6, 1.e-2])
  plot_multi(ax[0,2], 'r', 'B_r', r"$<|B|>$", logy=True, ylim=[1.e-4, 1.e-1])
  plot_multi(ax[1,0], 'r', 'uphi_r', r"$<u^{\phi}>$", logy=True, ylim=[1.e-3, 1.e1])
  plot_multi(ax[1,1], 'r', 'Ptot_r', r"$<P_{tot}>$", logy=True, ylim=[1.e-6, 1.e-2])
  plot_multi(ax[1,2], 'r', 'betainv_r', r"$<\beta^{-1}>$", logy=True, ylim=[1.e-2, 1.e1])

  if len(avgs) > 1: ax[0,2].legend(loc=1)

  plt.savefig(fname_out + '_ravgs.png')
  plt.close(fig)

def plot_omega():
  # Omega
  fig, ax = plt.subplots(2,2, figsize=(FIGX, FIGY))
  # Renormalize omega to omega/Omega_H for plotting
  for avg in avgs:
    if 'omega_th' in avg.keys(): #Then both are
      avg['omega_th'] *= 4/avg['a']
      avg['omega_th_av'] *= 4/avg['a']
  plot_multi(ax[0,0], 'th', 'omega_th', r"$\omega_f$ (EH, single shell)", ylim=[-1,2])
  plot_multi(ax[0,1], 'th', 'omega_th_av', r"$\omega_f$ (EH, 5-zone average)", ylim=[-1,2])

  # Legend
  if len(avgs) > 1: ax[0,0].legend(loc=3)

  # Horizontal guidelines
  for a in ax.flatten():
    a.axhline(0.5, linestyle='--', color='k')

  plt.savefig(fname_out + '_omega.png')
  plt.close(fig)

def plot_fluxes():
  fig,ax = plt.subplots(5,1, figsize=(FIGX, 5*PLOTY))

  plot_multi(ax[0],'t','Mdot', r"$|\dot{M}|$")
  plot_multi(ax[1],'t','Phi', r"$\Phi$")

  for avg in avgs:
    if 'Ldot' in avg.keys():
      avg['abs_Ldot'] = np.abs(avg['Ldot'])
  plot_multi(ax[2],'t','abs_Ldot', r"$|\dot{L}|$")

  for avg in avgs:
    if 'Edot' in avg.keys() and 'Mdot' in avg.keys():
      avg['EmM'] = np.fabs(np.fabs(avg['Edot']) - np.fabs(avg['Mdot']))
  plot_multi(ax[3], 't', 'EmM', r"$|\dot{E} - \dot{M}|$")

  plot_multi(ax[4], 't', 'Lum', "Lum", timelabels=True)

  if len(avgs) > 1: ax[0].legend(loc=2)

  plt.savefig(fname_out + '_fluxes.png')
  plt.close(fig)

  # Don't print the norm fluxes if you can't norm
  if 'Mdot' not in avg.keys():
    return

  fig, ax = plt.subplots(5,1, figsize=(FIGX, 5*PLOTY))
  plot_multi(ax[0], 't', 'Mdot', r"$|\dot{M}|$")
  if len(avgs) > 1: ax[0].legend(loc=2)

  for avg in avgs:
    if 'Phi' in avg.keys():
      avg['phi'] = avg['Phi']/np.sqrt(np.fabs(avg['Mdot']))
  plot_multi(ax[1], 't', 'phi', r"$\frac{\Phi}{\sqrt{|\dot{M}|}}$")

  for avg in avgs:
    if 'Ldot' in avg.keys():
      avg['ldot'] = np.fabs(avg['Ldot'])/np.fabs(avg['Mdot'])
  plot_multi(ax[2], 't', 'ldot', r"$\frac{|Ldot|}{|\dot{M}|}$")

  for avg in avgs:
    if 'Edot' in avg.keys():
      avg['edot'] = np.fabs(np.fabs(avg['Edot']) - np.fabs(avg['Mdot']))/(np.fabs(avg['Mdot']))
  plot_multi(ax[3], 't', 'edot', r"$\frac{|\dot{E} - \dot{M}|}{|\dot{M}|}$")

  for avg in avgs:
    if 'Lum' in avg.keys():
      avg['lum'] = np.fabs(avg['Lum'])/np.fabs(avg['Mdot'])
  plot_multi(ax[4], 't', 'lum', r"$\frac{Lum}{|\dot{M}|}$", timelabels=True)

  plt.savefig(fname_out + '_normfluxes.png')
  plt.close(fig)

def plot_extras():
  fig, ax = plt.subplots(3,1, figsize=(FIGX, 3*PLOTY))

  # Efficiency over mean mdot: just use the back half as <>
  for avg in avgs:
    if 'Edot' in avg.keys():
      mdot_mean = np.abs(np.mean(avg['Mdot'][len(avg['Mdot'])//2:]))
      avg['Eff'] = np.abs(avg['Mdot'] - avg['Edot'])/mdot_mean*100
  plot_multi(ax[0], 't', 'Eff', "Efficiency (%)", ylim=[-10,200])

  plot_multi(ax[1], 't', 'Edot', r"$\dot{E}$")

  for avg in avgs:
    if 'LBZ' in avg.keys():
      avg['aLBZ'] = np.abs(avg['LBZ'])
  plot_multi(ax[2], 't', 'aLBZ', "BZ Luminosity", timelabels=True)

  plt.savefig(fname_out + '_extras.png')
  plt.close(fig)

def plot_diags():
  fig, ax = plt.subplots(3,1, figsize=(FIGX, FIGY/2))
  ax = ax.flatten()

  plot_multi(ax[0], 't', 'Etot', "Total E")
  plot_multi(ax[1], 't', 'sigma_max', r"$\sigma_{max}$")
  plot_multi(ax[2], 't_d', 'divbmax_d', "max divB", timelabels=True)

  plt.savefig(fname_out + '_diagnostics.png')
  plt.close(fig)

if __name__ == "__main__":
  if len(sys.argv) < 3:
    util.warn('Format: python eht_plot.py analysis_output [analysis_output ...] [labels_list] image_name')
    sys.exit()

  # All interior arguments are files to overplot, except possibly the last
  if len(sys.argv) < 4:
    last_file = -1
  else:
    last_file = -2


  avgs = []
  for filename in sys.argv[1:last_file]:
    avgs.append(pickle.load(open(filename,'rb')))

  # Split the labels, or use the filename as a label
  labels = [ lab.replace("/",",") for lab in sys.argv[-2].split(",") ]

  if len(labels) < len(avgs):
    util.warn("Too few labels!")
    sys.exit()

  fname_out = sys.argv[-1]

  # For time plots.  Also take MAD/SANE for axes?
  ti = avgs[0]['t'][0]
  tf = avgs[0]['t'][-1]

  # Default styles
  if len(avgs) > 1:
    styles = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
  else:
    styles = ['k']
    
  if RADS: plot_rads()
  if OMEGA: plot_omega()
  if FLUXES: plot_fluxes()
  if EXTRAS: plot_extras()
  if DIAGS: plot_diags()

