import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import py21cmfast as p21c
import tools21cm as t2c 
import pickle, os, sys
from time import time 
from glob import glob

sysargv = np.array(sys.argv); print(['{}'.format(i) for i in sysargv])

def check_sysargv(knob, fall_back, d_type):
    smooth_info = np.array([knob in a for a in sys.argv])
    if np.any(smooth_info): 
        smooth_info = np.array(sysargv)[smooth_info]
        smooth_info = smooth_info[0].split(knob)[-1]
        smooth = d_type(smooth_info)
    else:
        smooth = d_type(fall_back)
    return smooth 

cosmo_params = p21c.CosmoParams(SIGMA_8=0.8)
astro_params = p21c.AstroParams({"HII_EFF_FACTOR":20.0})
flag_options = {"INHOMO_RECO":False, 'USE_MASS_DEPENDENT_ZETA':False, 'USE_TS_FLUCT':False}
print(astro_params)


data_direc = './data/' 
random_seed = 2

user_params  = p21c.UserParams({"HII_DIM":  100, "DIM": 300, "BOX_LEN": 100, "FIXED_IC": 1})
ic_A = p21c.initial_conditions(
		    user_params  = user_params,
		    cosmo_params = cosmo_params,
		    random_seed  = random_seed,
		    direc        = data_direc,
		    regenerate   = True if '--clean_data' in sysargv else None,
		)
user_params  = p21c.UserParams({"HII_DIM":  100, "DIM": 300, "BOX_LEN": 100, "FIXED_IC": -1})
ic_B = p21c.initial_conditions(
		    user_params  = user_params,
		    cosmo_params = cosmo_params,
		    random_seed  = random_seed,
		    direc        = data_direc,
		    regenerate   = True if '--clean_data' in sysargv else None,
		)

ics = {'A': ic_A, 'B': ic_B}
cubes = {'A': {}, 'B': {}}
pses  = {'A': {}, 'B': {}}

fig, axs = plt.subplots(1,2,figsize=(14,5))
fig.suptitle('ICs')
slices = [ic_A.hires_density[10],ic_B.hires_density[10]]
for i,ax in enumerate(axs):
	sl = slices[i]
	im = ax.pcolor(
		np.linspace(0,user_params.BOX_LEN,sl.shape[0]), np.linspace(0,user_params.BOX_LEN,sl.shape[1]),
		sl, cmap='jet',
		)
	ax.set_xlabel('X [Mpc]')
	ax.set_ylabel('Y [Mpc]')
	fig.colorbar(im, ax=ax);
plt.tight_layout()
plt.show()


for ke, it in ics.items():
	coeval7, coeval8, coeval9 = p21c.run_coeval(
		    redshift = [7.0, 8.0, 9.0],
		    # user_params  = user_params,
		    # cosmo_params = cosmo_params,
		    astro_params = astro_params,
		    # random_seed  = random_seed,
		    direc        = data_direc,
		    init_box     = it,
		)
	cubes[ke][7] = coeval7
	cubes[ke][8] = coeval8
	cubes[ke][9] = coeval9
	ps7, ks7 = t2c.power_spectrum_1d(coeval7.brightness_temp, kbins=15, box_dims=user_params.BOX_LEN)
	ps8, ks8 = t2c.power_spectrum_1d(coeval8.brightness_temp, kbins=15, box_dims=user_params.BOX_LEN)
	ps9, ks9 = t2c.power_spectrum_1d(coeval9.brightness_temp, kbins=15, box_dims=user_params.BOX_LEN)
	pses[ke][7] = ps7
	pses[ke][8] = ps8
	pses[ke][9] = ps9
pses['k'] = ks9

fig, axs = plt.subplots(1,2,figsize=(14,5))
slices = [cubes['A'][9].brightness_temp[10], cubes['B'][9].brightness_temp[10]]
fig.suptitle('z=9')
for i,ax in enumerate(axs):
	sl = slices[i]
	im = ax.pcolor(
		np.linspace(0,user_params.BOX_LEN,sl.shape[0]), np.linspace(0,user_params.BOX_LEN,sl.shape[1]),
		sl, cmap='jet',
		)
	ax.set_xlabel('X [Mpc]')
	ax.set_ylabel('Y [Mpc]')
	fig.colorbar(im, ax=ax);
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1,2,figsize=(14,5))
fig.suptitle('z=8')
slices = [cubes['A'][8].brightness_temp[10], cubes['B'][8].brightness_temp[10]]
for i,ax in enumerate(axs):
	sl = slices[i]
	im = ax.pcolor(
		np.linspace(0,user_params.BOX_LEN,sl.shape[0]), np.linspace(0,user_params.BOX_LEN,sl.shape[1]),
		sl, cmap='jet',
		)
	ax.set_xlabel('X [Mpc]')
	ax.set_ylabel('Y [Mpc]')
	fig.colorbar(im, ax=ax);
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(1,3,figsize=(14,4.5))
zs = [9,8,7]
ks = pses['k']
for i,ax in enumerate(axs):
	ax.set_title('z={}'.format(zs[i]))
	psA = pses['A'][zs[i]]
	psB = pses['B'][zs[i]]
	ax.loglog(ks, ks**3/2/np.pi**2*psA, label='A', ls='--')
	ax.loglog(ks, ks**3/2/np.pi**2*psB, label='B', ls='-.')
	ax.set_xlabel('k [1/Mpc]')
	if i>0: ax.set_ylabel('$\Delta^2_{21}$ [mK$^2$]')
	else: ax.set_ylabel('$\Delta^2_{\delta\delta}$')
	ax.legend()
plt.tight_layout()
plt.show()



