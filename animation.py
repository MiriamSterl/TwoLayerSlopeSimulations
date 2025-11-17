# Animations of how particles are moving through the flow fields, showing the eddy PV field

import h5py
import numpy as np
from numpy.fft import irfft2
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
import cmocean as cmo
import sys


#%% Read in parameters
slope = sys.argv[1]
field = sys.argv[2]
run = sys.argv[3]
background = sys.argv[4]


#%% Coordinates
x = np.load("../../Results/Results_GeophysicalFlows/x.npy")
y = np.load("../../Results/Results_GeophysicalFlows/y.npy")


#%% Reading in the potential vorticity field

def get_qhi(qhi):
    sh = qhi.shape
    qhiZ = np.empty(sh, dtype=np.complex64)
    for i in range(sh[0]):
        for j in range(sh[1]):
            for k in range(sh[2]):
                zijk = qhi[i, j, k]
                qhiZ[i, j, k] = zijk[0] + 1j*zijk[1]

    return qhiZ


f = h5py.File("../../Results/Results_GeophysicalFlows/SmallLd/simulation_s"+slope+"_strongmu_field"+field+"_equilibrium.jld2", "r")

# Time
t = np.array([float(ti) for ti in f["snapshots"]["t"].keys()])
idxs = t.argsort()
t = t[idxs]
time = t[0::12] # downsample to 12 hrs

# Read in qh, u, v data
qh = [f["snapshots"]["qh"][str(int(ti))][:] for ti in time]

# Get correct time array
dt = np.array(f["clock"]["dt"]).flatten()[0]
t = t*dt
t = t - t[0] # make time start at 0
time = t[0::12] # downsample to 12 hrs

# Get q from qh (Fourier transform)
qhi = np.zeros(np.shape(qh), dtype=np.complex64)
for i in range(np.shape(qhi)[0]):
    qhi[i] = get_qhi(qh[i])

q = irfft2(qhi, axes=(-2, -1))


#%% Defining the background PV gradient

Q1 = np.zeros((len(y),len(x)))
Q2 = np.zeros((len(y),len(x)))

if background=='1':
    Qy = np.array(f["params"]["Qy"]).flatten()
    Qy_upper = Qy[0]
    Qy_lower = Qy[-1]
    for i in range(len(x)):
        Q1[:,i] = Qy_upper*y
        Q2[:,i] = Qy_lower*y




#%% Reading in the particle data

data1 = xr.open_zarr('../../Results/Results_Parcels/SmallLd/simulation_s'+slope+'_strongmu_field'+field
                    +'_advection_layer1_run'+run+'_1hr12hr.zarr')
data2 = xr.open_zarr('../../Results/Results_Parcels/SmallLd/simulation_s'+slope+'_strongmu_field'+field
                    +'_advection_layer2_run'+run+'_1hr12hr.zarr')
t = (data1['time'].values[0,0:801] / np.timedelta64(1,'s')).astype(int)
t = t - t[0]
lon1 = data1['lon'].values[:,0:801]
lat1 = data1['lat'].values[:,0:801]
lon2 = data2['lon'].values[:,0:801]
lat2 = data2['lat'].values[:,0:801]

time_offset = int(int(run)*30*86400/(12*3600))


# Mark particles trapped in monster eddies in upper layer
# Read in flow field data at last time step of particle run
tGF = np.array([float(ti) for ti in f["snapshots"]["t"].keys()])
dt = np.array(f["clock"]["dt"]).flatten()[0]
idxs = tGF.argsort()
tGF = tGF[idxs]
start = int(int(run)*30*24) # particle run i starts after 30*i days
step = int(400*24) # particles are advected for 400 days
ti = tGF[start+step-1] # last time step in the data
qh_end = f["snapshots"]["qh"][str(int(ti))][:]
qhi_end = get_qhi(qh_end)
q_end = irfft2(qhi_end, axes=(-2, -1))
qlayer_end = q_end[0,:,:]

# interpolate q to particle locations
interp_func = RegularGridInterpolator((y,x),qlayer_end)
points = np.column_stack((lat1[:,-1], lon1[:,-1]))
q_interp = interp_func(points)

# identify particles with anomalously high q values (i.e., more than 2 standard deviations from the mean)
mean_q = np.nanmean(qlayer_end)
std_q = np.nanstd(qlayer_end)
deviation = np.abs(q_interp - mean_q)
anom = np.where(deviation > 2.5*std_q)

# remove particles with anomalously high q values
lon1_fine = np.delete(lon1,anom,axis=0) # remove trapped/filtered particles
lat1_fine = np.delete(lat1,anom,axis=0)
lon1_filter = lon1[anom] # the filtered particles
lat1_filter = lat1[anom]



#%% Making the animation

levels_layer1 = np.linspace(-5e-4,5e-4,25)
levels_layer2 = np.linspace(-5e-5,5e-5,25)
ticks_layer1 = [-4e-4,-2e-4,0,2e-4,4e-4]
ticks_layer2 = [-4e-5,-2e-5,0,2e-5,4e-5]
ticklabels_layer1 = [r'$-4$', r'$-2$', r'$0$', r'$2$', r'$4$']
ticklabels_layer2 = [r'$-4$', r'$-2$', r'$0$', r'$2$', r'$4$']


fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,6))
cax1 = ax1.contourf(x/1e3,y/1e3,q[0+time_offset,0,:,:]+Q1,cmap='cmo.balance',extend='both',levels=levels_layer1)
cbar1 = fig.colorbar(cax1,ax=ax1,ticks=ticks_layer1)
cbar1.ax.set_yticklabels(ticklabels_layer1,fontsize=11)
cbar1.set_label(r'$q$ ($10^{-4}$ s$^{-1}$)',fontsize=12)
particles1_fine = ax1.scatter(lon1_fine[:,0]/1e3,lat1_fine[:,0]/1e3,s=5,c='green',zorder=10)
particles1_filter = ax1.scatter(lon1_filter[:,0]/1e3,lat1_filter[:,0]/1e3,s=5,c='white',zorder=10)
ax1.set_title('Upper layer')

cax2 = ax2.contourf(x/1e3,y/1e3,q[0+time_offset,1,:,:]+Q2,cmap='cmo.balance',extend='both',levels=levels_layer2)
cbar2 = fig.colorbar(cax2,ax=ax2,ticks=ticks_layer2)
cbar2.ax.set_yticklabels(ticklabels_layer2,fontsize=11)
cbar2.set_label(r'$q$ ($10^{-5}$ s$^{-1}$)',fontsize=12)
particles2 = ax2.scatter(lon2[:,0]/1e3,lat2[:,0]/1e3,s=5,c='green',zorder=10)
ax2.set_title('Lower layer')

for ax in [ax1,ax2]:
    ax.set_xlabel('x (km)',fontsize=12)
    ax.set_ylabel('y (km)',fontsize=12)

title = fig.suptitle('t = '+str(t[0]/86400)+' days, slope '+slope,fontsize=16)
plt.tight_layout()


def update(i):
    global cax1, cax2, particles1, particles2, title

    # Remove previous contours
    for cf in cax1.collections:
        cf.remove()
    for cf in cax2.collections:
        cf.remove()

    # Create new contours
    cax1 = ax1.contourf(x/1e3,y/1e3,q[i+time_offset,0,:,:]+Q1,cmap='cmo.balance',extend='both',levels=levels_layer1)
    cax2 = ax2.contourf(x/1e3,y/1e3,q[i+time_offset,1,:,:]+Q2,cmap='cmo.balance',extend='both',levels=levels_layer2)

    # Update scatter plot instead of redrawing
    particles1_fine.set_offsets(np.c_[lon1_fine[:, i] / 1e3, lat1_fine[:, i] / 1e3])
    particles1_filter.set_offsets(np.c_[lon1_filter[:, i] / 1e3, lat1_filter[:, i] / 1e3])
    particles2.set_offsets(np.c_[lon2[:, i] / 1e3, lat2[:, i] / 1e3])

    # Update title
    title.set_text(f't = {t[i]/86400:.2f} days, slope {slope}')


anim = matplotlib.animation.FuncAnimation(fig, update, frames=len(t), repeat=False, interval=33)

if background=='1':
    fname = "animation_s"+slope+"_field"+field+"_run"+run+"_fullPV.mp4"
else:
    fname = "animation_s"+slope+"_field"+field+"_run"+run+".mp4"
anim.save("../../Results/Animations/"+fname,dpi=300,fps=15)