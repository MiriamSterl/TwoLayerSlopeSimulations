# Compute timeseries of cross-stream velocity autocovariance (for each run of the ensemble)

import numpy as np
import pandas as pd
import xarray as xr
from itertools import product

#%%
def acov(x,lag_len,biased,normed):
    """ 
    compute autocovariance of a timeseries x
    """
    acov = np.empty(lag_len + 1)
    acov[0] = x.dot(x)
    for i in range(lag_len):
        acov[i + 1] = x[i + 1 :].dot(x[: -(i + 1)])
    if biased:
        acov /= len(x)
    else:
        acov /= len(x) - np.arange(lag_len + 1)
    if normed:
        return acov / acov[0]
    else:
        return acov


def v_acov(slope,layer,field,run,biased,normed): 
    """
    compute autocovariance of cross-stream velocity timeseries, averaged over all particles
    """
    data = xr.open_zarr('../../Results/Results_Parcels/SmallLd/simulation_s'+slope+'_strongmu_field'+field
                        +'_advection_layer'+layer+'_run'+run+'_1hr12hr.zarr')
    t = (data['time'].values[0,0:801] / np.timedelta64(1,'s')).astype(int)
    v = data['v'].values[:,0:801]
    lon = data['lon'].values[:,0:801]
    t = t - t[0]
    N = np.shape(v)[0] # number of particles
    T = np.shape(v)[1] # number of timesteps

    acovs = np.zeros(np.shape(v))
    for i in range(N):
        acovs[i] = acov(v[i,:],T-1,biased,normed)

    acovs_av = np.mean(acovs,axis=0)
    return acovs_av


def trapsum(vec,dt):
    """
    Compute the integral of a vector using the trapezoidal rule
    Spacing between vector entries is dt
    """
    trap = 0.5*dt*(vec[:-1] + vec[1:])
    return np.cumsum(trap)
    


def compute_acov_diff(slope,layer,field,biased,normed):
    """
    Compute & save timeseries of cross-stream velocity autocovariance (for each run of the ensemble)
    and Lagrangian diffusivity as integral of autocovariance (averaged over the ensemble)
    """
    # Compute & save autocovariance timeseries for each run of the ensemble
    acovs_av = v_acov(slope,layer,field,'0',biased,normed)
    for i in range(1,10):
        acovs_av_i = v_acov(slope,layer,field,str(i),biased,normed)
        acovs_av = np.vstack((acovs_av,acovs_av_i))
    df_acovs_av = pd.DataFrame(acovs_av)
    if biased:
        add1 = '_biased'
    else:
        add1 = ''
    if normed:
        add2 = '_normed'
    else:
        add2 = ''
    acov_fname = '../../Results/Results_Diffusivities/SmallLd/lagrangian/autocovariance_v_s'+slope+'_layer'+layer+'_field'+field+'_1hr12hr'+add1+add2+'.csv'
    df_acovs_av.to_csv(acov_fname,index=False)
    np_acovs_av = df_acovs_av.to_numpy()

    # Compute diffusivity by integrating the autocovariance
    dt = 0.5*86400 # timestep of autocovariance timeseries = 0.5 day (Parcels output)
    diff = np.zeros(np.shape(np_acovs_av))
    for i in range(np.shape(np_acovs_av)[0]):
        diff[i,:] = np.hstack((0,trapsum(np_acovs_av[i,:],dt)))

    df_diff = pd.DataFrame(diff)
    diff_fname = '../../Results/Results_Diffusivities/SmallLd/lagrangian/acov_diff_s'+slope+'_layer'+layer+'_field'+field+'_1hr12hr'+add1+add2+'.csv'
    df_diff.to_csv(diff_fname,index=False)



#%% Run calculation for each slope, layer, and field

# slopes = ['0', '1e-4', '-1e-4', '2e-4', '-2e-4', '3e-4', '-3e-4', '5e-4', '-5e-4', '7e-4', '-7e-4',
#           '1e-3', '-1e-3', '2e-3', '-2e-3', '3e-3', '-3e-3', '5e-3', '7e-3']
slopes = ['5e-3', '7e-3']
layers = ['1','2']
#fields = ['1','2','3']
fields = ['2', '3']
for slope,layer,field in product(slopes,layers,fields):
    compute_acov_diff(slope, layer, field, biased=True, normed=False)