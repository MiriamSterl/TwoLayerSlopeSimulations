import h5py
import numpy as np
from numpy.fft import irfft2
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from itertools import product

x = np.load("../../Results/Results_GeophysicalFlows/x.npy")
y = np.load("../../Results/Results_GeophysicalFlows/y.npy")


#%%
def get_qhi(qhi):
    sh = qhi.shape
    qhiZ = np.empty(sh, dtype=np.complex64)
    for i in range(sh[0]):
        for j in range(sh[1]):
            for k in range(sh[2]):
                zijk = qhi[i, j, k]
                qhiZ[i, j, k] = zijk[0] + 1j*zijk[1]

    return qhiZ


#%%
def compute_fluxes(slope,field,run):
    # Read in particle data
    data1 = xr.open_zarr('../../Results/Results_Parcels/SmallLd/simulation_s'+slope+'_strongmu_field'+field
                        +'_advection_layer1_run'+run+'_1hr12hr.zarr',decode_timedelta=True)
    lon1 = data1['lon'].values[:,0:800]
    lat1 = data1['lat'].values[:,0:800]

    data2 = xr.open_zarr('../../Results/Results_Parcels/SmallLd/simulation_s'+slope+'_strongmu_field'+field
                        +'_advection_layer2_run'+run+'_1hr12hr.zarr',decode_timedelta=True)
    t_part = (data2['time'].values[0,0:800] / np.timedelta64(1,'s')).astype(int)
    lon2 = data2['lon'].values[:,0:800]
    lat2 = data2['lat'].values[:,0:800]

    N = np.shape(lon2)[0] # number of particles
    T = np.shape(lon2)[1] # number of timesteps

    # Read in flow field data
    f = h5py.File("../../Results/Results_GeophysicalFlows/SmallLd/simulation_s"+slope+"_strongmu_field"+field+"_equilibrium.jld2", "r")
    t = np.array([float(ti) for ti in f["snapshots"]["t"].keys()])
    dt = np.array(f["clock"]["dt"]).flatten()[0]
    idxs = t.argsort()
    t = t[idxs]

    # Flow field data corresponding to particle release
    start = int(int(run)*30*24) # particle run i starts after 30*i days
    step = int(400*24) # particles are advected for 400 days
    t_gf = t[start:start+step:12] # every 12 hours
    v = np.array([f["snapshots"]["v"][str(int(ti))][:] for ti in t_gf])
    qh = [f["snapshots"]["qh"][str(int(ti))][:] for ti in t_gf]
    # Get q from qh (Fourier transform)
    qhi = np.zeros(np.shape(qh), dtype=np.complex64)
    for i in range(np.shape(qhi)[0]):
        qhi[i] = get_qhi(qh[i])
    q = irfft2(qhi, axes=(-2, -1))

    # interpolate q,v to particle locations
    q_part1 = np.zeros((N,T))
    q_part2 = np.zeros((N,T))
    v_part1 = np.zeros((N,T))
    v_part2 = np.zeros((N,T))
    for i in range(T):
        q1 = q[i,0,:,:]
        q2 = q[i,1,:,:]
        v1 = v[i,0,:,:]
        v2 = v[i,1,:,:]
        interp_func_q1 = RegularGridInterpolator((y,x),q1)
        interp_func_q2 = RegularGridInterpolator((y,x),q2)
        interp_func_v1 = RegularGridInterpolator((y,x),v1)
        interp_func_v2 = RegularGridInterpolator((y,x),v2)
        points1 = np.column_stack((lat1[:,i], lon1[:,i]))
        points2 = np.column_stack((lat2[:,i], lon2[:,i]))
        q_part1[:,i] = interp_func_q1(points1)
        q_part2[:,i] = interp_func_q2(points2)
        v_part1[:,i] = interp_func_v1(points1)
        v_part2[:,i] = interp_func_v2(points2)

    # Compute fluxes
    flux_layer1 = -1 * np.mean(v_part1*q_part1, axis=0) # mean over particles
    flux_layer2 = -1 * np.mean(v_part2*q_part2, axis=0) # mean over particles

    return flux_layer1, flux_layer2


def compute_and_save_fluxes(slope,field):
    """
    For all particle runs per slope/field combination, compute and save fluxes
    """
    flux1, flux2 = compute_fluxes(slope,field,'0')
    for i in range(1,10):
        flux1_i, flux2_i = compute_fluxes(slope,field,str(i))
        flux1 = np.vstack((flux1,flux1_i))
        flux2 = np.vstack((flux2,flux2_i))
    flux_layer1 = pd.DataFrame(flux1)
    flux_layer2 = pd.DataFrame(flux2)

    # Save output
    df_flux_layer1 = pd.DataFrame(flux_layer1)
    df_flux_layer2 = pd.DataFrame(flux_layer2)

    outname = '../../Results/Results_Diffusivities/SmallLd/lagrangian/simulation_s'+slope+'_field'+field+'_1hr12hr'
    df_flux_layer1.to_csv(outname+'_layer1_flux.csv', index=False)
    df_flux_layer2.to_csv(outname+'_layer2_flux.csv', index=False)


# %% Compute & save fluxes

slopes = ['0', '1e-4', '-1e-4', '2e-4', '-2e-4', '3e-4', '-3e-4', '5e-4', '-5e-4', '7e-4', '-7e-4', 
          '1e-3', '-1e-3', '2e-3', '-2e-3', '3e-3', '-3e-3', '5e-3', '-5e-3', '7e-3', '-7e-3']
fields = ['1','2','3']

for field, slope in product(fields, slopes):
    compute_and_save_fluxes(slope, field)