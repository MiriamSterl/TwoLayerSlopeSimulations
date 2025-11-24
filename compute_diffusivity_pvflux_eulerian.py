# Compute Eulerian PV diffusivity from GeophysicalFlows output

import h5py
import numpy as np
from numpy.fft import irfft2
import pandas as pd
from itertools import product


#%% Simulation parameters
g = 9.81
f0 = 1e-4
rho1 = 1027.6
rho2 = 1028
gp = g*(rho2-rho1)/rho2
U = 0.2

H = {'1':1000, '2':4000} # layer depths
L_Rossby = {layer:np.sqrt(gp*H[layer])/f0 for layer in H.keys()} # Rossby radius per layer for small Ld

# background PV gradient
def dQdy(slope,layer):
    F = 1/L_Rossby[layer]**2

    if layer=='1':
        return F*U
    elif layer=='2':
        return -F*U + f0/H[layer]*float(slope)


#%% Functions to compute diffusivity

def get_qhi(qhi):
    sh = qhi.shape
    qhiZ = np.empty(sh, dtype=np.complex64)
    for i in range(sh[0]):
        for j in range(sh[1]):
            for k in range(sh[2]):
                zijk = qhi[i, j, k]
                qhiZ[i, j, k] = zijk[0] + 1j*zijk[1]

    return qhiZ


def compute_eulerian_diffusivity(slope,field):
    f = h5py.File("../../Results/Results_GeophysicalFlows/SmallLd/simulation_s"+slope+"_strongmu_field"+field+"_equilibrium.jld2", "r")

    # Time
    t = np.array([float(ti) for ti in f["snapshots"]["t"].keys()])
    idxs = t.argsort()
    t = t[idxs]
    time = t[0::12] # downsample to 12 hrs

    # Read in qh, u, v data
    qh = [f["snapshots"]["qh"][str(int(ti))][:] for ti in time]
    v = np.array([f["snapshots"]["v"][str(int(ti))][:] for ti in time])

    # # Get correct time array
    # dt = np.array(f["clock"]["dt"]).flatten()[0]
    # t = t*dt
    # t = t - t[0] # make time start at 0
    # time = t[0::12] # downsample to 12 hrs

    # Get q from qh (Fourier transform)
    qhi = np.zeros(np.shape(qh), dtype=np.complex64)
    for i in range(np.shape(qhi)[0]):
        qhi[i] = get_qhi(qh[i])

    q = irfft2(qhi, axes=(-2, -1))

    flux_layer1 = -1 * np.mean(v[:,0,:,:]*q[:,0,:,:], axis=(1,2))
    flux_layer2 = -1 * np.mean(v[:,1,:,:]*q[:,1,:,:], axis=(1,2))

    diff_layer1 = flux_layer1 / dQdy(slope, '1')
    diff_layer2 = flux_layer2 / dQdy(slope, '2')

    df_flux_layer1 = pd.DataFrame(flux_layer1)
    df_flux_layer2 = pd.DataFrame(flux_layer2)
    df_diff_layer1 = pd.DataFrame(diff_layer1)
    df_diff_layer2 = pd.DataFrame(diff_layer2)

    outname = '../../Results/Results_Diffusivities/SmallLd/eulerian/simulation_s'+slope+'_field'+field
    df_flux_layer1.to_csv(outname+'_layer1_flux.csv', index=False)
    df_flux_layer2.to_csv(outname+'_layer2_flux.csv', index=False)
    df_diff_layer1.to_csv(outname+'_layer1_diff.csv', index=False)
    df_diff_layer2.to_csv(outname+'_layer2_diff.csv', index=False)



# %% Compute & save diffusivities

slopes = ['0', '1e-4', '-1e-4', '2e-4', '-2e-4', '3e-4', '-3e-4', '5e-4', '-5e-4', '7e-4', '-7e-4', 
          '1e-3', '-1e-3', '2e-3', '-2e-3', '3e-3', '-3e-3', '5e-3', '7e-3']
fields = ['1', '2', '3']
for field,slope in product(fields,slopes):
    compute_eulerian_diffusivity(slope, field)