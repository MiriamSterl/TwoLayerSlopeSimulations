# Compute kurtosis of q and psi distributions (Eulerian)

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from numpy.fft import irfft2
import scipy.stats as stats
from itertools import product


#%% Conversion functions
def get_qhi(qhi):
    sh = qhi.shape
    qhiZ = np.empty(sh, dtype=np.complex64)
    for i in range(sh[0]):
        for j in range(sh[1]):
            for k in range(sh[2]):
                zijk = qhi[i, j, k]
                qhiZ[i, j, k] = zijk[0] + 1j*zijk[1]

    return qhiZ


#%% Compute q field
def compute_q(slope,field):
    f = h5py.File("../../Results/Results_GeophysicalFlows/SmallLd/simulation_s"+slope+"_strongmu_field"+field+"_equilibrium.jld2", "r")

    # Grid
    grd = f["grid"]
    Lx, Ly, nx, ny, x, y = map(np.array, (grd["Lx"], grd["Ly"], grd["nx"], grd["ny"], grd["x"], grd["y"]))
    Lx, Ly, nx, ny = Lx.flatten()[0], Ly.flatten()[0], nx.flatten()[0], ny.flatten()[0]

    # Wavenumbers
    nl = ny
    nkr = nx//2 + 1
    dk = 1/(Lx/nx)
    kr = np.arange(0, nkr)*2*np.pi/Lx
    l = np.concatenate((kr[:-1], -np.flipud(kr)[:-1]))
    kr = np.reshape(kr, (1, nkr))
    l = np.reshape(l, (nl, 1))
    Krsq = kr**2 + l**2
    invKrsq = 1 / Krsq
    invKrsq[0, 0] = 0

    # Other factors needed
    f0 = np.array(f["params"]["f₀"]).flatten()[0]
    gp = np.array(f["params"]["g′"]).flatten()[0][0]
    H = np.array(f["params"]["H"])
    H1 = H['1'].flatten()[0]
    H2 = H['2'].flatten()[0]
    
    gp = g*(rho2-rho1)/rho2
    f0sqongp = f0**2 / gp
    F1 = f0sqongp/H1
    Delta = Krsq + f0sqongp * (H1 + H2) / (H1 * H2)
    invKrsqonDelta = invKrsq / Delta

    # Time
    t = np.array([float(ti) for ti in f["snapshots"]["t"].keys()])
    idxs = t.argsort()
    t = t[idxs]
    time = t[0::12] # downsample to 12 hrs

    # Read in qh, u, v data
    qh = [f["snapshots"]["qh"][str(int(ti))][:] for ti in time]

    # Get q, psi from qh (Fourier transform)
    qhi = np.zeros(np.shape(qh), dtype=np.complex64)
    for i in range(np.shape(qhi)[0]):
        qhi[i] = get_qhi(qh[i])

    q = irfft2(qhi, axes=(-2, -1))

    return q


#%% Compute kurtosis

def compute_kurtosis(slope,field):
    q = compute_q(slope,field)
    q1 = q[:,0,:,:]
    q2 = q[:,1,:,:]

    time = np.shape(q)[0]
    kurtq1 = np.zeros(time)
    kurtq2 = np.zeros(time)
    for i in range(time):
        kurtq1[i] = stats.kurtosis(q1[i,:,:].flatten())
        kurtq2[i] = stats.kurtosis(q2[i,:,:].flatten())

    df_kurtq1 = pd.DataFrame(kurtq1)
    df_kurtq2 = pd.DataFrame(kurtq2)

    outname = '../../Results/Results_GeophysicalFlows/SmallLd/qpsi/simulation_s'+slope+'_field'+field
    df_kurtq1.to_csv(outname+'_kurtosis_q1.csv',index=False)
    df_kurtq2.to_csv(outname+'_kurtosis_q2.csv',index=False)


#%% Do for all slopes and fields

# slopes = ['0', '1e-4', '-1e-4', '2e-4', '-2e-4', '3e-4', '-3e-4', '5e-4', '-5e-4', '7e-4', '-7e-4', 
#           '1e-3', '-1e-3', '2e-3', '-2e-3', '3e-3', '-3e-3', '5e-3', '-5e-3', '7e-3', '-7e-3']
# fields = ['1','2','3']
slopes = ['5e-3', '-5e-3', '7e-3', '-7e-3']
fields = ['2','3']

# Iterate over slopes and fields
for field,slope in product(fields,slopes):
    compute_kurtosis(slope, field)
