# Compute timeseries of the covariance between the cross-stream velocity and the relative vorticity in the lower layer

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


def streamfunctionfrompv(qh, Krsq, f0sqongp, H1, H2, invKrsqonDelta):
    psih = qh.copy()*np.nan
    q1h, q2h = qh[0, ...], qh[1, ...]
    qq12 = f0sqongp * (q1h / H2 + q2h / H1)
    psih[0, :, :] = - Krsq * q1h - qq12
    psih[1, :, :] = - Krsq * q2h - qq12
    psih[0, :, :] *= invKrsqonDelta
    psih[1, :, :] *= invKrsqonDelta

    return psih


def cov(x,y,lag_len,biased,normed):
    """ 
    compute covariance of two timeseries x and y
    """
    cov = np.empty(lag_len + 1)
    cov[0] = x.dot(y)
    for i in range(lag_len):
        cov[i + 1] = x[i + 1 :].dot(y[: -(i + 1)])
    if biased:
        cov /= len(x)
    else:
        cov /= len(x) - np.arange(lag_len + 1)
    if normed:
        return cov / cov[0]
    else:
        return cov


def cov_v2_zeta2(slope,field,run,biased,normed): 
    """
    compute covariance of cross-stream velocity and relative vorticity timeseries, averaged over all particles
    """
    # Read in particle data
    data = xr.open_zarr('../../Results/Results_Parcels/SmallLd/simulation_s'+slope+'_strongmu_field'+field
                        +'_advection_layer2_run'+run+'_1hr12hr.zarr',decode_timedelta=True)
    lon = data['lon'].values[:,0:800]
    lat = data['lat'].values[:,0:800]
    N = np.shape(lon)[0] # number of particles
    T = np.shape(lon)[1] # number of timesteps

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
    H = np.array(f["params"]["H"])
    H1 = H['1'].flatten()[0]
    H2 = H['2'].flatten()[0]
    gp = np.array(f["params"]["g′"]).flatten()[0][0]
    # if g and ρ are saved, replace the line above by:
    # g = np.array(f["params"]["g"]).flatten()[0]
    # rho = np.array(f["params"]["ρ"])
    # rho1 = rho[0][0][0]
    # rho2 = rho[1][0][0]
    # gp = g*(rho2-rho1)/rho
    
    f0sqongp = f0**2 / gp
    F1 = f0sqongp/H1
    Delta = Krsq + f0sqongp * (H1 + H2) / (H1 * H2)
    invKrsqonDelta = invKrsq / Delta

    # Compute q, psi, zeta2
    qhi = np.zeros(np.shape(qh), dtype=np.complex64)
    psihi = np.zeros(np.shape(qh), dtype=np.complex64)
    for i in range(np.shape(qhi)[0]):
        qhi[i] = get_qhi(qh[i])
        psihi[i] = streamfunctionfrompv(qhi[i], Krsq, f0sqongp, H1, H2, invKrsqonDelta)
    zetahi = - Krsq * psihi
    zeta = irfft2(zetahi, axes=(-2, -1))

    v2 = v[:,1,:,:] # cross-stream velocity in layer 2
    zeta2 = zeta[:,1,:,:] # relative vorticity in layer 2

    # Interpolate v2, zeta2 to particle locations
    v_part2 = np.zeros((N,T))
    zeta_part2 = np.zeros((N,T))
    for i in range(T):
        interp_func_v2 = RegularGridInterpolator((y,x),v2[i,:,:])
        interp_func_zeta2 = RegularGridInterpolator((y,x),zeta2[i,:,:])
        points = np.column_stack((lat[:,i], lon[:,i]))
        v_part2[:,i] = interp_func_v2(points)
        zeta_part2[:,i] = interp_func_zeta2(points)


    # Compute covariance of cross-stream velocity timeseries and relative vorticity timeseries for each particle
    covs = np.zeros(np.shape(v_part2))
    for i in range(N):
        covs[i] = cov(v_part2[i,:],zeta_part2[i,:],T-1,biased,normed)

    # average covariance over all particles
    covs_av = np.mean(covs,axis=0)
    return covs_av


def trapsum(vec,dt):
    """
    Compute the integral of a vector using the trapezoidal rule
    Spacing between vector entries is dt
    """
    trap = 0.5*dt*(vec[:-1] + vec[1:])
    return np.cumsum(trap)
    


def compute_cov_int(slope,field,biased,normed):
    """
    Compute & save timeseries of integral of covariance (averaged over the ensemble)
    """
    # Compute & save autocovariance timeseries for each run of the ensemble
    covs_av = cov_v2_zeta2(slope,field,'0',biased,normed)
    for i in range(1,10):
        covs_av_i = cov_v2_zeta2(slope,field,str(i),biased,normed)
        covs_av = np.vstack((covs_av,covs_av_i))
    df_covs_av = pd.DataFrame(covs_av)
    np_acovs_av = df_covs_av.to_numpy() # TODO change acovs to covs

    # Compute diffusivity by integrating the autocovariance
    dt = 0.5*86400 # timestep of covariance timeseries = 0.5 day (Parcels output)
    integral = np.zeros(np.shape(np_acovs_av))
    for i in range(np.shape(np_acovs_av)[0]):
        integral[i,:] = np.hstack((0,trapsum(np_acovs_av[i,:],dt)))

    df_int = pd.DataFrame(integral)
    if biased:
        add1 = '_biased'
    else:
        add1 = ''
    if normed:
        add2 = '_normed'
    else:
        add2 = ''
    int_fname = '../../Results/Results_Diffusivities/SmallLd/lagrangian/cov_vzeta_s'+slope+'_field'+field+'_1hr12hr'+add1+add2+'.csv'
    df_int.to_csv(int_fname,index=False)



#%% Run calculation for each slope, layer, and field

slopes = ['0', '1e-4', '-1e-4', '2e-4', '-2e-4', '3e-4', '-3e-4', '5e-4', '-5e-4', '7e-4', '-7e-4',
          '1e-3', '-1e-3', '2e-3', '-2e-3', '3e-3', '-3e-3', '5e-3', '7e-3']
fields = ['1','2','3']
for field, slope in product(fields, slopes):
    compute_cov_int(slope, field, biased=True, normed=False)