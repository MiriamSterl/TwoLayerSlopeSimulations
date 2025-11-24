"""
Compute energy production term for all simulations
"""

import h5py
import numpy as np
from numpy.fft import irfft2, fft2
from itertools import product
import pandas as pd

#%%
# Conversion/computation functions
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


def compute_prod(psidata,time,nx,ny,k,dk,k2,kplot,U,F):
    psi1 = psidata[time,0,:,:]
    psi2 = psidata[time,1,:,:]
    P = U * F * 1j * k * np.conj(psi1) * psi2 / (nx**2 * ny**2)
    # Symmetry adjustment
    for jk in range(1, nx // 2):
            for ji in range(1, ny // 2):
                P[ji, jk] += P[ji, nx - jk]
    # Extract real part and reduce to upper-left quadrant
    P = 2 * np.real(P[:ny // 2 + 1, :nx // 2 + 1])

    # Find wavenumber magnitude rings/bins
    count = np.zeros(len(kplot))
    Pk = np.zeros(len(kplot))
    K = np.sqrt(k2[0:nx//2+1, 0:ny//2+1])
    for i in range(len(kplot)):
        a = np.where((kplot[i]-0.5*dk <= K) & (K <= kplot[i]+0.5*dk))
        count[i] = np.shape(a)[1]
        Pk[i] = np.sum(P[a])
    return Pk


#%%
def compute_spectrum(slope,field):
    # Read in simulation file
    f = h5py.File("../../Results/Results_GeophysicalFlows/SmallLd/simulation_s"+slope+"_strongmu_field"+field+"_equilibrium.jld2", "r")

    # Grid
    grd = f["grid"]
    Lx, Ly, nx, ny, x, y = map(np.array, (grd["Lx"], grd["Ly"], grd["nx"], grd["ny"], grd["x"], grd["y"]))
    Lx, Ly, nx, ny = Lx.flatten()[0], Ly.flatten()[0], nx.flatten()[0], ny.flatten()[0]

    # Wavenumbers
    nl = ny
    nkr = nx//2 + 1
    dk = 2 * np.pi / Lx
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
    U = np.array(f["params"]["U"]).flatten()[0]
    gp = np.array(f["params"]["g′"]).flatten()[0][0]
    # if g and ρ are saved, replace the line above by:
    # g = np.array(f["params"]["g"]).flatten()[0]
    # rho = np.array(f["params"]["ρ"])
    # rho1 = rho[0][0][0]
    # rho2 = rho[1][0][0]
    # gp = g*(rho2-rho1)/rho

    f0sqongp = f0**2 / gp
    F1 = f0sqongp/H1
    F2 = f0sqongp/H2
    F = F1 + F2
    Delta = Krsq + f0sqongp * (H1 + H2) / (H1 * H2)
    invKrsqonDelta = invKrsq / Delta

    # Time
    t = np.array([float(ti) for ti in f["snapshots"]["t"].keys()])
    idxs = t.argsort()
    t = t[idxs]
    time = t[0::120] # downsample to 5 days

    # Read in qh
    qh = [f["snapshots"]["qh"][str(int(ti))][:] for ti in time]

    # Get q, psi from qh (Fourier transform)
    qhi = np.zeros(np.shape(qh), dtype=np.complex64)
    psihi = np.zeros(np.shape(qh), dtype=np.complex64)
    for i in range(np.shape(qhi)[0]):
        qhi[i] = get_qhi(qh[i])
        psihi[i] = streamfunctionfrompv(qhi[i], Krsq, f0sqongp, H1, H2, invKrsqonDelta)

    psi = irfft2(psihi, axes=(-2, -1))
    psidata = fft2(psi)

    # Compute wavenumber magnitude
    lval = l[:,0]
    kval = lval
    KX, KY = np.meshgrid(kval, lval)
    k2 = KX**2 + KY**2 # wavenumber magnitude squared
    kplot = kr[0] # we will bin around these values of the wavenumber magnitude

    # Compute energy spectra for upper and lower layer for each time step
    Pk = np.zeros((len(time), len(kplot)))
    for i in range(len(time)):
        Pk[i] = compute_prod(psidata, i, nx, ny, kval, dk, k2, kplot, U, F)

    # Average over time
    Pk_av = np.mean(Pk, axis=0)

    # Save results
    df_av = pd.DataFrame(Pk_av)

    fname = '../../Results/Results_GeophysicalFlows/SmallLd/spectra/spectrum_s'+slope+'_field'+field+'_production'
    df_av.to_csv(fname+'.csv', index=False)



#%% Run calculation for each slope and field

slopes = ['0', '1e-4', '-1e-4', '2e-4', '-2e-4', '3e-4', '-3e-4', '5e-4', '-5e-4', '7e-4', '-7e-4', 
          '1e-3', '-1e-3', '2e-3', '-2e-3', '3e-3', '-3e-3', '5e-3', '-5e-3', '7e-3', '-7e-3']
fields = ['1', '2', '3']
for field,slope in product(fields,slopes):
    compute_spectrum(slope, field)
