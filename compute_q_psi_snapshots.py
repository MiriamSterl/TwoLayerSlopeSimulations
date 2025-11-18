"""
Save q and psi snapshots
"""

import h5py
import numpy as np
from numpy.fft import irfft2
import sys

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

def streamfunctionfrompv(qh, Krsq, f0sqongp, H1, H2, invKrsqonDelta):
    psih = qh.copy()*np.nan
    q1h, q2h = qh[0, ...], qh[1, ...]
    qq12 = f0sqongp * (q1h / H2 + q2h / H1)
    psih[0, :, :] = - Krsq * q1h - qq12
    psih[1, :, :] = - Krsq * q2h - qq12
    psih[0, :, :] *= invKrsqonDelta
    psih[1, :, :] *= invKrsqonDelta

    return psih


#%% Compute & save snapshots of q, psi fields

def save_snapshots(slope,field):
    fn_base = "simulation_s"+slope+"_strongmu_field"+field
    f = h5py.File("../../Results/Results_GeophysicalFlows/SmallLd/"+fn_base+"_equilibrium.jld2", "r")

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
    
    f0sqongp = f0**2 / gp
    F1 = f0sqongp/H1
    Delta = Krsq + f0sqongp * (H1 + H2) / (H1 * H2)
    invKrsqonDelta = invKrsq / Delta

    # Time - final snapshot
    t = np.array([float(ti) for ti in f["snapshots"]["t"].keys()])
    idxs = t.argsort()
    t = t[idxs]
    qh = np.array(f["snapshots"]["qh"][str(int(t[-1]))][:]) # final snapshot of qh

    # Get q and psi fields
    qhi = get_qhi(qh)
    q = irfft2(qhi, axes=(-2, -1))

    psihi = streamfunctionfrompv(qhi, Krsq, f0sqongp, H1, H2, invKrsqonDelta)
    psi = irfft2(psihi, axes=(-2, -1))

    np.save("../Results/Results_GeophysicalFlows/x.npy",x)
    np.save("../Results/Results_GeophysicalFlows/y.npy",y)

    np.save("../../Results/Results_GeophysicalFlows/SmallLd/"+fn_base+"_snapshot_q.npy",q)
    np.save("../../Results/Results_GeophysicalFlows/SmallLd/"+fn_base+"_snapshot_psi.npy",psi)


#%% Save snapshots for each slope and field

# slopes = ['0', '1e-4', '-1e-4', '2e-4', '-2e-4', '3e-4', '-3e-4', '5e-4', '-5e-4', '7e-4', '-7e-4', 
#           '1e-3', '-1e-3', '2e-3', '-2e-3', '3e-3', '-3e-3', '5e-3', '-5e-3', '7e-3', '-7e-3']
# fields = ['1', '2', '3']
slopes = ['5e-3', '-5e-3', '7e-3', '-7e-3']
fields = ['2', '3']
for field,slope in product(fields,slopes):
    save_snapshots(slope, field)


