# Compute Eulerian EKE from GeophysicalFlows output

import h5py
import numpy as np
import pandas as pd
from itertools import product


def compute_eulerian_eke(slope,field):
    f = h5py.File("../../Results/Results_GeophysicalFlows/SmallLd/simulation_s"+slope+"_strongmu_field"+field+"_equilibrium.jld2", "r")

    # Time
    t = np.array([float(ti) for ti in f["snapshots"]["t"].keys()])
    idxs = t.argsort()
    t = t[idxs]
    time = t[0::12] # downsample to 12 hrs

    # Read in u, v data
    u = np.array([f["snapshots"]["u"][str(int(ti))][:] for ti in time])
    v = np.array([f["snapshots"]["v"][str(int(ti))][:] for ti in time])

    u1 = u[:,0,:,:]
    v1 = v[:,0,:,:]
    u2 = u[:,1,:,:]
    v2 = v[:,1,:,:]

    eke_u_layer1 = np.mean(u1**2,axis=(1,2))
    eke_v_layer1 = np.mean(v1**2,axis=(1,2))
    eke_u_layer2 = np.mean(u2**2,axis=(1,2))
    eke_v_layer2 = np.mean(v2**2,axis=(1,2))

    df_eke_u_layer1 = pd.DataFrame(eke_u_layer1)
    df_eke_v_layer1 = pd.DataFrame(eke_v_layer1)
    df_eke_u_layer2 = pd.DataFrame(eke_u_layer2)
    df_eke_v_layer2 = pd.DataFrame(eke_v_layer2)

    outname = '../../Results/Results_EKE/SmallLd/eulerian/simulation_s'+slope+'_field'+field
    df_eke_u_layer1.to_csv(outname+'_layer1_eke_u.csv',index=False)
    df_eke_v_layer1.to_csv(outname+'_layer1_eke_v.csv',index=False)
    df_eke_u_layer2.to_csv(outname+'_layer2_eke_u.csv',index=False)
    df_eke_v_layer2.to_csv(outname+'_layer2_eke_v.csv',index=False)


# %% Compute & save EKE

slopes = ['0', '1e-4', '-1e-4', '2e-4', '-2e-4', '3e-4', '-3e-4', '5e-4', '-5e-4', '7e-4', '-7e-4', 
          '1e-3', '-1e-3', '2e-3', '-2e-3', '3e-3', '-3e-3', '5e-3', '7e-3']
fields = ['1', '2', '3']
for field,slope in product(fields,slopes):
    compute_eulerian_eke(slope, field)