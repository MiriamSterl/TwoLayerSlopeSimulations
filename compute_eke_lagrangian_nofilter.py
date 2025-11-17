# Compute Lagrangian EKE from Parcels output

import h5py
import numpy as np
import xarray as xr
from itertools import product
import pandas as pd


x = np.load("../../Results/Results_GeophysicalFlows/x.npy")
y = np.load("../../Results/Results_GeophysicalFlows/y.npy")


def compute_EKE(slope,layer,field,run):
    f = h5py.File("../../Results/Results_GeophysicalFlows/SmallLd/simulation_s"+slope+"_strongmu_field"+field+"_equilibrium.jld2", "r")
    U = np.array(f["params"]["U"]).flatten()[0]
    
    data = xr.open_zarr('../../Results/Results_Parcels/SmallLd/simulation_s'+slope+'_strongmu_field'+field
                            +'_advection_layer'+layer+'_run'+run+'_1hr12hr.zarr')
    lon = data['lon'].values[:,0:801]
    lat = data['lat'].values[:,0:801]
    u_tot = data['u'].values
    if layer==1:
        u = u_tot - U # eddy part (leave out mean flow)
    else:
        u = u_tot
    v = data['v'].values

    eke_u = np.mean(u**2,axis=0)
    eke_v = np.mean(v**2,axis=0)
    return eke_u,eke_v


def compute_save_EKE(slope,field):
    eke_u_layer1,eke_v_layer1 = compute_EKE(slope,'1',field,'0')
    eke_u_layer2,eke_v_layer2 = compute_EKE(slope,'2',field,'0')

    for i in np.arange(1,10):
        eke_u_data_layer1,eke_v_data_layer1 = compute_EKE(slope,'1',field,str(i))
        eke_u_layer1 = np.vstack((eke_u_layer1,eke_u_data_layer1))
        eke_v_layer1 = np.vstack((eke_v_layer1,eke_v_data_layer1))
        eke_u_data_layer2,eke_v_data_layer2 = compute_EKE(slope,'2',field,str(i))
        eke_u_layer2 = np.vstack((eke_u_layer2,eke_u_data_layer2))
        eke_v_layer2 = np.vstack((eke_v_layer2,eke_v_data_layer2))


    df_eke_u_layer1 = pd.DataFrame(eke_u_layer1).iloc[:,0:801]
    df_eke_v_layer1 = pd.DataFrame(eke_v_layer1).iloc[:,0:801]
    df_eke_u_layer2 = pd.DataFrame(eke_u_layer2).iloc[:,0:801]
    df_eke_v_layer2 = pd.DataFrame(eke_v_layer2).iloc[:,0:801]

    fn_base = 'simulation_s'+slope+'_strongmu_field'+field
    df_eke_u_layer1.to_csv("../../Results/Results_EKE/SmallLd/lagrangian/"+fn_base+"_eke_u_layer1_1hr12hr_nofilter.csv",index=False)
    df_eke_v_layer1.to_csv("../../Results/Results_EKE/SmallLd/lagrangian/"+fn_base+"_eke_v_layer1_1hr12hr_nofilter.csv",index=False)
    df_eke_u_layer2.to_csv("../../Results/Results_EKE/SmallLd/lagrangian/"+fn_base+"_eke_u_layer2_1hr12hr_nofilter.csv",index=False)
    df_eke_v_layer2.to_csv("../../Results/Results_EKE/SmallLd/lagrangian/"+fn_base+"_eke_v_layer2_1hr12hr_nofilter.csv",index=False)


#%% Compute & save EKE
# slopes = ['0', '1e-4', '-1e-4', '2e-4', '-2e-4', '3e-4', '-3e-4', '5e-4', '-5e-4', '7e-4', '-7e-4',
#           '1e-3', '-1e-3', '2e-3', '-2e-3', '3e-3', '-3e-3', '5e-3', '7e-3', '-5e-3', '-7e-3']
slopes = ['5e-3', '-5e-3', '7e-3', '-7e-3']
for slope in slopes:
    #compute_save_EKE(slope, '1')
    compute_save_EKE(slope, '2')
    compute_save_EKE(slope, '3')