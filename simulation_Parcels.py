"""
Particle advection
GeophysicalFlows output saved with resolution of 1 hour
"""

from parcels import FieldSet, ParticleSet, Variable, JITParticle
import numpy as np
import xarray as xr
import pandas as pd
import h5py
from datetime import timedelta
from operator import attrgetter
import sys

slope = sys.argv[1]
field = sys.argv[2]
fn_base = "simulation_s"+slope+"_strongmu_field"+field
rundays = int(sys.argv[3]) # number of days for which we want to track the particles


# =============================================================================
# Loading the flow fields
# =============================================================================

f = h5py.File("../../Results/Results_GeophysicalFlows/SmallLd/"+fn_base+"_equilibrium.jld2", "r")

# Background U
U = np.array(f["params"]["U"]).flatten()[0]

# Grid
grd = f["grid"]
Lx, Ly, nx, ny, x, y = map(np.array, (grd["Lx"], grd["Ly"], grd["nx"], grd["ny"], grd["x"], grd["y"]))
Lx, Ly, nx, ny = Lx.flatten()[0], Ly.flatten()[0], nx.flatten()[0], ny.flatten()[0]

# Time & velocities
dt = np.array(f["clock"]["dt"]).flatten()[0]
t = np.array([float(ti) for ti in f["snapshots"]["t"].keys()])
idxs = t.argsort()
t = t[idxs]

u = np.array([f["snapshots"]["u"][str(int(ti))][:] for ti in t])
u1 = u[:,0,:,:] + U # upper layer: add mean flow
u2 = u[:,1,:,:] # lower layer
v = np.array([f["snapshots"]["v"][str(int(ti))][:] for ti in t])
v1 = v[:,0,:,:]
v2 = v[:,1,:,:]

t = t*dt
t = t - t[0] # make time start at 0
timestep = t[1] # the time step of the flow field in s

npart = 21 # total (npart-1)^2 particles
particles_start_x = np.linspace(-Lx/2,Lx/2,npart)[:-1]
particles_start_y = np.linspace(-Ly/2,Ly/2,npart)[:-1]
[start_x,start_y] = np.meshgrid(particles_start_x,particles_start_y)


#%%
# =============================================================================
# Defining particle class and kernels
# =============================================================================

# Particle class that has velocities and displacements as variables
class ParticleWithUVD(JITParticle):
    u = Variable('u', initial=0)
    v = Variable('v', initial=0)
    dx = Variable('dx', initial=0) # x-displacement from original location
    x_prev = Variable('x_prev', to_write=False,initial=attrgetter('lon')) # previous x-location
    dy = Variable('dy', initial=0) # y-displacement from original location
    y_prev = Variable('y_prev', to_write=False,initial=attrgetter('lat')) # previous y-location
    
    
# Advection kernel that also checks periodic boundary conditions
def AdvectionRK4_periodic(particle, fieldset, time):
    (u1, v1) = fieldset.UV[particle]
    lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)
    if lon1 < fieldset.halo_west:
        lon1 += fieldset.halo_east - fieldset.halo_west
    elif lon1 > fieldset.halo_east:
        lon1 -= fieldset.halo_east - fieldset.halo_west
    if lat1 < fieldset.halo_south:
        lat1 += fieldset.halo_north - fieldset.halo_south
    elif lat1 > fieldset.halo_north:
        lat1 -= fieldset.halo_north - fieldset.halo_south    
    
    (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1, particle]
    lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)
    if lon2 < fieldset.halo_west:
        lon2 += fieldset.halo_east - fieldset.halo_west
    elif lon2 > fieldset.halo_east:
        lon2 -= fieldset.halo_east - fieldset.halo_west
    if lat2 < fieldset.halo_south:
        lat2 += fieldset.halo_north - fieldset.halo_south
    elif lat2 > fieldset.halo_north:
        lat2 -= fieldset.halo_north - fieldset.halo_south
    
    (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2, particle]
    lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)
    if lon3 < fieldset.halo_west:
        lon3 += fieldset.halo_east - fieldset.halo_west
    elif lon3 > fieldset.halo_east:
        lon3 -= fieldset.halo_east - fieldset.halo_west
    if lat3 < fieldset.halo_south:
        lat3 += fieldset.halo_north - fieldset.halo_south
    elif lat3 > fieldset.halo_north:
        lat3 -= fieldset.halo_north - fieldset.halo_south
     
    (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3, particle]
    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
    if particle.lon < fieldset.halo_west:
        particle.lon += fieldset.halo_east - fieldset.halo_west
    elif particle.lon > fieldset.halo_east:
        particle.lon -= fieldset.halo_east - fieldset.halo_west
    if particle.lat < fieldset.halo_south:
        particle.lat += fieldset.halo_north - fieldset.halo_south
    elif particle.lat > fieldset.halo_north:
        particle.lat -= fieldset.halo_north - fieldset.halo_south

        
# Kernel to sample velocities
def SampleUV(particle, fieldset, time):
    particle.u, particle.v = fieldset.UV[particle]

    
# Kernel to compute cross-stream displacement
def Displacement(particle, fieldset, time):
    particle.dy += particle.lat - particle.y_prev
    if particle.lat - particle.y_prev < -1e5:
        particle.dy += 7e5
    elif particle.lat - particle.y_prev > 1e5:
        particle.dy -= 7e5
    particle.y_prev = particle.lat
    
    particle.dx += particle.lon - particle.x_prev
    if particle.lon - particle.x_prev < -1e5:
        particle.dx += 7e5
    elif particle.lon - particle.x_prev > 1e5:
        particle.dx -= 7e5
    particle.x_prev = particle.lon
    
    

# =============================================================================
# Creating different FieldSets
# =============================================================================

def fieldsetPerLayer(layer,start):
    """
    i = layer number (1 = upper layer, 2 = lower layer)
    """
    step = int(rundays*86400/timestep) # select data for time duration = rundays
    if layer==1:
        data = {'U': u1[start:start+step,:,:], 'V': v1[start:start+step,:,:]}
    else:
        data = {'U': u2[start:start+step,:,:], 'V': v2[start:start+step,:,:]}
    dimensions = {'time': t[start:start+step], 'lon': x, 'lat': y}
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat',allow_time_extrapolation=True)
    
    fieldset.add_constant('halo_west', fieldset.U.grid.lon[0])
    fieldset.add_constant('halo_east', fieldset.U.grid.lon[-1])
    fieldset.add_constant('halo_south', fieldset.U.grid.lat[0])
    fieldset.add_constant('halo_north', fieldset.U.grid.lat[-1])
    fieldset.add_periodic_halo(zonal=True,meridional=True)
    
    return fieldset



# =============================================================================
# Advecting the particles
# =============================================================================

def advectParticles(i):
    start = int(30*i*86400/timestep) # release particles every 30 days
    fieldset_1 = fieldsetPerLayer(1,start)
    fieldset_2 = fieldsetPerLayer(2,start)
    
    
    pset_1 = ParticleSet(fieldset_1,pclass=ParticleWithUVD,lon=start_x,lat=start_y)
    pset_1.execute(pset_1.Kernel(SampleUV), dt=0) # recording initial velocities of the particles
    outname_1 = "../../Results/Results_Parcels/SmallLd/"+fn_base+"_advection_layer1_run"+str(i)+"_1hr12hr.zarr"
    output_file_1 = pset_1.ParticleFile(name=outname_1,outputdt=timedelta(hours=12), chunks=(len(pset_1), 10))
    pset_1.execute(pset_1.Kernel(AdvectionRK4_periodic)+pset_1.Kernel(Displacement)+pset_1.Kernel(SampleUV),
                    runtime=timedelta(days=rundays),
                    dt=timedelta(hours=1),
                    output_file=output_file_1)
    output_file_1.close()
    
    
    pset_2 = ParticleSet(fieldset_2,pclass=ParticleWithUVD,lon=start_x,lat=start_y)
    pset_2.execute(pset_2.Kernel(SampleUV), dt=0)
    outname_2 = "../../Results/Results_Parcels/SmallLd/"+fn_base+"_advection_layer2_run"+str(i)+"_1hr12hr.zarr"
    output_file_2 = pset_2.ParticleFile(name=outname_2,outputdt=timedelta(hours=12), chunks=(len(pset_2), 10))
    pset_2.execute(pset_2.Kernel(AdvectionRK4_periodic)+pset_2.Kernel(Displacement)+pset_2.Kernel(SampleUV),
                    runtime=timedelta(days=rundays),
                    dt=timedelta(hours=1),
                    output_file=output_file_2)
    output_file_2.close()
    
    
# =============================================================================
# And... Go!
# =============================================================================

for i in range(10):
    advectParticles(i)

