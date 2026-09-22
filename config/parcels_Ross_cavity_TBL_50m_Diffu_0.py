#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 12:13:52 2025

@author: jingjin
"""
import sys
sys.path.append('/home/jingjin/work/ocean_parcels/')

import warnings
from datetime import timedelta
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from operator import attrgetter

import parcels
from parcels import FileWarning
from parcels import StatusCode
from parcels import FieldSet, Field, NestedField, VectorField, ParticleFile, ParticleSet, JITParticle, Variable, ParcelsRandom
from parcels import AdvectionRK4_3D, DiffusionUniformKh

from retrive_cavity_loations import Read_Ross_Init_Loc, Ross_cavity_locations, FRIS_cavity_locations

def files_UVW_selections(suite_id, dataset_folder, gtype, year_to_run, Start_Year):
    if suite_id == 'cy838':
        file_start_year = 2200
    elif (suite_id == 'cx209') | (suite_id == 'cz826'):
        file_start_year = 2000
        
    return sorted(glob(f"{dataset_folder}/grid-{gtype}/nemo_{suite_id}o_1m_*{gtype}.nc"))[(Start_Year-file_start_year)*12:(Start_Year-file_start_year)*12+year_to_run*12]

def files_tracer_selections(suite_id, dataset_folder, variable, year_to_run, Start_Year):
    if suite_id == 'cy838':
        file_start_year = 2200
    elif (suite_id == 'cx209') | (suite_id == 'cz826'):
        file_start_year = 2000
        
    return sorted(glob(f"{dataset_folder}/{variable}/nemo_{suite_id}o_1m_*T.nc"))[(Start_Year-file_start_year)*12:(Start_Year-file_start_year)*12+year_to_run*12]
    
class TSParticle(JITParticle):
     prev_lon = Variable('prev_lon', dtype=np.float32, to_write=False, initial=attrgetter('lon'))
     prev_lat = Variable('prev_lat', dtype=np.float32, to_write=False, initial=attrgetter('lat'))
     prev_depth = Variable('prev_depth', dtype=np.float32, to_write=False, initial=attrgetter('depth'))
     prev_time = Variable('prev_time', dtype=np.float32, to_write=False, initial=attrgetter('time'))
     rec = Variable('rec', dtype=int, to_write=False, initial=0)
     dep = Variable('dep', dtype=np.float32, to_write=False, initial=0)
     age = Variable("age", dtype=np.float32, initial=0.0)
     temp = Variable("temp", dtype=np.float32, initial=attrgetter('T'))
     sal = Variable("sal", dtype=np.float32, initial=attrgetter('S'))
     uvel = Variable("uvel", dtype=np.float32, initial=attrgetter('U'))
     vvel = Variable("vvel", dtype=np.float32, initial=attrgetter('V'))
     wvel = Variable("wvel", dtype=np.float32, initial=attrgetter('W'))

def CheckOutOfBounds(particle, fieldset, time):
    if particle.state == StatusCode.ErrorOutOfBounds:
        particle.delete()

def KeepInOcean(particle, fieldset, time):
    if particle.state == StatusCode.ErrorThroughSurface:
       # particle.delete()
        particle_ddepth += 1.*ParcelsRandom.uniform(0, 1.)
        particle.state = StatusCode.Success   
        
def RecoveryParticle(particle, fieldset, time):
    if particle.state == StatusCode.ErrorOutOfBounds:
       particle_dlat += particle.prev_lat - particle.lat + 0.05*ParcelsRandom.uniform(-1., 1.)
       particle_dlon += particle.prev_lon -particle.lon + 0.1*ParcelsRandom.uniform(-1., 1.)
       particle_ddepth += particle.prev_depth - particle.depth + 1.*ParcelsRandom.uniform(-1., 1.)
       #particle_dlat += 0.5*ParcelsRandom.uniform(0., 1.)
       #particle_dlon += 1*ParcelsRandom.uniform(-1., -0.5)
       #particle_ddepth += 20**ParcelsRandom.uniform(0., 1.)       
       
       if particle.dep == 0 :
         particle.rec += 1
         particle.state = StatusCode.Success
       else :
         particle.rec = 0
         particle.state = StatusCode.Success
       if particle.rec == 100 :
         particle.delete()
         
def SampleAge(particle, fieldset, time):  # pragma: no cover
    particle.age += 6*particle.dt
    #if particle.age > fieldset.maxage:
    #    particle.delete()
        
def SampleT(particle, fieldset, time):
    particle.temp = fieldset.T[time, particle.depth, particle.lat, particle.lon]
    
def SampleS(particle, fieldset, time):
    particle.sal = fieldset.S[time, particle.depth, particle.lat, particle.lon]

def SampleVel(particle, fieldset, time):
    particle.uvel, particle.vvel, particle.wvel = fieldset.UVW[particle]
    
def Timestamps_Manual(year_to_run, Start_Year=2000):
    import xarray as xr 
    
    timestamps = []
    timestamps = np.expand_dims(
                np.array([np.datetime64(f"{y:04d}-{m:02d}-16") for y in range(Start_Year, Start_Year+year_to_run) for m in range(1, 13)]),
                axis=1
                )
    return timestamps

def cftime_to_datetime(cftimearray):
    from datetime import datetime
    
    return datetime(cftimearray.year, 
             cftimearray.month, 
             cftimearray.day, 
             cftimearray.hour, 
             cftimearray.minute, 
             cftimearray.second)

def release_times(ufiles, nemo_dimensions):
    import xarray as xr    
    
    dt = np.array([])
    for t in range(0, len(ufiles), 6):
        cftimearray = xr.open_dataset(ufiles[t]).variables[nemo_dimensions['time']].values[0]
        dt = np.append(dt, cftime_to_datetime(cftimearray))
        
    return dt

def time_varying_release(ufiles, nemo_dimensions):

    def add_new_release(release_total, release_new):
        release_total = np.append(release_total, release_new)
        return release_total
    
    lon_release = np.array([])
    lat_release = np.array([])
    depth_release = np.array([])
    time_release = np.array([])
    
    time_stamp = release_times(ufiles, nemo_dimensions)
    for times in time_stamp:
        lon_tmp, lat_tmp, depth_tmp = Ross_cavity_locations(suite_id, times.year, TBL_THK=50)
        
        lon_release = add_new_release(lon_release, lon_tmp)
        lat_release = add_new_release(lat_release, lat_tmp)
        depth_release = add_new_release(depth_release, depth_tmp)
        time_release = add_new_release(time_release, np.repeat(times, len(lon_tmp)))

    return lon_release, lat_release, depth_release, time_release

# ---------- diffusion --------------------
def set_diffusion(fieldset, diffusivity, dres=0.01):
    fname = '/projects/oceanparcels/cx209/eORCA1.2_mesh_mask.nc'
    dimensions = {'lon': 'glamt', 'lat': 'gphit'}
    meshSize_zonal = Field.from_netcdf(fname, 'e1u', dimensions, mesh='spherical', interp_method='cgrid_velocity')    
    fieldset.add_field(meshSize_zonal)
    meshSize_meridional = Field.from_netcdf(fname, 'e2u', dimensions, mesh='spherical', interp_method='cgrid_velocity')    
    fieldset.add_field(meshSize_meridional)
    fieldset.add_field(Field('Kh_zonal', data=diffusivity*np.ones(meshSize_zonal.data.shape),
                             grid=meshSize_zonal.grid, mesh='spherical'))
    fieldset.add_field(Field('Kh_meridional', data=diffusivity*np.ones(meshSize_meridional.data.shape),
                             grid=meshSize_meridional.grid, mesh='spherical'))
    
    fieldset.add_constant("dres", dres)
# -----------------------------------------
        
# Add a filter for the xarray decoding warning
#warnings.simplefilter("ignore", FileWarning)

suite_id = 'cz826'
year_to_run = 150
Start_Year = 2000 # --- default: 2000
diffusivity = 0 # --- 0 turn off, in units of m^2/s

mesh_mask_folder = '/projects/oceanparcels/cx209'
dataset_folder = f'/projects/oceanparcels/{suite_id}'
ufiles = files_UVW_selections(suite_id, dataset_folder, 'U', year_to_run, Start_Year)
vfiles = files_UVW_selections(suite_id, dataset_folder, 'V', year_to_run, Start_Year)
wfiles = files_UVW_selections(suite_id, dataset_folder, 'W', year_to_run, Start_Year)
tfiles = files_tracer_selections(suite_id, dataset_folder, 'thetao', year_to_run, Start_Year)
sfiles = files_tracer_selections(suite_id, dataset_folder, 'so', year_to_run, Start_Year)
#mldfiles = files_tracer_selections(dataset_folder, 'mlotst', year_to_run, Start_Year)
mesh_mask = f"{mesh_mask_folder}/eORCA1.2_mesh_mask.nc"

#ice_folder = '/home/jingjin/data/terrafirma/cx209/bathymetry-isf'
#icefiles = sorted(glob(f'{ice_folder}/*bathymetry-isf.nc'))[2000-1850:]

filenames = {
    "U": {"lon": mesh_mask, "lat": mesh_mask, "depth": wfiles[0], "data": ufiles},
    "V": {"lon": mesh_mask, "lat": mesh_mask, "depth": wfiles[0], "data": vfiles},
    "W": {"lon": mesh_mask, "lat": mesh_mask, "depth": wfiles[0], "data": wfiles},
    "T": {"lon": mesh_mask, "lat": mesh_mask, "depth": tfiles[0], "data": tfiles}, 
    "S": {"lon": mesh_mask, "lat": mesh_mask, "depth": sfiles[0], "data": sfiles},
   # "MLD": {"lon": mesh_mask, "lat": mesh_mask, "data": mldfiles},
}

variables = {
    "U": "uo",
    "V": "vo",
    "W": "wo",
    "T": "thetao",
    "S": "so",
   # "MLD": "mlotst",
}

# Note that all variables need the same dimensions in a C-Grid
nemo_vel_dimensions = {
    "lon": "glamf",
    "lat": "gphif",
    "depth": "depthw",
    "time": "time_centered",
}

nemo_tracer_dimensions = {
    "lon": "glamf",
    "lat": "gphif",
    "depth": "deptht",
    "time": "time_centered",
}

dimensions = {
    "U": nemo_vel_dimensions,
    "V": nemo_vel_dimensions,
    "W": nemo_vel_dimensions,
    "T": nemo_tracer_dimensions,
    "S": nemo_tracer_dimensions,
  #  "MLD": {"lon": "glamf", "lat": "gphif", "time": "time_centered"},
}

# Valid options are ('linear', 'nearest', 'freeslip', 'partialslip', 'bgrid_velocity', 'bgrid_w_velocity', 'cgrid_velocity', 'linear_invdist_land_tracer', 'bgrid_tracer', 'cgrid_tracer')
with warnings.catch_warnings():
    warnings.simplefilter("ignore", parcels.FileWarning)

    fieldset = parcels.FieldSet.from_netcdf(
        filenames, 
        variables, 
        dimensions, 
        mesh='spherical',
        timestamps=Timestamps_Manual(year_to_run, Start_Year),
        allow_time_extrapolation=True,
        interp_method={
            "U": "cgrid_velocity",
            "V": "cgrid_velocity",
            "W": "cgrid_velocity",
            "T": "cgrid_tracer",
            "S": "cgrid_tracer",
        }, 
        gridindexingtype='nemo')

#landmask = make_landmask(ufiles[0])
#fieldset.add_field(Field("landmask", data=landmask, lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat, depth=fieldset.U.grid.depth, mesh="spherical", interp_method='nearest'))
#bathy = read_bathy(icefiles[0])
#fieldset.add_field(Field("age", data=bathy, lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat, mesh="spherical", interp_method='nearest'))

if diffusivity > 0:
    set_diffusion(fieldset, diffusivity)
    
lon_release, lat_release, depth_release, time_release = time_varying_release(ufiles, nemo_vel_dimensions)

fi_out = f"/home/jingjin/work/ocean_parcels/MPI_test/Ross_cavity_{suite_id}_{Start_Year}-{Start_Year+year_to_run}_diffu_{diffusivity}.zarr"

pset = parcels.ParticleSet.from_list(
    fieldset=fieldset,
    pclass=TSParticle,
    lon=lon_release,
    lat=lat_release,
    depth=depth_release,
    time=time_release,
)

output_file = pset.ParticleFile(
    name=fi_out, 
    outputdt=timedelta(days=30),
    chunks=(len(pset), year_to_run*12),
)

if diffusivity > 0:
    kernels = [AdvectionRK4_3D, DiffusionUniformKh, SampleAge, SampleT, SampleS, SampleVel, RecoveryParticle, KeepInOcean]
else:
    kernels = [AdvectionRK4_3D, SampleAge, SampleT, SampleS, SampleVel, RecoveryParticle, KeepInOcean]

pset.execute(kernels,
             runtime=timedelta(days=360*year_to_run),
             dt=timedelta(days=5),
             output_file=output_file,
             verbose_progress=True,
             )
