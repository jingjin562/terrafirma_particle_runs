#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:23:58 2026

@author: jingjin
"""
import sys
sys.path.append('/home/jingjin/work/postpro/')
sys.path.append('/home/jingjin/work/ocean_parcels/')

import numpy as np
import xarray as xr
from particles_save_to_netcdf import create_netcdf


def pm180_lon(lon):
    lon = np.where(lon>180, lon-360, lon)
    lon = np.where(lon<-180, lon+360, lon)
    return lon

def distance(lon, lat):
    pdx = np.diff(lon, axis=1, prepend=0)
    pdy = np.diff(lat, axis=1, prepend=0)
    # approximation of the distance travelled
    return np.sqrt(np.square(pdx) + np.square(pdy)) > 0

def remove_stuck_particles(variable, nonstuck):
    return np.where(nonstuck, variable, np.nan)


#%% ------ read raw data ------
output_file = 'MPI_test/FRIS_cavity_2080-2230_diffu_0.zarr'
pfile = xr.open_zarr(f"/home/jingjin/work/ocean_parcels/{output_file}", decode_cf=True)
#pfile = xr.open_dataset(f"/home/jingjin/work/ocean_parcels/{output_file}")
lon = np.ma.filled(pfile.variables["lon"].values, np.nan)
lat = np.ma.filled(pfile.variables["lat"].values, np.nan)
time = np.ma.filled(pfile.variables["time"].values, np.nan)
z = np.ma.filled(pfile.variables["z"].values, np.nan)

#%% ------ pre-process stuck particles ------
lon = pm180_lon(lon)
nonstuck = distance(lon, lat)
lon_nonstuck = remove_stuck_particles(lon, nonstuck)
lat_nonstuck = remove_stuck_particles(lat, nonstuck)
z_nonstuck = remove_stuck_particles(z, nonstuck)

#%% ------ save to netcdf ------
path = '/home/jingjin/work/ocean_parcels/MPI_test/'
filename = 'FRIS_cavity_cx209_2080-2230_diffu_0_processed.nc'
create_netcdf(path, filename, 'lon', lon_nonstuck)
create_netcdf(path, filename, 'lat', lat_nonstuck)
create_netcdf(path, filename, 'depth', z_nonstuck)
create_netcdf(path, filename, 'time', time)