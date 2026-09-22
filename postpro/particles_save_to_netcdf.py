#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 16:19:26 2025

@author: jingjin
"""

def create_dimensions_time(ncfile, dim_t=None):
    # ---- requested input arrgument : ncfile = nc.Dataset(filename, 'a/w', 'NETCDF4')
    # Check if 'time_counter' dimension already exists
    if 'time_counter' not in ncfile.dimensions:
        ncfile.createDimension('time_counter', dim_t)   

def create_dimensions_number(ncfile, dim_n):
    # ---- requested input arrgument : ncfile = nc.Dataset(filename, 'a/w', 'NETCDF4')
    # Check if 'time_counter' dimension already exists
    if 'number' not in ncfile.dimensions:
        ncfile.createDimension('number', dim_n)
        
def create_variables_particles(ncfile, varname): # --- 2D spatial (lat vs. lon) type data
    if varname not in ncfile.variables:
        data_var = ncfile.createVariable(varname, 'float64', ('number', 'time_counter'))
    else:
        data_var = ncfile.variables[varname]

    return data_var

def create_particles_release(ncfile, varname): # --- 1D data
    if varname not in ncfile.variables:
        data_var = ncfile.createVariable(varname, 'float64', ('number'))
    else:
        data_var = ncfile.variables[varname]

    return data_var

def create_netcdf(path, filename, varname, var):
    import netCDF4 as nc
    import os
    import os.path as op
    
    fi_out = op.join(path, filename)
    ncfile = nc.Dataset(fi_out, 'a', 'NETCDF4')
    
    dim_n = var.shape[0]
    create_dimensions_number(ncfile, dim_n)
    create_dimensions_time(ncfile, dim_t=None)
    
    data_var = create_variables_particles(ncfile, varname)
    data_var[:] = var
    
    ncfile.close()

def create_init_locs_netcdf(path, filename, varname, var):
    import netCDF4 as nc
    import os
    import os.path as op
    
    fi_out = op.join(path, filename)
    ncfile = nc.Dataset(fi_out, 'a', 'NETCDF4')
    
    dim_n = var.shape[0]
    create_dimensions_number(ncfile, dim_n)
    
    data_var = create_particles_release(ncfile, varname)
    data_var[:] = var
    
    ncfile.close()
    
def zarr_to_netcdf(path, output_file_preflix, varname):
    import numpy as np
    import xarray as xr
    
    pfile = xr.open_zarr(f"{path}{output_file_preflix}.zarr", decode_cf=True)
    var = np.ma.filled(pfile.variables[varname].values, np.nan)
    pfile.close()
    
    create_netcdf(path, f'{output_file_preflix}.nc', varname, var)