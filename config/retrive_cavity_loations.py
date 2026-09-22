#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 10:54:57 2025

@author: jingjin
"""

import sys
sys.path.append('/home/jingjin/work/postpro/')
sys.path.append('/home/jingjin/work/terrafirma/')

import numpy as np  # numerical library
import netCDF4 as nc
from glob import glob
#from plotting_functions import boost_dimension

def AIS_cavities_locations(suite_id, time, TBL_THK = 100, BBL_THK = 100, geo_restricts_lon=None, geo_restricts_lat=None):
    # ------ ufile list ---------
    dataset_folder = f'/projects/oceanparcels/{suite_id}/grid-U'
    ufiles = sorted(glob(f"{dataset_folder}/nemo_{suite_id}o_1m_{time}*U.nc"))
    
    # ------ thkcello file list ---------
    thk_folder = f'/projects/oceanparcels/{suite_id}/thkcello'
    
    if suite_id == 'cz826':
        thkfiles = sorted(glob(f'{thk_folder}/nemo_{suite_id}o_*_*U.nc'))
    else:
        thkfiles = sorted(glob(f'{thk_folder}/nemo_{suite_id}o_*_{time}*U.nc'))
    
    # ------- isf file list ---------
    if suite_id == 'du195':
        suite_id = 'cy838'# --- for du195, use bathymetry-isf from cy838     
              
    isf_folder = f'/home/jingjin/data/terrafirma/{suite_id}/bathymetry-isf'
    
    if suite_id == 'cz826':
        isffiles = sorted(glob(f'{isf_folder}/bisicles_{suite_id}c_*_bathymetry-isf.nc'))
    else:
        isffiles = sorted(glob(f'{isf_folder}/bisicles_{suite_id}c_{time}*_bathymetry-isf.nc'))
    
    # -------- read umask, isfmask -------
    if geo_restricts_lon is None:
        geo_restricts_lon = np.arange(0, nc.Dataset(ufiles[0], 'r').dimensions['x'].size, 1)
    
    if geo_restricts_lat is None:
        geo_restricts_lat = np.arange(0, nc.Dataset(ufiles[0], 'r').dimensions['y'].size, 1)
    
    umask = nc.Dataset(ufiles[0], 'r').variables['uo'][0, :, geo_restricts_lat, geo_restricts_lon]
    isf_mask = boost_dimension(nc.Dataset(isffiles[0], 'r').variables['isf_draft'][geo_restricts_lat, geo_restricts_lon], np.shape(umask))
    umask = np.where(np.where(isf_mask==0, 0, umask)!=0, 1, 0)
    
    bathy = boost_dimension(nc.Dataset(isffiles[0], 'r').variables['Bathymetry_isf'][geo_restricts_lat, geo_restricts_lon], np.shape(umask))
    
    # -------- calculate water column ------
    cavity_water_column = np.abs(
        np.where(isf_mask==0, 0, np.where((isf_mask ==0) & (bathy==0), 0, isf_mask-bathy)))
    
        
    depth_3d_full = np.cumsum(nc.Dataset(thkfiles[0], 'r').variables['thkcello'][0, :, geo_restricts_lat, geo_restricts_lon], axis=0)
    depth_3d_full = np.where(umask==1, depth_3d_full, 0)
    
    # ----- test the sensitivety of the initial depth ------    
    
    depth_3d_TBL = np.where(cavity_water_column==0, 0, 
                            np.where(cavity_water_column>TBL_THK*2, isf_mask+TBL_THK, isf_mask+cavity_water_column*0.25))
    
    depth_3d_BBL = np.where(cavity_water_column==0, 0, 
                           np.where(cavity_water_column>BBL_THK*2, bathy-BBL_THK, bathy-cavity_water_column*0.25))
    # -------------------------------------------------------------------
    
    lon = np.where(
                umask==1,
                boost_dimension(nc.Dataset(ufiles[0], 'r').variables['nav_lon'][geo_restricts_lat, geo_restricts_lon], np.shape(umask)),
                0)
    
    
    lat = np.where(
                umask==1,
                boost_dimension(nc.Dataset(ufiles[0], 'r').variables['nav_lat'][geo_restricts_lat, geo_restricts_lon], np.shape(umask)),
                0)
    
    WetCells_in_the_TBL = (depth_3d_full!=0) & (depth_3d_TBL!=0) & (depth_3d_full<=depth_3d_TBL)
    WetCells_in_the_BBL = (depth_3d_full!=0) & (depth_3d_BBL!=0) & (depth_3d_full>=depth_3d_BBL)
    
    depth_list = depth_3d_full[WetCells_in_the_TBL]
    lon_list = lon[WetCells_in_the_TBL]
    lat_list = lat[WetCells_in_the_TBL]
    
    return lon_list, lat_list, depth_list

def Ross_cavity_locations(suite_id, time, TBL_THK = 100, BBL_THK = 100):
    Ross_lon = np.arange(90, 140, 1)
    Ross_lat = np.arange(0, 44, 1)
    
    lon_list, lat_list, depth_list = AIS_cavities_locations(suite_id, 
                                                            time, 
                                                            TBL_THK=TBL_THK, 
                                                            BBL_THK=BBL_THK, 
                                                            geo_restricts_lon=Ross_lon, 
                                                            geo_restricts_lat=Ross_lat)
    
    return lon_list, lat_list, depth_list

def FRIS_cavity_locations(suite_id, time, TBL_THK = 100, BBL_THK = 100):
    FRIS_lon = np.arange(200, 260, 1)
    FRIS_lat = np.arange(0, 60, 1)
    
    lon_list, lat_list, depth_list = AIS_cavities_locations(suite_id, 
                                                            time, 
                                                            TBL_THK=TBL_THK, 
                                                            BBL_THK=BBL_THK, 
                                                            geo_restricts_lon=FRIS_lon, 
                                                            geo_restricts_lat=FRIS_lat)
    
    return lon_list, lat_list, depth_list

def ROSS_calving_front():

    lons = np.array([ 166.14897,  167.15323,  168.15788,  169.1629 ,
                        170.1683 ,  171.17407,  172.1802 ,  173.18669,
                        174.19353,  175.2007 ,  176.2082 ,  177.21603,
                        178.22415,  179.23259, -179.7587 , -178.74971,
                       -177.74046, -176.73096, -175.72124, -174.71129,
                       -173.70113, -172.69078, -171.68027, -170.66959,
                       -169.65877, -168.64783, -167.63676, -166.62561,
                       -165.61438, -164.6031 , -163.59177, -162.58041,
                       -161.56905, -160.55768, -159.54634, -158.53505])
    num=150
    
    lats_list = np.array([])
    lons_list = np.array([])
    depths_list = np.array([])

    for i in range(0, len(lons)-1):
        if lons[i]*lons[i+1] > 0:
            lons_list = np.append(lons_list, np.linspace(lons[i], lons[i+1], num))
            
        elif lons[i]*lons[i+1] < 0:
            lons_list = np.append(lons_list, np.linspace(lons[i], 180, int(num/2)))
            lons_list = np.append(lons_list, np.linspace(-180, lons[i+1], int(num/2)))
    
    lats_list = np.append(lats_list, np.ones_like(lons_list)*(-75.0))    
    depths_list = np.append(depths_list, np.ones_like(lons_list)*300.0)
    
    return lons_list, lats_list, depths_list

def Read_Ross_Init_Loc():
    path = '/home/jingjin/work/ocean_parcels/locations_list'
    filename = 'Ross_init_locations.nc'
    lon_list = nc.Dataset(f'{path}/{filename}', 'r').variables['lon'][:]
    lat_list = nc.Dataset(f'{path}/{filename}', 'r').variables['lat'][:]
    depth_list = nc.Dataset(f'{path}/{filename}', 'r').variables['depth'][:]
    return lon_list, lat_list, depth_list
