#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 14:30:22 2025

@author: jingjin
"""

import sys
sys.path.append('/home/jingjin/work/postpro/')
sys.path.append('/home/jingjin/work/ocean_parcels/')
sys.path.append('/home/jingjin/work/terrafirma/')

import copy

import cartopy
import cartopy.crs as ccrs
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cmocean
import cmocean.cm as cmo
from matplotlib import colors
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation, PillowWriter, writers
from datetime import timedelta
import matplotlib.path as mpath
from terrafirma_analysis.utils.figures_utils import save_figure
from pandas import Timestamp

output_file = 'FRIS_cavity_cz826_2080-2210_diffu_0.zarr'
pfile = xr.open_zarr(f"/home/jingjin/work/ocean_parcels/MPI_test/{output_file}", decode_cf=True)
#pfile = xr.open_dataset(f"/home/jingjin/work/ocean_parcels/{output_file}")
lon = np.ma.filled(pfile.variables["lon"].values, np.nan)
lat = np.ma.filled(pfile.variables["lat"].values, np.nan)
time = np.ma.filled(pfile.variables["time"].values, np.nan)
z = np.ma.filled(pfile.variables["z"].values, np.nan)
"""
temp = np.ma.filled(pfile.variables["temp"].values, np.nan)
sal = np.ma.filled(pfile.variables["sal"].values, np.nan)
age = np.ma.filled(pfile.variables["age"].values, np.nan)

uvel = np.ma.filled(pfile.variables["uvel"].values, np.nan)
vvel = np.ma.filled(pfile.variables["vvel"].values, np.nan)
wvel = np.ma.filled(pfile.variables["wvel"].values, np.nan)
"""
pfile.close()

#%%
import gsw

def meshgrid(depth, lon=np.linspace(-180, 180, 360), lat=np.linspace(-90, 90, 180)):
    [zz, yy, xx] = np.meshgrid(depth, lat, lon, indexing='ij')
    return zz, yy, xx

def convert_to_pressure(depth, lat=np.linspace(-90, 90, 180)):
    return gsw.conversions.p_from_z(-depth, lat)

def get_background_sigma(sp, pt, depth, 
              lon=np.linspace(-180, 180, 360), lat=np.linspace(-90, 90, 180),
              ref_pressure=0):
   
    # --- sp: practical salinity , in units of psu
    # --- pt: potential temperature, in units of deg C
    # --- ref_pressure: 0 - reference pressure 0 dbar
    #                   1 - reference pressure 1000 dbar
    #                   2 - reference pressure 2000 dbar
    #                   3 - reference pressure 3000 dbar
    
    zz, yy, xx = meshgrid(depth, lon, lat)
    sp = meshgrid(sp, lon, lat)[0]
    pt = meshgrid(pt, lon, lat)[0]
    
    pressure = convert_to_pressure(zz, yy)
    SA = gsw.conversions.SA_from_SP(sp, pressure, xx, yy)
    CT = gsw.conversions.CT_from_pt(SA, pt)
    
    if ref_pressure==0:
        sigma = gsw.density.sigma0(SA, CT)
    elif ref_pressure==1:
        sigma = gsw.density.sigma1(SA, CT)
    elif ref_pressure==2:
        sigma = gsw.density.sigma2(SA, CT)
    elif ref_pressure==3:
        sigma = gsw.density.sigma3(SA, CT)
        
    sigma = np.where(sigma<0, 0, sigma)

    return sigma

def get_particles_sigma(lon, lat, z, sal, temp, ref_pressure=0):
    # --- sp: practical salinity , in units of psu
    # --- pt: potential temperature, in units of deg C
    # --- ref_pressure: 0 - reference pressure 0 dbar
    #                   1 - reference pressure 1000 dbar
    #                   2 - reference pressure 2000 dbar
    #                   3 - reference pressure 3000 dbar
    pressure = convert_to_pressure(z, lat)
    SA = gsw.conversions.SA_from_SP(sal, pressure, lon, lat)
    CT = gsw.conversions.CT_from_pt(SA, temp)
    if ref_pressure==0:
        sigma = gsw.density.sigma0(SA, CT)
    elif ref_pressure==1:
        sigma = gsw.density.sigma1(SA, CT)
    elif ref_pressure==2:
        sigma = gsw.density.sigma2(SA, CT)
    elif ref_pressure==3:
        sigma = gsw.density.sigma3(SA, CT)
        
    sigma = np.where((sigma<0) | (sal==0), 0, sigma)

    return sigma

def SouthPolarStereo_basemap(fig_current=None, ax_current=None):
    if fig_current is None:
        fig_current = plt.gcf()
    
    if ax_current is None:
        ax_current = plt.gca()
    
    # Limit the map to -60 degrees latitude and below.
    ax_current.set_extent([-180, 180, -90, -20], ccrs.PlateCarree())

    ax_current.add_feature(cartopy.feature.LAND, zorder=1, facecolor=[0.5, 0.5, 0.5])
    #ax_current.add_feature(cartopy.feature.OCEAN, zorder=1)
    ax_current.coastlines()
    ax_current.gridlines()
    
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax_current.set_boundary(circle, transform=ax_current.transAxes)

def distance(lon, lat):
    pdx = np.diff(lon, axis=1, prepend=0)
    pdy = np.diff(lat, axis=1, prepend=0)

    # approximation of the distance travelled
    return np.sqrt(np.square(pdx) + np.square(pdy)) > 1e-5
    
def remove_nan_lon(lon):
    plon = lon.flatten()
    plon = np.where(plon>180, plon-360, plon)
    plon = np.where(plon<-180, plon+360, plon)
    return plon[~np.isnan(plon)]

def remove_nan_lat(lat):
    plat = lat.flatten()
    return plat[~np.isnan(plat)]

def remove_empty_frame(time):
    timerange, index, counts = np.unique(time, return_index=True, return_counts=True)
    count_id = np.where(index<=time.shape[1])
    return timerange[count_id]

def stereo_probability_map(t, title):
    
    def density(lon, lat, bins_x, bins_y):
        plon = remove_nan_lon(lon)
        plat = remove_nan_lat(lat)

        #calculate the 2D normalised histogram & bin edges
        H, x, y = np.histogram2d(plon, plat, bins=[bins_x, bins_y], density=True)
        return H, x, y
    
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(ncols=5, nrows=5, figure=fig)
    ### Southern Hemisphere
    ax = fig.add_subplot(
        gs[:,:],
        projection=ccrs.SouthPolarStereo(),
    )
    fig.subplots_adjust(left=0.03, right=0.85, top=0.97, bottom=0.02)
    
    SouthPolarStereo_basemap()
    
    bins_x = np.linspace(-180,180,360)
    bins_y = np.linspace(-90, 90, 180)

    H, xedges, yedges = density(lon[:, :t], lat[:, :t], bins_x, bins_y)
    H = np.where(H == 0, np.nan, H)*100/np.nansum(H)
    scat = ax.pcolormesh(
        xedges, 
        yedges, 
        H.T,
        norm=LogNorm(vmin=1e-6, vmax=0.1),
        cmap=plt.cm.magma.copy(),
        transform=ccrs.PlateCarree(),
    )

    cax = fig.add_axes([ax.get_position().x1+0.01,
                        ax.get_position().y0,
                        0.02,
                        ax.get_position().height])
    cbar = fig.colorbar(scat, ax=ax, cax=cax, orientation='vertical', extend='both')
    cbar.set_label('%')
    ax.set_title(f'{title}')
    fig.canvas.draw()
    #plt.tight_layout()
    return fig, scat, ax

def stereo_concentration_map(t, title):
    
    def histogram(lon, lat, bins_x, bins_y):
        plon = remove_nan_lon(lon)
        plat = remove_nan_lat(lat)

        #calculate the 2D normalised histogram & bin edges
        H, x, y = np.histogram2d(plon, plat, bins=[bins_x, bins_y])
        return H, x, y
    
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(ncols=5, nrows=5, figure=fig)
    ### Southern Hemisphere
    ax = fig.add_subplot(
        gs[:,:],
        projection=ccrs.SouthPolarStereo(),
    )
    fig.subplots_adjust(left=0.03, right=0.85, top=0.97, bottom=0.02)
    
    SouthPolarStereo_basemap()
    
    bins_x = np.linspace(-180,180,360)
    bins_y = np.linspace(-90, 90, 180)

    H, xedges, yedges = histogram(lon[:, :t], lat[:, :t], bins_x, bins_y)

    scat = ax.pcolormesh(
        xedges, 
        yedges, 
        H.T,
        norm=LogNorm(vmin=1, vmax=1e5),
        cmap=plt.cm.magma.copy(),
        transform=ccrs.PlateCarree(),
    )

    cax = fig.add_axes([ax.get_position().x1+0.01,
                        ax.get_position().y0,
                        0.02,
                        ax.get_position().height])
    cbar = fig.colorbar(scat, ax=ax, cax=cax, orientation='vertical', extend='max')
    cbar.set_label('')
    ax.set_title(f'{title}')
    fig.canvas.draw()
    #plt.tight_layout()
    return fig, scat, ax

def histogram_1d(lon, lat, var, bins, xlabel=None, ylabel=None, xlim=None, ylim=None, legend_loc=None):
    global exp_name
    
    def histogram(var, bins):        
        pvar = var.flatten()        
        H, x = np.histogram(pvar[~np.isnan(pvar)], bins=bins, density=True)
        return H, x
    
    nonstuck = distance(lon, lat)
    H, x = histogram(np.where(nonstuck, var, np.nan), bins)
    
    H = np.where(H==0, np.nan, H)*100/np.nansum(H)
    width= np.diff(x)
   # bins_centres=np.linspace(x[0], x[-1], len(x)-1)
    
    fig= plt.figure(figsize=(5, 4))
    gs = gridspec.GridSpec(ncols=5, nrows=5, figure=fig)
    ax1 = fig.add_subplot(
        gs[:,:]
    )
    fig.subplots_adjust(left=0.08, right=0.90, top=0.97, bottom=0.10)
    
    # ------ parbability density bars: initial particles -----------
    """
    if xlabel.split('(')[0].strip().lower() == 'depth':
        path = '/home/jingjin/work/ocean_parcels/locations_list/'
        if exp_name.split('_')[1].strip() == '50':
            filename = 'Ross_TBL_50_repeatdt_6mo.nc'
        elif exp_name.split('_')[1].strip() == '100':
            filename = 'Ross_TBL_100_repeatdt_1yr.nc'           
        var_init = xr.open_dataset(f'{path}{filename}').variables['depth_release'].values
    else:
    """
    var_init = var[:, 0]
    var_init = np.where(var_init==0, np.nan, var_init)

    H_init, x_init = histogram(var_init, bins)
    H_init = np.where(H_init==0, np.nan, H_init)*100/np.nansum(H_init)
    color = 'tab:orange'
    bar2 = ax1.bar(x_init[:-1], H_init, width = width, color=color, 
            edgecolor='black', align='edge', alpha=0.5, 
            label='initial')
    
    # -------- parbability density bars: transit particles ---------
    color = 'tab:blue'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel, color=color)
    bar1 = ax1.bar(x[:-1], H, width = width, color=color, 
            edgecolor='black', align='edge', alpha=0.5, label='transit')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x[::int(len(x)/8)])
    
    if xlim is None:
        ax1.set_xlim([x[0], x[-1]])
    else:
        ax1.set_xlim(xlim)
    
    if ylim is not None:
        ax1.set_ylim(ylim)
    
    # ------- cumulative probability -----------
    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('cumulative probability (%)', color=color)  # we already handled the x-label with ax1
    
    # ------- initial particles ---------
    line2 = ax2.plot(x_init[:-1], np.nancumsum(H_init), color='tab:purple', alpha=0.8, label='initial')
    
    # ------- transit particles -------
    line1 = ax2.plot(x[:-1], np.cumsum(H), color=color, alpha=0.8, label='transit')
    
    ax2.set_ylim([0, 100])
    ax2.tick_params(axis='y', labelcolor=color)
    
    # ask matplotlib for the plotted objects and their labels
    bars, bar_labels = ax1.get_legend_handles_labels()
    lines, line_labels = ax2.get_legend_handles_labels()
    
    if legend_loc is None:
        if xlabel.split('(')[0].strip().lower() == 'salinity':
            legend_loc = 'upper left'
        else:
            legend_loc = 'center right'
        
    ax2.legend(bars + lines, bar_labels + line_labels, loc=legend_loc, frameon=False)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    return 

def particles_release_year(time):
    
    release_year = np.zeros(time.shape[0])
    for t in range(0, len(release_year), 1):
        release_year[t] = int(Timestamp(time[t,0]).year)
        
    return release_year

#%%

sp = np.linspace(34.6, 34.8, 30)
pt = np.linspace(1.0, 2.9, 30)
depth = np.linspace(500, 3500, 30)
sigma_lon = np.linspace(-180, 180, 360)
sigma_lat = np.linspace(-90, -20, 70)
sigma2_background = get_background_sigma(sp, pt, depth, lon=sigma_lon, lat=sigma_lat, ref_pressure=2)

time_id = np.where(time == timerange[-1])
sigma0_particles = get_particles_sigma(lon[time_id], lat[time_id], z[time_id], sal[time_id], temp[time_id], ref_pressure=0)

#%%
sigma2 = np.zeros_like(lon)
for t in range(0, lon.shape[1], 1):
    sigma2[:, t] = get_particles_sigma(lon[:, t], lat[:, t], z[:, t], 
                                       sal[:, t], temp[:, t], ref_pressure=2)

#%%
global exp_name

if_save_figure = True

if_concentration = False
if_probability = True
if_sal_probability = False
if_temp_probability = False
if_depth_probability = False
if_sigma_probability = False

if_time_selection = False

exp_name = 'Static icesheet'
path_to_save_figure = f'/home/jingjin/work/Figures/Probability_maps_FRIS_cz826_2080-2210.jpg'

if if_time_selection:
    release_year = particles_release_year(time)
    time_filter = (release_year>2100) & (release_year<2150)
    time_id = np.where(time_filter)[0]
    plon = lon[time_id, :]
    plat = lat[time_id, :]
    pvar = z[time_id, :]
else:
    plon = lon
    plat = lat
    pvar = z
    
if if_probability:
    stereo_probability_map(lon.shape[1], f'Probability map, {exp_name}')
    
if if_concentration:
    stereo_concentration_map(lon.shape[1], f'The number of particles, {exp_name}')

if if_sal_probability:
    bins = np.arange(31.7, 35.7, 0.1)
    ylim = [0, 55]
    legend_loc = [0.4, 0.6]
    histogram_1d(plon, plat, pvar, bins, xlabel=f'salinity (psu), {exp_name}', ylabel='probability (%)', 
                 ylim = ylim, legend_loc=legend_loc)
    
if if_temp_probability:
    bins = np.arange(-2.6, 4.6, 0.2)
    ylim = [0, 45]
    histogram_1d(plon, plat, pvar, bins, xlabel=f'temperature ($^\circ$C), {exp_name}', 
                 ylabel='probability (%)', ylim = ylim)
    
if if_depth_probability:
    bins = np.arange(0, 4000, 100)
    ylim = [0, 55]
    histogram_1d(plon, plat, pvar, bins, xlabel=f'depth (m), {exp_name}', ylabel='probability (%)', ylim=ylim)

if if_sigma_probability:
    bins = np.arange(35.1, 37.5, 0.05)
    ylim = [0, 25]
    legend_loc = [0.4, 0.5]
    histogram_1d(plon, plat, pvar, bins, xlabel=f'sigma2 (kg m$^{{-3}}$), {exp_name}', ylabel='probability (%)', 
                 ylim=ylim, legend_loc=legend_loc)

if if_save_figure:
    save_figure(path_to_save_figure)
    

