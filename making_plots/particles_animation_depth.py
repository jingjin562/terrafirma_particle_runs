#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 11:19:51 2025

@author: jingjin
"""
import sys
sys.path.append('/home/jingjin/work/postpro/')
sys.path.append('/home/jingjin/work/ocean_parcels/')

import copy

import cartopy
import cartopy.crs as ccrs
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cmocean.cm as cmo
from matplotlib import colors
from matplotlib.animation import FuncAnimation, PillowWriter, writers
from datetime import timedelta
import matplotlib.path as mpath
from matplotlib.colors import Normalize

def distance(lon, lat):
    pdx = np.diff(lon, axis=1, prepend=0)
    pdy = np.diff(lat, axis=1, prepend=0)

    # approximation of the distance travelled
    return np.sqrt(np.square(pdx) + np.square(pdy)) > 1e-5

def remove_empty_frame(time):
    timerange, index, counts = np.unique(time, return_index=True, return_counts=True)
    count_id = np.where(index<=time.shape[1])
    return timerange[count_id]

def survival_rate(lon, lat, time):
    survival = []
    nonstuck = distance(lon, lat)
    timerange = remove_empty_frame(time)

    for t in range(0, len(timerange), 1):  
        time_id = np.where(time == timerange[t])
        test = nonstuck[time_id]
        survival.append((len(test[test==True])/len(test))*100)
        
    return survival

def SouthPolarStereo_basemap(fig_current=None, ax_current=None):
    if fig_current is None:
        fig_current = plt.gcf()
    
    if ax_current is None:
        ax_current = plt.gca()
    
    # Limit the map to -60 degrees latitude and below.
    ax_current.set_extent([-180, 180, -90, -40], ccrs.PlateCarree())

    ax_current.add_feature(cartopy.feature.LAND, zorder=1)
    ax_current.add_feature(cartopy.feature.OCEAN, zorder=1)
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

class ParticleAnimation:
    def __init__(self, lon, lat, time, var, path_to_save,
                 vmin=None, vmax=None, cmap=None, plotted_var="Variable", fps=2):
        """
        Initialize the animation parameters.
        """
        self.lon = lon
        self.lat = lat
        self.time = time
        self.var = np.where(var==0, np.nan, var)
        self.path_to_save = path_to_save
        self.timerange = self._remove_empty_frame(time)
        self.nonstuck = self._distance_mask(lon, lat)
        self.plotted_var = plotted_var.split('(')[0].strip() if '(' in plotted_var else plotted_var
        self.fps = fps
        
        # Auto color scaling if not specified
        self.vmin = np.nanmin(var) if vmin is None else vmin
        self.vmax = np.nanmax(var) if vmax is None else vmax
        self.cmap = self._choose_cmap(self.plotted_var) if cmap is None else cmap

    # ---------- Utility Functions ----------
    @staticmethod
    def _choose_cmap(varname):
        import cmocean.cm as cmo
        """Automatically choose a colormap based on variable name."""
        if varname.lower() == 'temperature':
            return cmo.thermal.copy()
        elif varname.lower() == 'salinity':
            return cmo.haline.copy()
        elif varname.lower() == 'depth':
            return plt.cm.jet.copy()
        elif varname.lower() in ['u', 'v', 'w']:
            return plt.cm.RdBu_r.copy()
        else:
            return plt.cm.viridis.copy()
    
    def _distance_mask(self, lon, lat):
        pdx = np.diff(lon, axis=1, prepend=0)
        pdy = np.diff(lat, axis=1, prepend=0)

        # approximation of the distance travelled
        return np.sqrt(np.square(pdx) + np.square(pdy)) > 1e-5

    def _remove_empty_frame(self, time):
        timerange, index, counts = np.unique(time, return_index=True, return_counts=True)
        count_id = np.where(index<=time.shape[1])
        return timerange[count_id]
    
    def SouthPolarStereo_boundary(self):
        # Compute a circle in axes coordinates, which we can use as a boundary
        # for the map. We can pan/zoom as much as we like - the boundary will be
        # permanently circular.
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        return circle
    
    def setup_figure(self):
        """Create base figure and Cartopy projection."""
        fig = plt.figure(figsize=(5, 5))
        gs = gridspec.GridSpec(ncols=5, nrows=5, figure=fig)
        ax = fig.add_subplot(gs[:, :], projection=ccrs.SouthPolarStereo())
        ax.set_extent([-180, 180, -90, -40], ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.LAND, facecolor=[0.7, 0.7, 0.7], zorder=1)
        #ax.add_feature(cartopy.feature.OCEAN, zorder=1)
        ax.coastlines()
        ax.gridlines()
        circle=self.SouthPolarStereo_boundary()
        ax.set_boundary(circle, transform=ax.transAxes)
        ax.gridlines(draw_labels=False)
        
        fig.subplots_adjust(left=0.03, right=0.83, top=0.97, bottom=0.02)
        return fig, ax

    def single_frame(self, t):
        fig, ax = self.setup_figure()
        time_id = np.where(self.time == self.timerange[t])

        scat = ax.scatter(
            np.where(self.nonstuck[time_id], self.lon[time_id], np.nan),
            np.where(self.nonstuck[time_id], self.lat[time_id], np.nan),
            marker=".",
            s=2,
            c=self.var[time_id],
            cmap=self.cmap,
            alpha=1,
            vmin=self.vmin,
            vmax=self.vmax,
            linewidth=0.15,
            transform=ccrs.PlateCarree(),
        )

        cax = fig.add_axes([
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ])
        
        if self.plotted_var.lower() == 'depth':
            extend = 'max'
        else:
            extend = 'both'
            
        cbar = fig.colorbar(scat, ax=ax, cax=cax, orientation="vertical", extend=extend)
        cbar.set_label(f'{self.plotted_var}')
        ax.set_title(f"Month {t}")
        return fig, scat, ax

    def update(self, t):
        time_id = np.where(self.time == self.timerange[t])
        self.scat.set_offsets(np.vstack((
            np.where(self.nonstuck[time_id], self.lon[time_id], np.nan),
            np.where(self.nonstuck[time_id], self.lat[time_id], np.nan)
        )).T)
        self.scat.set_array(self.var[time_id])
        self.ax.set_title(f"Month {t}")
        return self.scat,

    def create_animation(self):
        fig, scat, ax = self.single_frame(0)
        self.fig, self.scat, self.ax = fig, scat, ax

        ani = animation.FuncAnimation(
            fig, self.update, frames=len(self.timerange), interval=200, blit=False
        )

        writer = animation.FFMpegWriter(
            fps=self.fps,
            metadata=dict(artist="You"),
            bitrate=1800
        )
        ani.save(self.path_to_save, writer=writer)
        plt.close(fig)
        print(f"Animation saved to {self.path_to_save}")

output_file = 'FRIS_cavity_2080-2230_diffu_0.zarr'
pfile = xr.open_zarr(f"/home/jingjin/work/ocean_parcels/MPI_test/{output_file}", decode_cf=True)
#pfile = xr.open_dataset(f"/home/jingjin/work/ocean_parcels/{output_file}")
lon = np.ma.filled(pfile.variables["lon"].values, np.nan)
lat = np.ma.filled(pfile.variables["lat"].values, np.nan)
time = np.ma.filled(pfile.variables["time"].values, np.nan)
z = np.ma.filled(pfile.variables["z"].values, np.nan)

"""
age = np.ma.filled(pfile.variables["age"].values, np.nan)
temp = np.ma.filled(pfile.variables["temp"].values, np.nan)
sal = np.ma.filled(pfile.variables["sal"].values, np.nan)
uvel = np.ma.filled(pfile.variables["uvel"].values, np.nan)
vvel = np.ma.filled(pfile.variables["vvel"].values, np.nan)
wvel = np.ma.filled(pfile.variables["wvel"].values, np.nan)
"""
pfile.close()

plotted_var = 'Depth (m)'
var = z
vmin = 0
vmax = 3500
path_to_save = f'/home/jingjin/work/Figures/particles_FRIS_cavity_{plotted_var.split('(')[0].strip().lower()}_2080-2230.mp4'

anim = ParticleAnimation(lon, lat, time, var, path_to_save, vmin=vmin, vmax=vmax, plotted_var=plotted_var)
anim.create_animation()
