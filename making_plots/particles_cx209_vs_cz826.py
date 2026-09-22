#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 13:35:02 2025

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
import gsw


class Particles_multipanel:
    def __init__(self, cavity, time_list, path_to_save, 
                 if_save_figure=False):
        """
        Initialize the animation parameters.
        """
        self.cavity = cavity
        #self.variable = variable
        self.path_to_save = path_to_save
        self.time_list = time_list
        self.if_save_figure = if_save_figure
    
    def read_parcles_results(self, output_file):         
        pfile = xr.open_zarr(f"/home/jingjin/work/ocean_parcels/MPI_test/{output_file}", decode_cf=True)
        #pfile = xr.open_dataset(f"/home/jingjin/work/ocean_parcels/{output_file}")
        var = {}
        variables = ['lon', 'lat', 'time', 'z']
        var = {variable: np.ma.filled(pfile.variables[variable].values, np.nan) for variable in variables}    
        return var

    def output_file_archive(self, cavity):
        suite_ids = ['cx209', 'cz826']
        
        if cavity == 'Ross':
            output_file = {}
            output_file['cx209'] = 'Ross_cavity_2000-2150_diffu_0.zarr'
            output_file['cz826'] = 'Ross_cavity_cz826_2000-2150_diffu_0.zarr'
        elif cavity == 'FRIS':
            output_file = {}
            output_file['cx209'] = 'FRIS_cavity_2080-2230_diffu_0.zarr'
            output_file['cz826'] = 'FRIS_cavity_cz826_2080-2210_diffu_0.zarr'
            
        return output_file
    
    def particles_release_year(self, output_file): 
        pfile = xr.open_zarr(f"/home/jingjin/work/ocean_parcels/MPI_test/{output_file}", decode_cf=True) 
        time = np.ma.filled(pfile.variables['time'].values, np.nan)
        pfile.close()
        release_year = np.zeros(time.shape[0])
        for t in range(0, len(release_year), 1):
            release_year[t] = int(Timestamp(time[t,0]).year)           
        return release_year
    
    def particles_release_number_first_15_years(self, cavity):
        number = {}
        if cavity == 'Ross':
            number['cz826']=41760
            number['cx209']=44998
        return number
    
    def particles_release_number_last_15_years(self, cavity):
        number = {}
        if cavity == 'Ross':
            number['cz826']=41760
            number['cx209']=64552
        return number
    
    def data_selected_by_release_year(self, output_file, variable):
        
        release_year = self.particles_release_year()
        
        if 'Ross' in self.output_file:
            time_range_array_1 = np.arange(2000, 2150, 25)
            time_range_array_2 = np.arange(2025, 2150+25, 25)
        elif 'FRIS' in self.output_file:
            time_range_array_1 = np.arange(2080, 2210, 25)
            time_range_array_2 = np.arange(2080+25, 2210+25, 25)
        
        data_selected_by_release_year = {}
        
        pfile = xr.open_zarr(f"/home/jingjin/work/ocean_parcels/MPI_test/{output_file}", decode_cf=True)
        data = np.ma.filled(pfile.variables[variable].values, np.nan)
        
        for t in range(0, len(time_list), 1):
            time_range = time_list[t]
            time_filter = (release_year>time_range_array_1[t]) & (release_year<time_range_array_2[t])
            time_id = np.where(time_filter)[0]
            data_selected_by_release_year[time_range] = data[time_id, :]
        return data_selected_by_release_year
    
    def data_selected_by_age(self, output_file, variable):
        data_selected_by_age = {}
        
        age_range_array_1 = np.array([0, 1, 5])*12
        age_range_array_2 = np.array([1, 5, 10])*12
        
        pfile = xr.open_zarr(f"/home/jingjin/work/ocean_parcels/MPI_test/{output_file}", decode_cf=True)
        data = np.ma.filled(pfile.variables[variable].values, np.nan)
        
        for t in range(0, len(time_list), 1):
            time_range = time_list[t]
            age_filter = np.arange(age_range_array_1[t], age_range_array_2[t], 1)
            data_selected_by_age[time_range] = data[:, age_filter]
        return data_selected_by_age
    
    def time_list_data(self, output_file, variable):       
        if self.selection_method == 'release_year':
            if 'sigma' in variable:
                data = {}
                lon = self.data_selected_by_release_year(output_file,'lon')
                lat = self.data_selected_by_release_year(output_file,'lat')
                depth = self.data_selected_by_release_year(output_file,'z')
                temp = self.data_selected_by_release_year(output_file,'temp')
                sal = self.data_selected_by_release_year(output_file,'sal')
                for time_range in time_list:
                    data[time_range] = self.particles_sigma2(lon[time_range], 
                                                             lat[time_range],
                                                             depth[time_range],
                                                             sal[time_range],
                                                             temp[time_range])
            else:
                data = self.data_selected_by_release_year(output_file, variable)
        
        elif self.selection_method == 'age':
            if 'sigma' in variable:
                data = {}
                lon = self.data_selected_by_age(output_file,'lon')
                lat = self.data_selected_by_age(output_file,'lat')
                depth = self.data_selected_by_age(output_file,'z')
                temp = self.data_selected_by_age(output_file,'temp')
                sal = self.data_selected_by_age(output_file,'sal')
                for time_range in time_list:
                    data[time_range] = self.particles_sigma2(lon[time_range], 
                                                             lat[time_range],
                                                             depth[time_range],
                                                             sal[time_range],
                                                             temp[time_range])
            else:
                data = self.data_selected_by_age(output_file,variable)
        return data
    
    def remove_nan_lon(self, lon):
        plon = lon.flatten()
        plon = np.where(plon>180, plon-360, plon)
        plon = np.where(plon<-180, plon+360, plon)
        return plon[~np.isnan(plon)]

    def remove_nan_lat(self, lat):
        plat = lat.flatten()
        return plat[~np.isnan(plat)]

    def remove_empty_frame(self, output_file):
        pfile = xr.open_zarr(f"/home/jingjin/work/ocean_parcels/MPI_test/{output_file}", decode_cf=True) 
        time = np.ma.filled(pfile.variables['time'].values, np.nan)
        pfile.close()
        timerange, index, counts = np.unique(time, return_index=True, return_counts=True)
        count_id = np.where(index<=time.shape[1])
        return timerange[count_id]
    
    def convert_to_pressure(self, depth, lat=np.linspace(-90, 90, 180)):       
        return gsw.conversions.p_from_z(-depth, lat)
    
    def get_particles_sigma(self, lon, lat, z, sal, temp, ref_pressure=0):
        # --- sp: practical salinity , in units of psu
        # --- pt: potential temperature, in units of deg C
        # --- ref_pressure: 0 - reference pressure 0 dbar
        #                   1 - reference pressure 1000 dbar
        #                   2 - reference pressure 2000 dbar
        #                   3 - reference pressure 3000 dbar
        pressure = self.convert_to_pressure(z, lat)
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
    
    def particles_sigma2(self, lon, lat, z, sal, temp, ref_pressure=2):
        sigma2 = np.zeros_like(lon)
        
        for t in range(0, lon.shape[1], 1):
            plon = lon[:, t]
            plon = np.where(plon>180, plon-360, plon)
            plon = np.where(plon<-180, plon+360, plon)
            sigma2[:, t] = self.get_particles_sigma(lon[:, t], lat[:, t], z[:, t], 
                                               sal[:, t], temp[:, t], ref_pressure=ref_pressure)
        return sigma2
    
    def SouthPolarStereo_boundary(self):
        # Compute a circle in axes coordinates, which we can use as a boundary
        # for the map. We can pan/zoom as much as we like - the boundary will be
        # permanently circular.
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        return circle
    
    def changeFontSize(self, ax, size):
        ax.title.set_fontsize(size)
        for item in ([ax.xaxis.label, ax.yaxis.label] +
                      ax.get_xticklabels() + ax.get_yticklabels() + 
                      ax.get_legend().get_texts()):
             item.set_fontsize(size)
    
    def figure_layout_stereo(self):
        fig = plt.figure(figsize=[11, 7])
        plt.clf()
        widths = [3, 3, 3]
        heights = [3, 3]
        gs = fig.add_gridspec(ncols=3, nrows=2, width_ratios=widths,
                                height_ratios=heights, hspace=0.02, wspace=0.35)
        fig.subplots_adjust(left=0.01, right=0.90, top=0.95, bottom=0.005)
        ax = {}
        ax['cx209'] = {
            self.time_list[0]: fig.add_subplot(gs[0, 0], projection=ccrs.SouthPolarStereo()),
            self.time_list[1]: fig.add_subplot(gs[0, 1], projection=ccrs.SouthPolarStereo()),
            self.time_list[2]: fig.add_subplot(gs[0, 2], projection=ccrs.SouthPolarStereo()),
            }
        ax['cz826'] = {
            self.time_list[0]: fig.add_subplot(gs[1, 0], projection=ccrs.SouthPolarStereo()),
            self.time_list[1]: fig.add_subplot(gs[1, 1], projection=ccrs.SouthPolarStereo()),
            self.time_list[2]: fig.add_subplot(gs[1, 2], projection=ccrs.SouthPolarStereo()),
            }
        
        return fig, ax
    
    def figure_layout(self):
        fig = plt.figure(figsize=[13,7])
        plt.clf()
        widths = [4, 4, 4]
        heights = [3, 3]
        gs = fig.add_gridspec(ncols=3, nrows=2, width_ratios=widths,
                                height_ratios=heights, hspace=0.30, wspace=0.35)
        fig.subplots_adjust(left=0.06, right=0.95, top=0.93, bottom=0.07)
        ax = {}
        ax['cx209'] = {
            self.time_list[0]: fig.add_subplot(gs[0, 0]),
            self.time_list[1]: fig.add_subplot(gs[0, 1]),
            self.time_list[2]: fig.add_subplot(gs[0, 2]),
            }
        ax['cz826'] = {
            self.time_list[0]: fig.add_subplot(gs[1, 0]),
            self.time_list[1]: fig.add_subplot(gs[1, 1]),
            self.time_list[2]: fig.add_subplot(gs[1, 2]),
            }
        
        return fig, ax
    
    def _basemap(self):
        """Create base figure and Cartopy projection."""
        fig, ax = self.figure_layout_stereo()
        
        for id in self.suite_id:
            for time_range in self.time_list:
                ax_current = ax[id][time_range]
                ax_current.set_extent([-180, 180, -90, -20], ccrs.PlateCarree())
                ax_current.add_feature(cartopy.feature.LAND, facecolor=[0.7, 0.7, 0.7], zorder=1)
                #ax.add_feature(cartopy.feature.OCEAN, zorder=1)
                ax_current.coastlines()
                ax_current.gridlines()
                circle=self.SouthPolarStereo_boundary()
                ax_current.set_boundary(circle, transform=ax_current.transAxes)
                ax_current.gridlines(draw_labels=False)
        
        return fig, ax
    
    def subpanel_label(self):
        label = {}
        label['cx209'] = ['(a)', '(b)', '(c)']
        label['cz826'] = ['(d)', '(e)', '(f)']
        exp_name = {}
        exp_name['cx209'] = 'interactive'
        exp_name['cz826'] = 'static'
        return label, exp_name
    
    def histogram_1d_density(self, var, bins):
        pvar = var.flatten()        
        H, x = np.histogram(pvar[~np.isnan(pvar)], bins=bins, density=True)
        H = np.where(H==0, np.nan, H)*100/np.nansum(H)
        return H, x
    
    def histogram_concentration(self, lon, lat, bins_x, bins_y):
        plon = self.remove_nan_lon(lon)
        plat = self.remove_nan_lat(lat)

        #calculate the 2D normalised histogram & bin edges
        H, x, y = np.histogram2d(plon, plat, bins=[bins_x, bins_y])
        return H, x, y
    
    def histogram_2d_density(self, lon, lat, bins_x, bins_y, selection_method='release_year'):
        plon = self.remove_nan_lon(lon)
        plat = self.remove_nan_lat(lat)

        #calculate the 2D normalised histogram & bin edges
        H, x, y = np.histogram2d(plon, plat, bins=[bins_x, bins_y], density=True)
        H = np.where(H == 0, np.nan, H)*100/np.nansum(H)
        return H, x, y
    
    def _bar_chart(self, variable, bins, selection_method='release_year', 
                   xlabel=None, ylabel=None, xlim=None, ylim=None, legend_loc=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.legend_loc = legend_loc
        self.selection_method = selection_method
        self.suite_id = ['cx209', 'cz826']
        
        subpanel_label, exp_name = self.subpanel_label()
        number = self.particles_release_number_last_15_years(self.cavity)
        
        plt.clf()
        
        output_file = self.output_file_archive(self.cavity)
        
        data = {}
        data = {id: self.time_list_data(output_file[id], variable) for id in self.suite_id}
        
        color_init = 'tab:orange'
        color_transit = 'tab:blue'
        color_cumu = 'tab:red'
        color_cumu_init = 'tab:purple'
        
        fig, ax = self.figure_layout()
        fig.suptitle('Probability density, released in the last 15 years', y=0.997, fontsize=14)
        
        for id in self.suite_id:        
            hist = {}
            hist = {time_range: self.histogram_1d_density(data[id][time_range][-number[id]:], bins) for time_range in self.time_list}
            
            hist_init = self.histogram_1d_density(data[id][self.time_list[0]][-number[id]:, 0], bins)
            
            title_label = subpanel_label[id]
            exp_current = exp_name[id]
            
            for time_range in self.time_list:
                ax_current = ax[id][time_range]
                ax_current.set_title(f'{title_label[time_list.index(time_range)]} {time_range}, {exp_current}')
                
                if time_range == self.time_list[0]:
                    # ------ parbability density bars: initial particles -----------
                    width = np.diff(hist_init[1])
                    bar2 = ax_current.bar(hist_init[1][:-1], hist_init[0], width = width, color=color_init, 
                            edgecolor='black', align='edge', alpha=0.5, 
                            label='initial')
                
                # -------- parbability density bars: transit particles ---------
                width = np.diff(hist[time_range][1])
                bar1 = ax_current.bar(hist[time_range][1][:-1], hist[time_range][0], width = width, color=color_transit, 
                        edgecolor='black', align='edge', alpha=0.5, 
                        label='transit')
                
                ax_current.set_xlabel(self.xlabel)
                ax_current.set_ylabel(self.ylabel, color=color_transit)
                ax_current.tick_params(axis='y', labelcolor=color_transit)
                ax_current.set_xticks(hist[time_range][1][::int(len(hist[time_range][1])/5)])
                
                if self.xlim is None:
                    ax_current.set_xlim([hist[time_range][1][0], hist[time_range][1][-1]])
                else:
                    ax_current.set_xlim(self.xlim)
                
                if self.ylim is not None:
                    ax_current.set_ylim(self.ylim)
                  
                # ------- cumulative probability -----------
                ax_copy = ax_current.twinx()  # instantiate a second Axes that shares the same x-axis
                
                ax_copy.set_ylabel('cumulative probability (%)', color=color_cumu)  # we already handled the x-label with ax1
                
                if time_range == self.time_list[0]:
                    # ------- initial particles ---------
                    line2 = ax_copy.plot(hist_init[1][:-1], np.nancumsum(hist_init[0]), color=color_cumu_init, alpha=0.8, label='initial')
                    
                # ------- transit particles -------
                line1 = ax_copy.plot(hist[time_range][1][:-1], np.cumsum(hist[time_range][0]), color=color_cumu, alpha=0.8, label='transit')
                
                ax_copy.set_ylim([0, 100])
                ax_copy.tick_params(axis='y', labelcolor=color_cumu)
                
                if time_range == self.time_list[0]:
                    # ask matplotlib for the plotted objects and their labels
                    bars, bar_labels = ax_current.get_legend_handles_labels()
                    lines, line_labels = ax_copy.get_legend_handles_labels()
                    
                    if self.legend_loc is None:
                        if self.xlabel is not None:
                            if self.xlabel.split('(')[0].strip().lower() == 'salinity':
                                legend_loc = 'upper left'
                            else:
                                legend_loc = 'center right'
                        else:
                            legend_loc = 'center right'
                                           
                    ax_copy.legend(bars + lines, bar_labels + line_labels, loc=legend_loc, frameon=False)
    
            #fig.tight_layout()  # otherwise the right y-label is slightly clipped
            
            # ----- update fontsize ----------
            
            # self.changeFontSize(plt.gca(), 17)
        
        if self.if_save_figure:
            save_figure(f'{self.path_to_save}')
            
    def _concentration(self, 
                       bins_x=np.linspace(-180,180,360),  bins_y = np.linspace(-90, 90, 180),
                       selection_method='release_year'):
        self.selection_method = selection_method
        self.suite_id = ['cx209', 'cz826']
        
        subpanel_label, exp_name = self.subpanel_label()
        number = self.particles_release_number_first_15_years(self.cavity)
        
        plt.clf()
        fig, ax = self._basemap()
        
        fig.suptitle('The number of particles, released in the last 15 years', y=0.99, fontsize=14)
        for id in self.suite_id:
            output_file = self.output_file_archive(cavity)
            lon = self.time_list_data(output_file[id], 'lon')
            lat = self.time_list_data(output_file[id], 'lat')
            
            title_label = subpanel_label[id]
            exp_current = exp_name[id]
            for time_range in self.time_list:
                self.lon = lon[time_range][:number[id], ]
                self.lat = lat[time_range][:number[id], ]
                H, xedges, yedges = self.histogram_concentration(self.lon, self.lat, bins_x, bins_y)
                
                ax_current = ax[id][time_range]
                scat = ax_current.pcolormesh(
                    xedges, 
                    yedges, 
                    H.T,
                    norm=LogNorm(vmin=1, vmax=1e5),
                    cmap=plt.cm.magma.copy(),
                    transform=ccrs.PlateCarree(),
                )
        
                cax = fig.add_axes([ax_current.get_position().x1+0.01,
                                    ax_current.get_position().y0,
                                    0.02,
                                    ax_current.get_position().height])
                cbar = fig.colorbar(scat, ax=ax_current, cax=cax, orientation='vertical', extend='max')
                cbar.set_label('')
                ax_current.set_title(f'{title_label[time_list.index(time_range)]} {time_range}, {exp_current}')
                fig.canvas.draw()
            
        if self.if_save_figure:
            save_figure(f'{self.path_to_save}')
            
    def _probability(self, 
                     bins_x=np.linspace(-180,180,360),  bins_y = np.linspace(-90, 90, 180),
                     selection_method='release_year'):
        self.selection_method = selection_method
        self.suite_id = ['cx209', 'cz826']
        
        subpanel_label, exp_name = self.subpanel_label()
        number = self.particles_release_number_first_15_years(self.cavity)
        
        plt.clf()
        fig, ax = self._basemap()
        
        fig.suptitle('Probability density, released in the first 15 years', y=0.99, fontsize=14)
        for id in self.suite_id:
            output_file = self.output_file_archive(cavity)
            lon = self.time_list_data(output_file[id], 'lon')
            lat = self.time_list_data(output_file[id], 'lat')
            
            title_label = subpanel_label[id]
            exp_current = exp_name[id]
            for time_range in self.time_list:
                self.lon = lon[time_range][:number[id], ]
                self.lat = lat[time_range][:number[id], ]
                H, xedges, yedges = self.histogram_2d_density(self.lon, self.lat, bins_x, bins_y)
                
                ax_current = ax[time_range]
                scat = ax_current.pcolormesh(
                    xedges, 
                    yedges, 
                    H.T,
                    norm=LogNorm(vmin=1e-5, vmax=1e-1),
                    cmap=plt.cm.magma.copy(),
                    transform=ccrs.PlateCarree(),
                )
        
                cax = fig.add_axes([ax_current.get_position().x1+0.01,
                                    ax_current.get_position().y0,
                                    0.02,
                                    ax_current.get_position().height])
                cbar = fig.colorbar(scat, ax=ax_current, cax=cax, orientation='vertical', extend='both')
                cbar.set_label('')
                ax_current.set_title(f'{title_label[time_list.index(time_range)]} {time_range}, {exp_current}')
                fig.canvas.draw()
            
        if self.if_save_figure:
            save_figure(f'{self.path_to_save}')


#%%
time_list = ['Age 0-1',
            'Age 1-5',
            'Age 5-10',]

cavity = 'Ross'

if_depth_probability = True

path_to_save = '/home/jingjin/work/Figures/cx209_vs_cz826_Probability_depth_age_variation_last_15_years.jpg'
if_save_figure = True

particles_plots = Particles_multipanel(cavity, time_list, path_to_save, 
                     if_save_figure=if_save_figure)
selection_method = 'age' # ----'age' or 'release_year'

if if_depth_probability:
    variable = 'z'
    bins = np.arange(0, 1000+100, 100)
    ylim = [0, 60]
    xlabel=f'depth (m)'
    ylabel='probability (%)'
    particles_plots._bar_chart(variable, bins, selection_method=selection_method, 
                               xlabel=xlabel, ylabel=ylabel, ylim=ylim)