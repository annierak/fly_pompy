import time
import scipy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import matplotlib.patches
matplotlib.use("Agg")
import sys
import itertools
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
import os
import cPickle as pickle

from scipy.optimize import curve_fit

from pompy import models, processors
import json

from puff_probs import basic_fly_models
from puff_probs import utility


def make_time_averaged_plume(wind_mag):
    dt = 0.25
    simulation_time = 50*60. #seconds
    collection_begins = 25*60. #to let the plume fill the space

    #traps
    source_locations = [(0.,0.),]

    #Odor arena
    xlim = (0., 1200.)
    ylim = (-50., 50.)
    sim_region = models.Rectangle(xlim[0], ylim[0], xlim[1], ylim[1])
    wxlim = (-1500., 1500.)
    wylim = (-1500., 1500.)
    wind_region = models.Rectangle(wxlim[0]*1.2,wylim[0]*1.2,
    wxlim[1]*1.2,wylim[1]*1.2)

    source_pos = scipy.array([scipy.array(tup) for tup in source_locations]).T



    #wind model setup
    diff_eq = False
    constant_wind_angle = 0. #directly positive x wind direction
    aspect_ratio= (wxlim[1]-wxlim[0])/(wylim[1]-wylim[0])
    noise_gain=3.
    noise_damp=0.071
    noise_bandwidth=0.71
    wind_grid_density = 200
    Kx = Ky = 10000 #highest value observed to not cause explosion: 10000
    wind_field = models.WindModel(wind_region,int(wind_grid_density*aspect_ratio),
    wind_grid_density,noise_gain=noise_gain,noise_damp=noise_damp,
    noise_bandwidth=noise_bandwidth,Kx=Kx,Ky=Ky,
    diff_eq=diff_eq,angle=constant_wind_angle,mag=wind_mag)

    #detection_threshold
    detection_threshold = 0.05


    # Set up plume model
    plume_width_factor = 1.
    centre_rel_diff_scale = 2.*plume_width_factor
    puff_release_rate = 10
    puff_spread_rate=0.005
    puff_init_rad = 0.01
    max_num_puffs=int(2e5)


    plume_model = models.PlumeModel(
        sim_region, source_pos, wind_field,simulation_time,dt,
        centre_rel_diff_scale=centre_rel_diff_scale,
        puff_release_rate=puff_release_rate,
        puff_init_rad=puff_init_rad,puff_spread_rate=puff_spread_rate,
        max_num_puffs=max_num_puffs)

    # Create a concentration array generator
    array_z = 0.01

    array_dim_y = 400
    array_dim_x = 5000
    puff_mol_amount = 1.
    array_gen = processors.ConcentrationArrayGenerator(
        sim_region, array_z, array_dim_x, array_dim_y, puff_mol_amount)
    conc_locs_x,conc_locs_y = np.meshgrid(
        np.linspace(xlim[0],xlim[1],array_dim_x),
        np.linspace(ylim[0],ylim[1],array_dim_y))

    fig = plt.figure(figsize=(4*8, 8))


    #Setup flies
    num_flies = 1000
    fly_x_0,fly_y_0 = np.linspace(xlim[0],xlim[1],num_flies),20.*np.ones(num_flies)
    fly_velocity = np.array([0.,-1.6])
    detection_threshold = 0.05


    flies = basic_fly_models.Flies(fly_x_0,fly_y_0,fly_velocity,False,
        puff_mol_amount / (8 * np.pi**3)**0.5,plume_model.puffs,
            detection_threshold,utility.compute_Gaussian,use_grid=False)


    #Compute initial concentration field and display as image
    ax = plt.subplot(211)
    buffr = 1
    ax.set_xlim((xlim[0]-buffr,xlim[1]+buffr))
    ax.set_ylim((ylim[0]-buffr,ylim[1]+buffr))

    # ax.set_xlim((200,250))
    # ax.set_ylim((-10,10))

    conc_array = array_gen.generate_single_array(plume_model.puffs)


    xmin = sim_region.x_min; xmax = sim_region.x_max
    ymin = sim_region.y_min; ymax = sim_region.y_max
    im_extents = (xmin,xmax,ymin,ymax)
    vmin,vmax = 0.,1.
    conc_im = ax.imshow(conc_array.T[::-1], extent=im_extents,
    vmin=vmin, vmax=vmax, cmap='Reds',aspect='auto')
    # ax.set_aspect('equal')

    #Plot flies
    edgecolor_dict = {0 : 'red', 1 : 'white'}
    facecolor_dict = {0 : 'red', 1 : 'white'}

    fly_edgecolors = [edgecolor_dict[mode] for mode in flies.mask_caught]
    fly_facecolors =  [facecolor_dict[mode] for mode in flies.mask_caught]

    fly_dots = plt.scatter(flies.x, flies.y,
            edgecolor=fly_edgecolors,facecolor = fly_facecolors,alpha=0.9)

    axtext = ax.text(-0.2,0.5,'',transform=ax.transAxes)


    #Accumulated concentration field as image
    ax1 = plt.subplot(212)
    buffr = 1
    ax1.set_xlim((xlim[0]-buffr,xlim[1]+buffr))
    ax1.set_ylim((ylim[0]-buffr,ylim[1]+buffr))

    accum_threshold_crosses = np.zeros_like(conc_array)
    accum_threshold_crosses += (conc_array>=detection_threshold).astype(float)

    vmin,vmax = 0.,50.
    conc_im1 = ax1.imshow(conc_array.T[::-1], extent=im_extents,
    vmin=vmin, vmax=vmax, cmap='Reds',aspect='auto')


    #Display initial wind vector field -- subsampled from total
    velocity_field = wind_field.velocity_field
    u,v = velocity_field[:,:,0],velocity_field[:,:,1]
    full_size = scipy.shape(u)[0]
    shrink_factor = 10
    x_origins,y_origins = wind_field.x_points,wind_field.y_points
    coords = scipy.array(list(itertools.product(x_origins, y_origins)))
    x_coords,y_coords = coords[:,0],coords[:,1]
    vector_field = ax.quiver(x_coords,y_coords,u,v)




    #
    plt.ion()
    plt.show()
    #
    t=0.
    capture_interval = 25
    #
    while t<simulation_time:
        for k in range(capture_interval):
            wind_field.update(dt)
            plume_model.update(dt,verbose=True)
            t+=dt
            print(t)


            if t>collection_begins:
                # accum_threshold_crosses += (conc_array>=detection_threshold).astype(float)
                flies.update(dt,t,None)
                conc_array = array_gen.generate_single_array(plume_model.puffs)
                conc_im.set_data((conc_array>=detection_threshold).astype(float).T)#[::-1])
                # conc_im1.set_data(accum_threshold_crosses.T[::-1])

                fly_dots.set_offsets(np.c_[flies.x,flies.y])
                fly_edgecolors = [edgecolor_dict[mode] for mode in flies.mask_caught]
                fly_facecolors =  [facecolor_dict[mode] for mode in flies.mask_caught]
                fly_dots.set_edgecolor(fly_edgecolors)
                fly_dots.set_facecolor(fly_facecolors)
                axtext.set_text(
                    str(np.sum(flies.mask_caught).astype(float)/len(flies.mask_caught))[0:5])

                plt.pause(.0001)





make_time_averaged_plume(1.4)
