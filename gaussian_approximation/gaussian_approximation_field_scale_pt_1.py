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

for wind_mag in np.arange(0.4,3.8,0.2):

    dt = 0.01
    simulation_time = 1.*60. #seconds

    release_delay = 30.*60/(wind_mag)

    t_start = 0.0
    t = 0. - release_delay


    #traps
    source_locations = [(0.,0.),]

    #Wind arena as big as wind arena for desert simulation
    xlim = (-1500., 1500.)
    ylim = (-1500., 1500.)
    wind_region = models.Rectangle(xlim[0]*1.2,ylim[0]*1.2,
    xlim[1]*1.2,ylim[1]*1.2)

    #Odor arena just directly surrounding the single plume
    xlim_odor = (-100., 1500.)
    ylim_odor = (-300., 300.)
    sim_region = models.Rectangle(xlim_odor[0], ylim_odor[0], xlim_odor[1], ylim_odor[1])

    source_pos = scipy.array([scipy.array(tup) for tup in source_locations]).T

    #wind model setup
    diff_eq = True
    constant_wind_angle = 0. #directly positive x wind direction
    aspect_ratio= (xlim[1]-xlim[0])/(ylim[1]-ylim[0])
    noise_gain=3.
    noise_damp=0.071
    noise_bandwidth=0.71
    wind_grid_density = 20
    Kx = Ky = 10 #highest value observed to not cause explosion: 10000
    wind_field = models.WindModel(wind_region,int(wind_grid_density*aspect_ratio),
    wind_grid_density,noise_gain=noise_gain,noise_damp=noise_damp,
    noise_bandwidth=noise_bandwidth,Kx=Kx,Ky=Ky,
    diff_eq=diff_eq,angle=constant_wind_angle,mag=wind_mag)

    # Set up plume model
    centre_rel_diff_scale = 2.
    puff_release_rate = 10
    puff_spread_rate=0.005
    puff_init_rad = 0.01
    max_num_puffs=int(2e5)


    plume_model = models.PlumeModel(
        sim_region, source_pos, wind_field,simulation_time+release_delay,dt,
        centre_rel_diff_scale=centre_rel_diff_scale,
        puff_release_rate=puff_release_rate,
        puff_init_rad=puff_init_rad,puff_spread_rate=puff_spread_rate,
        max_num_puffs=max_num_puffs)

    # Create a concentration array generator
    array_z = 0.01

    array_dim_x = 1000
    array_dim_y = array_dim_x
    puff_mol_amount = 1.
    array_gen = processors.ConcentrationArrayGenerator(
        sim_region, array_z, array_dim_x, array_dim_y, puff_mol_amount)

    fig = plt.figure(figsize=(8, 8))

    #Compute initial concentration field and display as image
    ax = plt.subplot(211)
    buffr = 1
    ax.set_xlim((xlim[0]-buffr,xlim[1]+buffr))
    ax.set_ylim((ylim[0]-buffr,ylim[1]+buffr))

    conc_array = array_gen.generate_single_array(plume_model.puffs)

    xmin = sim_region.x_min; xmax = sim_region.x_max
    ymin = sim_region.y_min; ymax = sim_region.y_max
    im_extents = (xmin,xmax,ymin,ymax)
    vmin,vmax = 0.,50.
    conc_im = ax.imshow(conc_array.T[::-1], extent=im_extents,
    vmin=vmin, vmax=vmax, cmap='Reds')

    #Accumulated concentration field as image
    ax1 = plt.subplot(212)
    buffr = 1
    ax1.set_xlim((xlim[0]-buffr,xlim[1]+buffr))
    ax1.set_ylim((ylim[0]-buffr,ylim[1]+buffr))

    conc_array_accum = np.zeros_like(conc_array)
    conc_array_accum += conc_array

    vmin,vmax = 0.,50.
    conc_im1 = ax1.imshow(conc_array.T[::-1], extent=im_extents,
    vmin=vmin, vmax=vmax, cmap='Reds')


    #Display initial wind vector field -- subsampled from total
    velocity_field = wind_field.velocity_field
    u,v = velocity_field[:,:,0],velocity_field[:,:,1]
    full_size = scipy.shape(u)[0]
    shrink_factor = 10
    x_origins,y_origins = wind_field.x_points,wind_field.y_points
    coords = scipy.array(list(itertools.product(x_origins, y_origins)))
    x_coords,y_coords = coords[:,0],coords[:,1]
    vector_field = ax.quiver(x_coords,y_coords,u,v)


    plt.ion()
    plt.show()

    capture_interval = 25

    while t<simulation_time:
        for k in range(capture_interval):
            wind_field.update(dt)
            plume_model.update(dt,verbose=True)
            t+=dt
            print(t)

            velocity_field = wind_field.velocity_field
            u,v = velocity_field[:,:,0],velocity_field[:,:,1]
            vector_field.set_UVC(u,v)

            if t>0:
                conc_array = array_gen.generate_single_array(plume_model.puffs)
                conc_im.set_data(conc_array.T[::-1])

                conc_array_accum +=conc_array
                conc_im1.set_data(conc_array_accum.T[::-1])

        plt.pause(.0001)

    conc_array_accum_avg = conc_array_accum/(simulation_time*dt)

    output_file = 'conc_avg'+str(simulation_time)+'s'+'_ws'+str(wind_mag)+'.pkl'

    with open(output_file, 'w') as f:
        pickle.dump((simulation_time,wind_mag,conc_array_accum_avg),f)
