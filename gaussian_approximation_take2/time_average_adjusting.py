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
from odor_tracking_sim import swarm_models,trap_models,wind_models


def make_time_averaged_plume(wind_mag):
    dt = 0.25
    simulation_time = 35*60. #seconds
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

    #Setup second plume model (Gaussian approximation)
    source_pos = scipy.array([scipy.array(tup) for tup in source_locations])
    gaussianfitPlumes = models.AdjustedGaussianFitPlume(source_pos,constant_wind_angle,wind_mag)
    source_pos = scipy.array([scipy.array(tup) for tup in source_locations]).T


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

    fig = plt.figure(figsize=(4*4, 4))

    #Video collection
    file_name = 'adjusted_gaussian_test_comparison_wind_mag'+str(wind_mag)
    output_file = file_name+'.pkl'

    frame_rate = 20
    times_real_time = 1 # seconds of simulation / sec in video
    capture_interval = int(scipy.ceil(times_real_time*(1./frame_rate)/dt))

    FFMpegWriter = animate.writers['ffmpeg']
    metadata = {'title':file_name,}
    writer = FFMpegWriter(fps=frame_rate, metadata=metadata)
    writer.setup(fig, file_name+'.mp4', 500)



    #Setup basic flies
    num_flies = 1000
    fly_x_0,fly_y_0 = np.linspace(xlim[0],xlim[1],num_flies),20.*np.ones(num_flies)
    fly_velocity = np.array([0.,-1.6])
    detection_threshold = 0.05

    #
    # flies = basic_fly_models.Flies(fly_x_0,fly_y_0,fly_velocity,False,
    #     puff_mol_amount / (8 * np.pi**3)**0.5,plume_model.puffs,
    #         detection_threshold,utility.compute_Gaussian,use_grid=False)

    #Setup swarm flies

    trap_param = {
        'source_locations' : [source_pos],
        'source_strengths' : [1.],
        'epsilon'          : 0.01,
        'trap_radius'      : 1.,
        'source_radius'    : 1000.
        }

    traps = trap_models.TrapModel(trap_param)

    wind_param = {
            'speed': wind_mag,
            'angle': 2*np.pi,
            'evolving': False,
            'wind_dt': None,
            'dt': dt
            }
    wind_field_noiseless = wind_models.WindField(param=wind_param)


    swarm_param = {
            'swarm_size'          : num_flies,
            'initial_heading'     : scipy.radians(270*np.ones(num_flies)),
            'x_start_position'    : np.copy(fly_x_0),
            'y_start_position'    : np.copy(fly_y_0),
            'flight_speed'        : scipy.full((num_flies,), 1.6),
            'release_time'        : np.zeros(num_flies),
            'release_delay'       : 0.,
            'cast_interval'       : [1, 3],
            'wind_slippage'       : [0.,0.],
            'heading_data':None,
            'odor_thresholds'     : {
                'lower': 0.0005,
                'upper': detection_threshold
                },
            'schmitt_trigger':False,
            'low_pass_filter_length':3, #seconds
            't_stop':3000.,
            'dt_plot':dt,
            'cast_timeout':20,
            'airspeed_saturation':True
            }

    swarm1 = swarm_models.BasicSwarmOfFlies(wind_field_noiseless,traps,param=swarm_param,
        start_type='fh',track_plume_bouts=False,track_arena_exits=False)

    swarm_param.update({'x_start_position'    : np.copy(fly_x_0),
        'y_start_position' : np.copy(fly_y_0)})
    swarm2 = swarm_models.BasicSwarmOfFlies(wind_field_noiseless,traps,param=swarm_param,
        start_type='fh',track_plume_bouts=False,track_arena_exits=False)

    #concentration computer for flies
    box_min,box_max = -100.,1200.

    r_sq_max=20;epsilon=0.00001;N=1e6

    array_gen_flies = processors.ConcentrationValueFastCalculator(
                box_min,box_max,r_sq_max,epsilon,puff_mol_amount,N)


    #Compute initial concentration field and display as image
    ax = plt.subplot(211)
    buffr = 20
    ax.set_xlim((xlim[0]-buffr,xlim[1]+buffr))
    ax.set_ylim((ylim[0]-buffr,ylim[1]+buffr))

    # zoom_x_min,zoom_x_max = 900,950
    # ax.set_xlim((zoom_x_min,zoom_x_max))
    # ax.set_ylim((-10,10))

    conc_array = array_gen.generate_single_array(plume_model.puffs)


    xmin = sim_region.x_min; xmax = sim_region.x_max
    ymin = sim_region.y_min; ymax = sim_region.y_max
    im_extents = (xmin,xmax,ymin,ymax)
    # vmin,vmax = 0.,1.
    vmin,vmax = 0.,.05
    conc_im = ax.imshow(conc_array.T[::-1], extent=im_extents,
    vmin=vmin, vmax=vmax, cmap='Reds',aspect='auto')
    plt.colorbar(conc_im,ax=ax)

    # ax.set_aspect('equal')

    #Puff center scatter plot
    puffs_reshaped = plume_model.puffs.reshape(-1,plume_model.puffs.shape[-1])
    px, py, _, r_sq = puffs_reshaped.T

    # puff_dots = plt.scatter(px,py,s=r_sq,alpha=0.5,color='k')

    #Plot flies
    edgecolor_dict1 = {0 : 'red', 1 : 'white'}
    facecolor_dict1 = {0 : 'red', 1 : 'white'}

    # fly_edgecolors = [edgecolor_dict1[mode] for mode in flies.mask_caught]
    # fly_facecolors =  [facecolor_dict1[mode] for mode in flies.mask_caught]

    # fly_dots1 = plt.scatter(flies.x, flies.y,
    #         edgecolor=fly_edgecolors,facecolor = fly_facecolors,alpha=0.9)

    #Plot swarm

    Mode_StartMode = 0
    Mode_FlyUpWind = 1
    Mode_CastForOdor = 2
    Mode_Trapped = 3


    edgecolor_dict2 = {Mode_StartMode : 'blue',
    Mode_FlyUpWind : 'yellow',
    Mode_CastForOdor : 'red',
    Mode_Trapped :   'black'}

    facecolor_dict2 = {Mode_StartMode : 'blue',
    Mode_FlyUpWind : 'yellow',
    Mode_CastForOdor : 'white',
    Mode_Trapped :   'black'}

    fly_edgecolors = [edgecolor_dict2[mode] for mode in swarm1.mode]
    fly_facecolors =  [facecolor_dict2[mode] for mode in swarm1.mode]

    fly_dots2 = plt.scatter(swarm1.x_position, swarm1.x_position,
            edgecolor=fly_edgecolors,facecolor = fly_facecolors,alpha=0.9)


    axtext = ax.text(-0.1,0.5,'',transform=ax.transAxes,
        verticalalignment='center',
        horizontalalignment='center')


    #Second image: Gaussian fit plume
    ax1 = plt.subplot(212)
    buffr = 20
    ax1.set_xlim((xlim[0]-buffr,xlim[1]+buffr))
    ax1.set_ylim((ylim[0]-buffr,ylim[1]+buffr))


    conc_d = gaussianfitPlumes.conc_im(im_extents)

    #Perform adjustments to the probability plume


    cmap = 'YlOrBr'

    conc_im1 = plt.imshow(conc_d,extent=im_extents,
        interpolation='none',cmap = cmap,origin='lower',aspect='auto')
    plt.colorbar(conc_im1,ax=ax1)

    fly_dots3 = plt.scatter(swarm2.x_position, swarm2.x_position,
            edgecolor=fly_edgecolors,facecolor = fly_facecolors,alpha=0.9)
    ax1text = ax1.text(-0.1,0.5,'',transform=ax1.transAxes,
        verticalalignment='center',
        horizontalalignment='center')


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
    # plt.ion()
    # plt.show()
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
                # flies.update(dt,t,plume_model.puffs)
                swarm1.update(t,dt,wind_field_noiseless,array_gen_flies,traps,plumes=plume_model,
                    pre_stored=False)
                swarm2.update(t,dt,wind_field_noiseless,gaussianfitPlumes,traps)

                conc_array = array_gen.generate_single_array(plume_model.puffs)
                # conc_im.set_data((conc_array>=detection_threshold).astype(float).T)#[::-1])
                # conc_im.set_data((conc_array>=0.01).astype(float).T[::-1])
                conc_im.set_data(conc_array.T[::-1])


                # conc_im1.set_data(accum_threshold_crosses.T[::-1])


                # fly_dots1.set_offsets(np.c_[flies.x,flies.y])
                # fly_edgecolors = [edgecolor_dict1[mode] for mode in flies.mask_caught]
                # fly_facecolors =  [facecolor_dict1[mode] for mode in flies.mask_caught]
                # fly_dots1.set_edgecolor(fly_edgecolors)
                # fly_dots1.set_facecolor(fly_facecolors)


                fly_dots2.set_offsets(np.c_[swarm1.x_position,swarm1.y_position])
                fly_edgecolors = [edgecolor_dict2[mode] for mode in swarm1.mode]
                fly_facecolors =  [facecolor_dict2[mode] for mode in swarm1.mode]
                fly_dots2.set_edgecolor(fly_edgecolors)
                fly_dots2.set_facecolor(fly_facecolors)

                axtext.set_text(
                    "Start Mode: {0:.3f} \n Surging: {1:.3f} \n Casting: {2:.3f} \n Trapped: {3:.3f} \n".format(
                        np.sum(swarm1.mode==Mode_StartMode).astype(float)/len(swarm1.mode),
                        np.sum(swarm1.mode==Mode_FlyUpWind).astype(float)/len(swarm1.mode),
                        np.sum(swarm1.mode==Mode_CastForOdor).astype(float)/len(swarm1.mode),
                        np.sum(swarm1.mode==Mode_Trapped).astype(float)/len(swarm1.mode)
                        ))

                fly_dots3.set_offsets(np.c_[swarm2.x_position,swarm2.y_position])
                fly_edgecolors = [edgecolor_dict2[mode] for mode in swarm2.mode]
                fly_facecolors =  [facecolor_dict2[mode] for mode in swarm2.mode]
                fly_dots3.set_edgecolor(fly_edgecolors)
                fly_dots3.set_facecolor(fly_facecolors)

                ax1text.set_text(
                    "Start Mode: {0:.3f} \n Surging: {1:.3f} \n Casting: {2:.3f} \n Trapped: {3:.3f} \n".format(
                        np.sum(swarm2.mode==Mode_StartMode).astype(float)/len(swarm1.mode),
                        np.sum(swarm2.mode==Mode_FlyUpWind).astype(float)/len(swarm1.mode),
                        np.sum(swarm2.mode==Mode_CastForOdor).astype(float)/len(swarm1.mode),
                        np.sum(swarm2.mode==Mode_Trapped).astype(float)/len(swarm1.mode)
                        ))


                puffs_reshaped = plume_model.puffs.reshape(-1,plume_model.puffs.shape[-1])
                px, py, _, r_sq = puffs_reshaped.T

                # puff_dots.set_offsets(np.c_[px,py])
                # print(np.unique(r_sq[(px>zoom_x_min)&(px<zoom_x_max)]))
                # puff_dots.set_sizes(10*r_sq)


                # plt.pause(.0001)

                writer.grab_frame()





make_time_averaged_plume(1.6)
