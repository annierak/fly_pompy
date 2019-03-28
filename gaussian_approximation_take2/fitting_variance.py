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

    #Compute initial concentration field and display as image
    ax = plt.subplot(211)
    buffr = 1
    ax.set_xlim((xlim[0]-buffr,xlim[1]+buffr))
    ax.set_ylim((ylim[0]-buffr,ylim[1]+buffr))

    ax.set_xlim((200,250))
    ax.set_ylim((-10,10))

    conc_array = array_gen.generate_single_array(plume_model.puffs)


    xmin = sim_region.x_min; xmax = sim_region.x_max
    ymin = sim_region.y_min; ymax = sim_region.y_max
    im_extents = (xmin,xmax,ymin,ymax)
    vmin,vmax = 0.,1.
    conc_im = ax.imshow(conc_array.T[::-1], extent=im_extents,
    vmin=vmin, vmax=vmax, cmap='Reds',aspect='auto')
    ax.set_aspect('equal')


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
                conc_array = array_gen.generate_single_array(plume_model.puffs)
                accum_threshold_crosses += (conc_array>=detection_threshold).astype(float)
        if t>collection_begins:
            conc_im.set_data((conc_array>=detection_threshold).astype(float).T[::-1])
            conc_im1.set_data(accum_threshold_crosses.T[::-1])
            plt.pause(.0001)
            time.sleep(600)

    threshold_cross_prob = accum_threshold_crosses/((t-collection_begins)/dt)
    # conc_array_accum_avg = conc_array_accum/((simulation_time-collection_begins)*dt)

    output_file = 'accum_threshold_prob_'+str(
        detection_threshold)+'_plume_width_factor_'+str(
            plume_width_factor)+'_wind_speed_'+str(
                wind_mag)+str(simulation_time)+'s.pkl'


    # output_file = 'accum_threshold_prob_'+str(
    #     detection_threshold)+'_plume_width_factor_'+str(
    #         plume_width_factor)+str(simulation_time)+'s.pkl'

    with open(output_file, 'w') as f:
        pickle.dump(threshold_cross_prob,f)

    output_file = 'accum_threshold_prob_'+str(
        detection_threshold)+'_plume_width_factor_'+str(
            plume_width_factor)+'_wind_speed_'+str(
                wind_mag)+str(simulation_time)+'s.pkl'



    #----Start here to load saved data
    with open(output_file,'r') as f:
        threshold_cross_prob = pickle.load(f)

    #Check that the largest value in threshold_cross_prob is 1
    print(np.max(threshold_cross_prob))


    plt.close()
    plt.figure(1)
    plt.imshow(threshold_cross_prob.T[::-1])

    plt.figure(2)

    x_crosses = np.arange(9,4951,10)

    sigmas = []
    mus = []
    mags = []

    def gaussian(x,a,mu,sigma):
        return (a/np.sqrt(2*np.pi*sigma**2))*np.exp(-((x-mu)**2)/(2*sigma**2))

    def gaussian_chopped(x,a,mu,sigma):
        output= (a/np.sqrt(2*np.pi*sigma**2))*np.exp(-((x-mu)**2)/(2*sigma**2))
        output[output>1.] = 1.
        return output


    def power_fun(x,a,k):
        return a*x**k

    #
    # plt.ion()
    # plt.show()

    for x_cross in x_crosses:
        plt.figure(2)
        plt.plot(conc_locs_y[:,x_cross],threshold_cross_prob[x_cross,:], \
            label=str(conc_locs_x[0,x_cross])[0:5])
        popt,_ = curve_fit(gaussian,
            conc_locs_y[:,x_cross],threshold_cross_prob[x_cross,:],
            bounds = ([0,-np.inf,0],[np.inf,np.inf,np.inf]))
        a,mu,sigma=popt
        # plt.figure(100)
        # plt.clf()
        # plt.plot(conc_locs_y[:,x_cross],threshold_cross_prob[x_cross,:],label='real')
        # plt.plot(conc_locs_y[:,x_cross],gaussian(conc_locs_y[:,x_cross],*popt),label='fit')
        # plt.xlim([-10,10])
        # plt.legend
        # raw_input()
        sigmas.append(sigma)
        mus.append(mu)
        mags.append(a)




    # plt.figure(3)
    # plt.plot(conc_locs_x[0,x_crosses],
    #     np.sum(threshold_cross_prob[x_crosses,:],axis=1),'o')
    # plt.legend()


    plt.figure(4)
    # plt.plot(conc_locs_x[0,x_crosses],mus,'o',label='Means')
    plt.plot(conc_locs_x[0,x_crosses],sigmas,'o',label='Variances')
    # plt.ylim([0,2])

    popt,_ = curve_fit(power_fun,conc_locs_x[0,x_crosses],sigmas,
        bounds=([0,0],[np.inf,1]))

    sigma_a,sigma_k = popt

    plt.plot(conc_locs_x[0,x_crosses],power_fun(conc_locs_x[0,x_crosses],*popt),
        label='fit')

    plt.legend(bbox_to_anchor=(1, 0.5))


    plt.figure(5)

    plt.plot(conc_locs_x[0,x_crosses],mags/(np.sqrt(2*np.pi*np.array(sigmas)**2)),'o',label='Magnitudes')
    plt.plot(conc_locs_x[0,x_crosses],mags,'o',label='Magnitudes')

    popt,_ = curve_fit(power_fun,conc_locs_x[0,x_crosses],mags,
        bounds=([0,-np.inf],[np.inf,0]))

    a_a,a_k = popt

    plt.plot(conc_locs_x[0,x_crosses],power_fun(conc_locs_x[0,x_crosses],*popt),
        label='fit')


    plt.legend(bbox_to_anchor=(1, 0.5))

    plt.show()
    raw_input


    print('sigma_a: '+str(sigma_a)+', sigma_k: '+str(sigma_k))
    print('a_a: '+str(a_a)+', a_k: '+str(a_k))

    # param_dict = {
    # 'sigma_a':sigma_a,
    # 'sigma_k':sigma_k,
    # 'a_a':a_a,
    # 'a_k':a_k
    # }

    #Hacky version before I figure what function to fit: just save all the
    #sigmas and as, and the delta x

    # mags = np.concatenate((mags,np.linspace(mags[-1],0,int(len(mags)/10))))
    # sigmas = np.concatenate((sigmas,2.*np.ones(int(len(mags)/10))))


    param_dict = {
    'sigmas':sigmas,
    'as':mags,
    'dx': conc_locs_x[0,x_crosses[1]]-conc_locs_x[0,x_crosses[0]]
    }



    param_file_name = 'fit_gaussian_plume_params_odor_threshold_'+str(
        detection_threshold)+'_plume_width_factor_'+str(
            plume_width_factor)+'_wind_speed_'+str(
                wind_mag)+'.pkl'

    with open(param_file_name,'w') as f:
        pickle.dump(param_dict,f)


import multiprocessing

# angles = [0.8,1.2,1.6,1.8,2.0]
mags = [0.4,0.6,1.0,2.2,2.4]

pool = multiprocessing.Pool(processes=6)
# pool.map(mags, angles)
make_time_averaged_plume(1.4)
