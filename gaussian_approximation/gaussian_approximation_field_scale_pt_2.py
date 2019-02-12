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

import pandas as pd

#Construct a dataframe to store the Gauss fit values for each windspeed
cols = ['wind_mag','Q','C_y','n']
fit_vals_df= pd.DataFrame(columns=cols)

counter = 0
for wind_mag in np.arange(0.4,3.8,0.2):

    dt = 0.01
    simulation_time = 1.*60. #seconds

    release_delay = 30.*60/(wind_mag)

    t_start = 0.0
    t = 0. - release_delay

    xlim = (-1500., 1500.)
    ylim = (-1500., 1500.)


    xlim_odor = (-100., 1500.)
    ylim_odor = (-300., 300.)

    im_extents = (xlim_odor[0],xlim_odor[1],ylim_odor[0],ylim_odor[1])

    array_dim_x = 1000
    array_dim_y = array_dim_x
    conc_locs_x,conc_locs_y = np.meshgrid(
        np.linspace(xlim_odor[0],xlim_odor[1],array_dim_x),
        np.linspace(ylim_odor[0],ylim_odor[1],array_dim_y))



    output_file = 'conc_avg'+str(simulation_time)+'s'+'_ws'+str(wind_mag)+'.pkl'

    #----Start here to load saved data
    with open(output_file,'r') as f:
        (simulation_time,wind_mag,conc_array_accum_avg) = pickle.load(f)


    #The approximation function
    def gauss_approx((x,y),Q,C_y,n):
        return (Q/(2*np.pi*(0.5*C_y*x**((2-n)/2))**2))*\
            np.exp(-1*(y**2/(2*(0.5*C_y*x**((2-n)/2)))))

    #Find the fit Q,C_y,n
    initial_guess = (20.,0.4,1.)


    p_opt,p_cov = curve_fit(gauss_approx,(conc_locs_x.flatten(),conc_locs_y.flatten()),
        conc_array_accum_avg.flatten(),p0=initial_guess)

    Q_est,C_y_est,n_est = p_opt

    fit_vals_df.loc[counter] = pd.Series(
        {'wind_mag':wind_mag,'Q':Q_est,'C_y':C_y_est,'n':n_est})

    print('wind_mag',wind_mag,'Q',Q_est,'C_y',C_y_est,'n',n_est)

    #Plot a cross-section of the fitted approximation function with the accumulated conc field
    C_est = gauss_approx((conc_locs_x,conc_locs_y),Q_est,C_y_est,n_est)
    cross_section_indices = ((C_est-0.0)<0.01)&((C_est-0.0)>0.001)

    ax = plt.subplot(111)
    buffr = 10
    ax.set_xlim((xlim_odor[0]-buffr,xlim_odor[1]+buffr))
    ax.set_ylim((ylim_odor[0]-buffr,ylim_odor[1]+buffr))

    vmin,vmax = 0.,50.
    conc_im = ax.imshow(conc_array_accum_avg.T[::-1], extent=im_extents,
    vmin=vmin, vmax=vmax, cmap='Reds')

    ax.scatter(conc_locs_x[cross_section_indices],
        conc_locs_y[cross_section_indices],color='blue')

    counter +=1
    plt.show()
    raw_input()

fit_vals_df.to_pickle('/fit_vals_df.pkl')
