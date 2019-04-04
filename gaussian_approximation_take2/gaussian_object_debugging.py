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


source_locations = [(0.,0.),]
source_pos = scipy.array([scipy.array(tup) for tup in source_locations])
#don't transpose for the Gaussian plumes
constant_wind_angle = 0. #directly positive x wind direction
wind_mag = 1.6

gaussianfitPlumes = models.GaussianFitPlume(source_pos,constant_wind_angle,wind_mag)

xlim = (0., 1200.)
ylim = (-50., 50.)
sim_region = models.Rectangle(xlim[0], ylim[0], xlim[1], ylim[1])
xmin = sim_region.x_min; xmax = sim_region.x_max
ymin = sim_region.y_min; ymax = sim_region.y_max
im_extents = (xmin,xmax,ymin,ymax)

conc_d = gaussianfitPlumes.conc_im(im_extents)
# conc_d[conc_d>.9]=0.
# conc_d[conc_d<.01]=0.


cmap = 'YlOrBr'

ax = plt.subplot(211)
conc_im1 = plt.imshow(conc_d,extent=im_extents,
    interpolation='none',cmap = cmap,origin='lower',aspect='auto')
plt.colorbar(conc_im1,ax=ax)

ax1 = plt.subplot(212)
conc_ima = plt.imshow(np.abs(conc_d-0.01)<0.005,extent=im_extents,
interpolation='none',cmap = cmap,origin='lower',aspect='auto')
# plt.colorbar(conc_im1,ax=ax1)


plt.show()
