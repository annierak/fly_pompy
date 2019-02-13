import numpy as np
from pompy import utility
import odor_tracking_sim.trap_models as trap_models
import matplotlib.pyplot as plt
from pompy import models
import scipy
import matplotlib

radius_sources = 1000.
number_sources = 9
box_min,box_max = -1500,1500

location_list, strength_list = utility.create_circle_of_sources(number_sources,
                radius_sources,None)

location_list = [(500.,500.)]

source_pos = np.array([np.array(tup) for tup in location_list])

Q,C_y,n = (20.,0.4,.5)
wind_angle = np.pi/6

suttonPlumes = models.SuttonModelPlume(Q,C_y,n,source_pos,wind_angle)

im_extents= (box_min,box_max,box_min,box_max)

conc_d = suttonPlumes.conc_im(im_extents)

cmap = matplotlib.colors.ListedColormap(['white', 'orange'])

conc_im = plt.imshow(conc_d,extent=(box_min,box_max,box_min,box_max),
    interpolation='none',cmap = cmap,origin='lower')

log_im = scipy.log(conc_d)
cutoff_l = scipy.percentile(log_im[~scipy.isinf(log_im)],10)
cutoff_u = scipy.percentile(log_im[~scipy.isinf(log_im)],99)

conc_im.set_data(log_im)
n = matplotlib.colors.Normalize(vmin=cutoff_l,vmax=cutoff_u)
conc_im.set_norm(n)


plt.show()
