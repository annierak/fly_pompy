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

from pompy import models, processors
import json

dt = 0.01
frame_rate = 4
times_real_time = 1 # seconds of simulation / sec in video
capture_interval = int(scipy.ceil(times_real_time*((1./frame_rate)/dt)))
simulation_time = 10# 1.*60. #seconds

file_name = 'live_plume_demo'
output_file = file_name+'.pkl'

#traps
source_locations = [(-20+2.5,-10+2.5),(-10.,0.)]

#Odor arena
xlim = (-20., 20.)
ylim = (-10., 10.)
sim_region = models.Rectangle(xlim[0], ylim[0], xlim[1], ylim[1])
wind_region = models.Rectangle(xlim[0]*1.2,ylim[0]*1.2,
xlim[1]*1.2,ylim[1]*1.2)

source_pos = scipy.array([scipy.array(tup) for tup in source_locations]).T


#wind model setup
diff_eq = True
constant_wind_angle = 1*scipy.pi/8
aspect_ratio= (xlim[1]-xlim[0])/(ylim[1]-ylim[0])
noise_gain=3.
noise_damp=0.071
noise_bandwidth=0.71
wind_grid_density = 20
Kx = Ky = 10 #highest value observed to not cause explosion: 10000
wind_field = models.WindModel(wind_region,int(wind_grid_density*aspect_ratio),
wind_grid_density,noise_gain=noise_gain,noise_damp=noise_damp,
noise_bandwidth=noise_bandwidth,Kx=Kx,Ky=Ky,
diff_eq=diff_eq,angle=constant_wind_angle)


# Set up plume model
centre_rel_diff_scale = 2.
puff_release_rate = 10
puff_spread_rate=0.005
puff_init_rad = 0.01
max_num_puffs=int(2e5)


plume_model = models.PlumeModel(
    sim_region, source_pos, wind_field,simulation_time,
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

# Set up figure
fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(211)
buffr = 1
ax.set_xlim((xlim[0]-buffr,xlim[1]+buffr))
ax.set_ylim((ylim[0]-buffr,ylim[1]+buffr))


#Video
FFMpegWriter = animate.writers['ffmpeg']
metadata = {'title':file_name,}
writer = FFMpegWriter(fps=frame_rate, metadata=metadata)
writer.setup(fig, file_name+'.mp4', 500)


#Put the time in the corner
(xmin,xmax) = ax.get_xlim();(ymin,ymax) = ax.get_ylim()
text = '0 min 0 sec'
timer= ax.text(xmax,ymax,text,color='r',horizontalalignment='right')

# Display initial concentration field as image
conc_array = array_gen.generate_single_array(plume_model.puffs)
xmin = sim_region.x_min; xmax = sim_region.x_max
ymin = sim_region.y_min; ymax = sim_region.y_max
im_extents = (xmin,xmax,ymin,ymax)
vmin,vmax = 0.,50.
conc_im = ax.imshow(conc_array.T[::-1], extent=im_extents,
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

#Display observed wind direction
arrow_magn = (xmax-xmin)/10
x_wind,y_wind = scipy.cos(constant_wind_angle),scipy.sin(constant_wind_angle)
wind_arrow = matplotlib.patches.FancyArrowPatch(posA=(
xmin+(xmax-xmin)/2,ymax-0.2*(ymax-ymin)),posB=
(xmin+(xmax-xmin)/2+arrow_magn*x_wind,
ymax-0.2*(ymax-ymin)+arrow_magn*y_wind),
color='green',mutation_scale=10,arrowstyle='-|>')
plt.gca().add_patch(wind_arrow)

t_start = 0.
t = 0.

array_z = 0.01
array_dim_x = 1000
array_dim_y = array_dim_x
puff_mol_amount = 1.
r_sq_max=20
epsilon=0.00001
N=1e6

array_gen1 = processors.ConcentrationValueFastCalculator(
    xmin,xmax,r_sq_max,epsilon,puff_mol_amount,N)

#Place "measurement" object
measurer_loc = (-9.,.5)
measurer = plt.plot([measurer_loc[0],],[measurer_loc[1],],'o',label='Measurement Location')

concentration_window = np.zeros(10)
ax2 = plt.subplot(212)
conc_trace, = plt.plot(concentration_window)
ax2.set_ylim([0,20])


plt.ion()
plt.show()

last1 = time.time()


while t<simulation_time:
    for k in range(capture_interval):
        wind_field.update(dt)
        plume_model.update(dt)
        t+=dt
        print(t)

        velocity_field = wind_field.velocity_field
        u,v = velocity_field[:,:,0],velocity_field[:,:,1]
        vector_field.set_UVC(u,v)

        new_conc = array_gen1.calc_conc_list(plume_model.puffs, [measurer_loc[0],],[measurer_loc[1]])[0]
        concentration_window[:-1] = concentration_window[1:]
        concentration_window[-1] = new_conc
        conc_trace.set_ydata(concentration_window)

        x_wind,y_wind = scipy.cos(constant_wind_angle),scipy.sin(constant_wind_angle)
        wind_arrow.set_positions((xmin+(xmax-xmin)/2,ymax-0.2*(ymax-ymin)),
        (xmin+(xmax-xmin)/2+arrow_magn*x_wind,
        ymax-0.2*(ymax-ymin)+arrow_magn*y_wind))
        text ='{0} min {1} sec'.format(
        int(scipy.floor(abs(t/60.))),int(scipy.floor(abs(t)%60.)))
        timer.set_text(text)

        conc_array = array_gen.generate_single_array(plume_model.puffs)
        conc_im.set_data(conc_array.T[::-1])

    plt.pause(.0001)
    last = time.time()
    writer.grab_frame()

writer.finish()

print('Time to run 10 seconds, live data:'+str(time.time()-last1))
