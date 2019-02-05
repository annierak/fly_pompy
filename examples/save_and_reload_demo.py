import time
import scipy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import matplotlib.patches
matplotlib.use("Agg")
import sys
import itertools
import h5py
import json
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
import os

from pompy import data_importers
from pompy import models, processors

# #------------------------------BEGIN PLUME GENERATE & SAVE SECTION----------------------------------#
#


dt = 0.01
frame_rate = 20
times_real_time = 5 # seconds of simulation / sec in video
capture_interval = int(scipy.ceil(times_real_time*((1./frame_rate)/dt)))
simulation_time = 10# 1.*60. #seconds

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
ax = fig.add_axes([0.1, 0.1, .8, .8])
buffr = 1
ax.set_xlim((xlim[0]-buffr,xlim[1]+buffr))
ax.set_ylim((ylim[0]-buffr,ylim[1]+buffr))

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


#Initialize stored plume object
plumeStorer = models.PlumeStorer(plume_model,capture_interval*dt,
simulation_time)

#Initialize stored concentration array object for display
concStorer = models.ConcentrationStorer(conc_array.T[::-1],
conc_im,
capture_interval*dt,simulation_time,vmin,vmax,centre_rel_diff_scale,
puff_release_rate,
puff_spread_rate,
puff_init_rad,
puff_mol_amount)


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

#Initialize stored wind vector field object
windStorer = models.WindStorer(wind_field.velocity_field,
wind_field.x_points,wind_field.y_points,capture_interval*dt,simulation_time,
wind_grid_density,noise_gain,noise_damp,
noise_bandwidth)

t_start = 0.
t = 0.

plt.ion()
plt.show()

while t<simulation_time:
    for k in range(capture_interval):
        wind_field.update(dt)
        plume_model.update(dt)
        t+=dt
        print(t)
        plt.pause(.01)

        velocity_field = wind_field.velocity_field
        u,v = velocity_field[:,:,0],velocity_field[:,:,1]
        vector_field.set_UVC(u,v)


        x_wind,y_wind = scipy.cos(constant_wind_angle),scipy.sin(constant_wind_angle)
        wind_arrow.set_positions((xmin+(xmax-xmin)/2,ymax-0.2*(ymax-ymin)),
        (xmin+(xmax-xmin)/2+arrow_magn*x_wind,
        ymax-0.2*(ymax-ymin)+arrow_magn*y_wind))
        text ='{0} min {1} sec'.format(
        int(scipy.floor(abs(t/60.))),int(scipy.floor(abs(t)%60.)))
        timer.set_text(text)

        conc_array = array_gen.generate_single_array(plume_model.puffs)
        conc_im.set_data(conc_array.T[::-1])


        concStorer.store(conc_array.T[::-1])
        last = time.time()

        windStorer.store(velocity_field)
        plumeStorer.store(plume_model.puffs)
#
#------------------------------BEGIN RELOAD & USE SECTION-------------------------------------------#
file_name = 'save_and_reload_demo'
output_file = file_name+'.pkl'

dt = 0.25
frame_rate = 20
times_real_time = 5 # seconds of simulation / sec in video
capture_interval = int(scipy.ceil(times_real_time*(1./frame_rate)/dt))
    # print(capture_interval)
    # raw_input()

# simulation_time =  10#1.*60. #seconds
release_delay = 0.*60

# Set up figure
fig = plt.figure(figsize=(11, 11))
ax = fig.add_subplot(211)

#Video
FFMpegWriter = animate.writers['ffmpeg']
metadata = {'title':file_name,}
writer = FFMpegWriter(fps=frame_rate, metadata=metadata)
writer.setup(fig, file_name+'.mp4', 500)

#
#Import wind and plume objects
conc_file = os.getcwd()+'/'+concStorer.hdf5_filename
wind_file = os.getcwd()+'/'+windStorer.hdf5_filename
plume_file = os.getcwd()+'/'+plumeStorer.hdf5_filename

#Import wind and plume objects
# conc_file = os.getcwd()+'/'+'concObject2.5-22:9.hdf5'
# wind_file = os.getcwd()+'/'+'windObject2.5-22:9.hdf5'
# plume_file = os.getcwd()+'/'+'puffObject2.5-22:9.hdf5'

importedConc = data_importers.ImportedConc(conc_file,release_delay)
importedWind = data_importers.ImportedWind(wind_file,release_delay)

array_z = 0.01
array_dim_x = 1000
array_dim_y = array_dim_x
puff_mol_amount = 1.

importedPlumes = data_importers.ImportedPlumes(plume_file,
    array_z,array_dim_x,array_dim_y,puff_mol_amount,release_delay,
    box_approx=True,epsilon = 0.0001)


xmin,xmax,ymin,ymax = -20., 20.,-10., 10.

vmin,vmax,cmap = importedConc.get_image_params()

#Initial concentration plotting
image = importedConc.plot(0)
# cmap = matplotlib.colors.ListedColormap(['white', 'orange'])
image.set_cmap('Reds')

buffr = 1
ax.set_xlim((xmin-buffr,xmax+buffr))
ax.set_ylim((ymin-buffr,ymax+buffr))


#Put the time in the corner
(xmin,xmax) = ax.get_xlim();(ymin,ymax) = ax.get_ylim()
text = '0 min 0 sec'
timer= ax.text(xmax,ymax,text,color='r',horizontalalignment='right')

#Place "measurement" object
measurer_loc = (-9.,1.)
measurer = plt.plot([measurer_loc[0],],[measurer_loc[1],],'o',label='Measurement Location')

concentration_window = np.zeros(10)
ax2 = plt.subplot(212)
conc_trace, = plt.plot(concentration_window)

t_start = 0*60.0 #- release_delay
t = 0*60.0 #- release_delay

plt.ion()
# plt.show()
# raw_input()

last = time.time()

while t<simulation_time:
    for k in range(capture_interval):
        print('t: {0:1.2f}'.format(t))
        #measurement
        new_conc = importedPlumes.value(t,[measurer_loc[0],],[measurer_loc[1]])[0]
        print(new_conc)
        concentration_window[:-1] = concentration_window[1:]
        concentration_window[-1] = new_conc
        conc_trace.set_ydata(concentration_window)
        ax2.set_ylim([0,np.max(concentration_window)])
        #Update time display
        text ='{0} min {1} sec'.format(int(scipy.floor(t/60.-t_start/60.)),int(scipy.floor(t%60.)))
        timer.set_text(text)
        t+= dt

        conc_array = importedConc.array_at_time(t)

        image.set_data(conc_array)

    plt.pause(0.0001)
    writer.grab_frame()

writer.finish()

print('Time to run 10 seconds, reloaded data:'+str(time.time()-last))
