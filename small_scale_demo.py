from pompy import models, processors
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.animation import FuncAnimation
import scipy
import utility
import sys
import time
import itertools
import cPickle as pickle


dt = 0.01
frame_rate = 20
times_real_time = 5 # seconds of simulation / sec in video
capture_interval = times_real_time*int((1./frame_rate)/dt)
simulation_time = 60*2. #seconds


#Odor arena
xlim = (-15., 15.)
ylim = (0., 40.)
sim_region = models.Rectangle(xlim[0], ylim[0], xlim[1], ylim[1])
wind_region = models.Rectangle(xlim[0]*1.2,ylim[0]*1.2,
xlim[1]*1.2,ylim[1]*1.2)
source_pos = (7.5,25)

#wind setup

#empirical wind data
wind_data_file = '2017_10_26_wind_vectors_1_min_pre_60_min_post_release.csv'
wind_dt = 5
observedWind = models.EmpiricalWindField(wind_data_file,wind_dt,dt)

#wind model setup
aspect_ratio= (xlim[1]-xlim[0])/(ylim[1]-ylim[0])
noise_gain=3.
noise_damp=0.071
noise_bandwidth=0.71
wind_grid_density = 15
Kx = Ky = 30
wind_field = models.WindModel(wind_region,int(wind_grid_density*aspect_ratio),
wind_grid_density,noise_gain=noise_gain,noise_damp=noise_damp,
noise_bandwidth=noise_bandwidth, EmpiricalWindField=observedWind,Kx=Kx,Ky=Ky)

# Set up plume model
centre_rel_diff_scale = 2.
puff_release_rate = 20
puff_spread_rate=0.005
puff_init_rad = 0.01
max_num_puffs=100000
plume_model = models.PlumeModel(
    sim_region, source_pos, wind_field,
    centre_rel_diff_scale=centre_rel_diff_scale,
    puff_release_rate=puff_release_rate,
    puff_init_rad=puff_init_rad,puff_spread_rate=puff_spread_rate,
    max_num_puffs=max_num_puffs)

# Create a concentration array generator
array_z = 0.01
array_dim_x = 1000
array_dim_y = 1000
puff_mol_amount = 1.
array_gen = processors.ConcentrationArrayGenerator(
    sim_region, array_z, array_dim_x, array_dim_y, puff_mol_amount)

# Set up figure
fig = plt.figure(figsize=(7.5, 9))
ax = fig.add_axes([0., 0., 1., 1.])
buffr = 4
ax.set_xlim((xlim[0]-buffr,xlim[1]+buffr))
ax.set_ylim((ylim[0]-buffr,ylim[1]+buffr))

# Display initial concentration field as image
conc_array = array_gen.generate_single_array(plume_model.puff_array)
xmin = sim_region.x_min; xmax = sim_region.x_max
ymin = sim_region.y_min; ymax = sim_region.y_max
im_extents = (xmin,xmax,ymin,ymax)
vmin,vmax = 0.,50.
conc_im = ax.imshow(conc_array.T[::-1], extent=im_extents,
vmin=vmin, vmax=vmax, cmap='Reds')

#Display initial wind vector field
velocity_field = wind_field.velocity_field
u,v = velocity_field[:,:,0],velocity_field[:,:,1]
x_origins,y_origins = wind_field.x_points,wind_field.y_points
coords = scipy.array(list(itertools.product(x_origins, y_origins)))
x_coords,y_coords = coords[:,0],coords[:,1]
vector_field = ax.quiver(x_coords,y_coords,u,v)

#Display observed wind direction
arrow_magn = (xmax-xmin)/20
x_wind,y_wind = observedWind.current_value()
wind_arrow = matplotlib.patches.FancyArrowPatch(posA=(
xmin+(xmax-xmin)/2,ymax-0.2*(ymax-ymin)),posB=
(xmin+(xmax-xmin)/2+arrow_magn*x_wind,
ymax-0.2*(ymax-ymin)+arrow_magn*y_wind),
color='green',mutation_scale=10,arrowstyle='-|>')
plt.gca().add_patch(wind_arrow)

#Initialize stored concentration array object
concStorer = models.ConcentrationStorer(conc_array.T[::-1],conc_im,
dt,simulation_time,vmin,vmax)

# Define animation update function
def update(i):
    for k in range(capture_interval):
        observedWind.update(dt)
        wind_field.update(dt)
        plume_model.update(dt)
    velocity_field = wind_field.velocity_field
    print(velocity_field[0,0,:])
    u,v = velocity_field[:,:,0],velocity_field[:,:,1]
    vector_field.set_UVC(u,v)
    arrow_magn = 4
    x_wind,y_wind = observedWind.current_value()
    wind_arrow.set_positions((xmin+(xmax-xmin)/2,ymax-0.2*(ymax-ymin)),
    (xmin+(xmax-xmin)/2+arrow_magn*x_wind,
    ymax-0.2*(ymax-ymin)+arrow_magn*y_wind))
    conc_array = array_gen.generate_single_array(plume_model.puff_array)
    conc_im.set_data(conc_array.T[::-1])
    concStorer.store(dt,conc_array.T[::-1])
    return [conc_im]#,vector_field]

# Run and save output to video
anim = FuncAnimation(fig, update, frames=int(frame_rate*simulation_time/times_real_time), repeat=False)

# plt.show()

#Save the animation to video
saved = anim.save('plume_saving_test.mp4', dpi=100, fps=frame_rate, extra_args=['-vcodec', 'libx264'])
# concStorer.finish_filling()

#Save the concentration to pkl
# output_file = ('test_conc_array.pkl')
# with open(output_file, 'w') as f:
#         pickle.dump(concStorer,f)
