from pompy import models, processors
from odor_tracking_sim import trap_models
import odor_tracking_sim.trap_models as trap_models
import odor_tracking_sim.utility as utility
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.animation import FuncAnimation
import scipy
import scipy.sparse
import sys
import time
import itertools


dt = 0.01
frame_rate = 20
times_real_time = 5 # seconds of simulation / sec in video
capture_interval = int(scipy.ceil(times_real_time*((1./frame_rate)/dt)))
simulation_time = 5.*60. #seconds
t_start = -20*60. #time before fly release


#traps
number_sources = 8
radius_sources = 1000.0
trap_radius = 0.5
location_list, strength_list = utility.create_circle_of_sources(number_sources,
                radius_sources,None)
trap_param = {
        'source_locations' : location_list,
        'source_strengths' : strength_list,
        'epsilon'          : 0.01,
        'trap_radius'      : trap_radius,
        'source_radius'    : radius_sources
}

traps = trap_models.TrapModel(trap_param)

#Odor arena
xlim = (-1500., 1500.)
ylim = (-1500., 1500.)
sim_region = models.Rectangle(xlim[0], ylim[0], xlim[1], ylim[1])
wind_region = models.Rectangle(xlim[0]*1.2,ylim[0]*1.2,
xlim[1]*1.2,ylim[1]*1.2)

source_pos = scipy.array([scipy.array(tup) for tup in traps.param['source_locations']]).T

#wind model setup
diff_eq = True
constant_wind_angle = 7*scipy.pi/4
aspect_ratio= (xlim[1]-xlim[0])/(ylim[1]-ylim[0])
noise_gain=3.
noise_damp=0.071
noise_bandwidth=0.71
wind_grid_density = 200
Kx = Ky = 10000 #highest value observed to not cause explosion: 10000
wind_field = models.WindModel(wind_region,int(wind_grid_density*aspect_ratio),
wind_grid_density,noise_gain=noise_gain,noise_damp=noise_damp,
noise_bandwidth=noise_bandwidth,Kx=Kx,Ky=Ky,
diff_eq=diff_eq,angle=constant_wind_angle)


# Set up plume model
centre_rel_diff_scale = 2.
# puff_release_rate = 0.001
puff_release_rate = 10
puff_spread_rate=0.005
puff_init_rad = 0.01
max_num_puffs=int(2e5)
# max_num_puffs=100

plume_model = models.PlumeModel(
    sim_region, source_pos, wind_field,simulation_time-t_start,
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
buffr = -300
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
u,v = u[0:full_size-1:shrink_factor,0:full_size-1:shrink_factor],\
    v[0:full_size-1:shrink_factor,0:full_size-1:shrink_factor]
x_origins,y_origins = wind_field.x_points,wind_field.y_points
x_origins,y_origins = x_origins[0:full_size-1:shrink_factor],\
    y_origins[0:full_size-1:shrink_factor]
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


def init():
    #do nothing
    pass


t = t_start
# Define animation update function
def update(i):
    global t, arrow_magn, shrink_factor, full_size
    for k in range(capture_interval):
        wind_field.update(dt)
        plume_model.update(dt)
        # raw_input()
        t+=dt
        print(t)

    velocity_field = wind_field.velocity_field
    u,v = velocity_field[:,:,0],velocity_field[:,:,1]
    u,v = u[0:full_size-1:shrink_factor,0:full_size-1:shrink_factor],\
        v[0:full_size-1:shrink_factor,0:full_size-1:shrink_factor]
    vector_field.set_UVC(u,v)
    x_wind,y_wind = scipy.cos(constant_wind_angle),scipy.sin(constant_wind_angle)
    wind_arrow.set_positions((xmin+(xmax-xmin)/2,ymax-0.2*(ymax-ymin)),
    (xmin+(xmax-xmin)/2+arrow_magn*x_wind,
    ymax-0.2*(ymax-ymin)+arrow_magn*y_wind))
    text ='{0} min {1} sec'.format(
    int(scipy.floor(abs(t/60.))),int(scipy.floor(abs(t)%60.)))
    timer.set_text(text)

    conc_array = array_gen.generate_single_array(plume_model.puffs)

    log_im = scipy.log(conc_array.T[::-1])
    cutoff_l = scipy.percentile(log_im[~scipy.isinf(log_im)],10)
    cutoff_u = scipy.percentile(log_im[~scipy.isinf(log_im)],99)

    conc_im.set_data(log_im)
    n = matplotlib.colors.Normalize(vmin=cutoff_l,vmax=cutoff_u)
    conc_im.set_norm(n)



    concStorer.store(conc_array.T[::-1])
    last = time.time()

    windStorer.store(velocity_field)
    plumeStorer.store(plume_model.puffs)

    return [conc_im]

# Run and save output to video
anim = FuncAnimation(fig, update, frames=int(
frame_rate*(simulation_time-t_start)/times_real_time),
init_func=init,repeat=False)

# plt.show()

#Save the animation to video
saved = anim.save('straight_wind.mp4', dpi=100, fps=frame_rate, extra_args=['-vcodec', 'libx264'])
# concStorer.finish_filling()
