import h5py
import json
import scipy
import itertools
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import sys
import time
from pompy.models import PlumeStorer,Rectangle
import dill as pickle
import pompy.processors as processors


class ImportedPlumes(object):

    def __init__(self,hdf5_file,array_z,array_dim_x,array_dim_y,puff_mol_amount,
    release_delay,box_approx=False,r_sq_max=20,epsilon=0.05,N=1e5):
        self.data = h5py.File(hdf5_file,'r')
        run_param = json.loads(self.data.attrs['jsonparam'])
        sim_region_tuple = run_param['simulation_region']
        self.sim_region = Rectangle(*sim_region_tuple)
        self.dt_store = run_param['dt_store']
        self.t_stop = run_param['simulation_time']
        self.run_param = run_param
        if box_approx:
            box_min,box_max = sim_region_tuple[1],sim_region_tuple[2]
            self.array_gen = processors.ConcentrationValueFastCalculator(
                box_min,box_max,r_sq_max,epsilon,puff_mol_amount,N)
        else:
            self.array_gen = processors.ConcentrationValueCalculator(puff_mol_amount)
        self.puff_array = self.data['puff_array']
        self.array_ends = self.data['array_end']
        self.release_delay  = release_delay
        self.box_approx = box_approx
    def puff_array_at_time(self,t):
        ind = int(scipy.floor((t+self.release_delay)/self.dt_store))
        try:
            array_end = self.array_ends[ind]
        except(ValueError):
            array_end = self.array_ends[ind-1]
        if array_end>0:
            try:
                return self.puff_array[ind,0:array_end,:]
            except(ValueError):
                return self.puff_array[ind-1,0:array_end,:]
        else:
            return []
    def value(self,t,xs,ys):
        #box approx True = use box method approximation to calc odor values
        #else False = the original pompy direct computation
        puff_array = self.puff_array_at_time(t)
        if self.box_approx:
            print('doing approx version')
        else:
            print('doing exact version')
        return self.array_gen.calc_conc_list(puff_array, xs, ys, z=0)

class ImportedConc(object):
#Note that each array here has been stored in the order that produces the right
#image (x goes top to bottom, y left to right)
    def __init__(self,hdf5_file,release_delay,cmap='Blues',):
        self.data = h5py.File(hdf5_file,'r')
        run_param = json.loads(self.data.attrs['jsonparam'])
        self.simulation_region = run_param['simulation_region']
        self.xmin,self.xmax,self.ymin,self.ymax = run_param['simulation_region']
        self.dt_store = run_param['dt_store']
        #concentration data stored here, x by y by t
        self.t_stop = run_param['simulation_time']
        self.cmap = cmap
        self.vmin,self.vmax = run_param['imshow_bounds']
        self.conc_array = self.data['conc_array']
        self.grid_size = scipy.shape(self.conc_array[0,:,:])
        self.release_delay = release_delay
    def array_at_time(self,t):
        ind = scipy.floor((t+self.release_delay)/self.dt_store)
        try:
            return self.conc_array[ind,:,:]
        except(ValueError):
            return self.conc_array[ind-1,:,:]

    def plot(self,t,vmin=None,vmax=None):
        array_at_time = self.array_at_time(t)
        if vmin==None:
            vmin, vmax = self.vmin,self.vmax
        image=plt.imshow(array_at_time, extent=self.simulation_region,
        cmap=self.cmap,vmin=vmin,vmax=vmax)
        return image

    def value(self,t,xs,ys):
        array_at_time = self.array_at_time(t)
        # plt.imshow(array_at_time, extent=self.simulation_region,
        # cmap=self.cmap,vmin=self.vmin,vmax=self.vmax)
        # print(scipy.shape(scipy.sum(array_at_time,0)))
        # print(scipy.where(scipy.sum(array_at_time,0)>0.01))
        # print(scipy.where(scipy.sum(array_at_time,1)>0.01))
        conc_list = scipy.zeros(scipy.size(xs))
        x_inds = scipy.floor(
        (xs-self.xmin)/(self.xmax-self.xmin)*self.grid_size[0])
        y_inds = scipy.floor(
        abs(ys-self.ymax)/(self.ymax-self.ymin)*self.grid_size[1])
        # print(x_inds,y_inds)
        # plt.show()
        for i,(x_ind,y_ind) in zip(range(len(xs)),zip(x_inds,y_inds)):
            try:
                conc_list[i] = array_at_time[y_ind,x_ind]
            except(IndexError):
                # print(x_ind,y_ind)
                conc_list[i]=0.
        return conc_list

    def get_image_params(self):
        return (self.vmin,self.vmax,self.cmap)

class ImportedWind(object):
    def __init__(self,hdf5_file,release_delay):
        self.data = h5py.File(hdf5_file,'r')
        run_param = json.loads(self.data.attrs['jsonparam'])
        self.run_param = run_param
        self.dt_store = run_param['dt_store']
        self.t_stop = run_param['simulation_time']
        self.x_points = scipy.array(run_param['x_points'])
        self.y_points = scipy.array(run_param['y_points'])
        self.velocity_field = self.data['velocity_field']
        self.evolving = True
        self.release_delay = release_delay


    def quiver_at_time(self,t):
        ind = scipy.floor((t+self.release_delay)/self.dt_store)
        try:
            velocity_field = self.velocity_field[ind,:,:,:]
        except(ValueError):
            velocity_field = self.velocity_field[ind-1,:,:,:]
        return scipy.array(velocity_field[:,:,0]),\
            scipy.array(velocity_field[:,:,1])

    def get_plotting_points(self):
        x_origins,y_origins = self.x_points,self.y_points
        coords = scipy.array(list(itertools.product(x_origins, y_origins)))
        x_coords,y_coords = coords[:,0],coords[:,1]
        return x_coords,y_coords

    def velocity_at_pos(self,t,x,y):
    #This will use the same interpolating method that pompy wind field uses
        us,vs = self.quiver_at_time(t)
        interp_u = interp.RectBivariateSpline(self.x_points,self.y_points,us)
        interp_v = interp.RectBivariateSpline(self.x_points,self.y_points,vs)
        return scipy.array([float(interp_u(x, y)),
                                 float(interp_v(x, y))])
    def value(self,t,x,y):
    #performs velocity_at_pos on an array of x,y coordinates
        if type(x)==scipy.ndarray:
            us,vs = self.quiver_at_time(t)
            interp_u = interp.RectBivariateSpline(self.x_points,self.y_points,us)
            interp_v = interp.RectBivariateSpline(self.x_points,self.y_points,vs)
            wind = scipy.array([
                [float(interp_u(x[i], y[i])),\
                float(interp_v(x[i], y[i]))] for i in range(len(x))
                ])
            return wind[:,0],wind[:,1]
        else:
            return self.velocity_at_pos(t,x,y)
