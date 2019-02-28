# -*- coding: utf-8 -*-
"""
Helper classes to process outputs of models.
"""

from __future__ import division

__authors__ = 'Matt Graham'
__license__ = 'MIT'

import math
import numpy as np
from collections import defaultdict
import itertools
import time

class ConcentrationValueCalculator(object):

    """
    Calculates odour concentration values at points in simulation region from
    puff property arrays.
    """

    def __init__(self, puff_molecular_amount):
        """
        Parameters
        ----------
        puff_mol_amount : float
            Molecular content of each puff (e.g. in moles or raw number of
            molecules). This is conserved as the puff is transported within
            the plume but the puff becomes increasingly diffuse as its radius
            grows due to diffusion.
        """
        # precompute constant used to scale Gaussian amplitude
        self._ampl_const = puff_molecular_amount / (8 * np.pi**3)**0.5

    def _puff_conc_dist(self, x, y, z, px, py, pz, r_sq):
        # calculate Gaussian puff concentration distribution
        return (
            self._ampl_const / r_sq**1.5 *
            np.exp(-((x - px)**2 + (y - py)**2 + (z - pz)**2) / (2 * r_sq))
        )

    def calc_conc_point(self, puff_array, x, y, z=0):
        """
        Calculate concentration at a single point.

        Parameters
        ----------
        puff_array : numpy-array-like of Puff objects
            Collection of currently alive puff instances at a particular
            time step which it is desired to calculate concentration field
            values from.
        x : float
            x-coordinate of point.
        y : float
            y-coordinate of point.
        z : float
            z-coordinate of point.
        """
        # filter for non-nan puff entries and separate properties for
        # convenience
        px, py, pz, r_sq = puff_array[~np.isnan(puff_array[:, 0]), :].T
        return self._puff_conc_dist(x, y, z, px, py, pz, r_sq).sum(-1)

    def calc_conc_list(self, puffs, x, y, z=0):
        """
        Calculate concentrations across a 1D list of points in a xy-plane.

        Parameters
        ----------
        puff_array : numpy-array-like of Puff objects
            Collection of currently alive puff instances at a particular
            time step which it is desired to calculate concentration field
            values from.
        x : (np) numpy-array-like of floats
            1D array of x-coordinates of points.
        y : (np) numpy-array-like of floats
            1D array of y-coordinates of points.
        z : float
            z-coordinate (height) of plane.
        """
        # filter for non-nan puff entries and separate properties for
        # convenience
        puffs_reshaped = puffs.reshape(-1,puffs.shape[-1])
        px, py, pz, r_sq = puffs_reshaped[~np.isnan(puffs_reshaped[:, 0]), :].T
        na = np.newaxis

        return self._puff_conc_dist(x[:, na], y[:, na], z, px[na, :],
                                    py[na, :], pz[na, :], r_sq[na, :]).sum(-1)

    def calc_conc_grid(self, puff_array, x, y, z=0):
        """
        Calculate concentrations across a 2D grid of points in a xy-plane.

        Parameters
        ----------
        puff_array : numpy-array-like of Puff objects
            Collection of currently alive puff instances at a particular
            time step which it is desired to calculate concentration field
            values from.
        x : (nx,ny) numpy-array-like of floats
            2D array of x-coordinates of grid points.
        y : (nx,ny) numpy-array-like of floats
            2D array of y-coordinates of grid points.
        z : float
            z-coordinate (height) of grid plane.
        """
        # filter for non-nan puff entries and separate properties for
        # convenience
        px, py, pz, r_sq = puff_array[~np.isnan(puff_array[:, 0]), :].T
        na = np.newaxis
        return self._puff_conc_dist(x[:, :, na], y[:, :, na], z,
                                    px[na, na, :], py[na, na, :],
                                    pz[na, na, :], r_sq[na, na, :]).sum(-1)
class ConcentrationValueFastCalculator(object):
        """
        Calculates odour concentration values at points in simulation region from
        puff property arrays, using O(N+M) approximation algorithm for speed improvement.
        """
        def __init__(self,box_min,box_max,r_sq_max,epsilon,puff_mol_amt,N):
                    # puff_spread_rate,trap_locs,wind_speed):
                    #bounds = box_min,box_max
            self.grid = ConcentrationGrid(box_min,box_max,r_sq_max,epsilon,N)
            self.box_min = box_min
            self.box_max = box_max
            self.neighbors_dct = defaultdict(list)
            self.all_boxes = list(itertools.product(range(self.grid.grid_max),\
                range(self.grid.grid_max)))
            for (i,j) in self.all_boxes:
            #Find all the boxes 1 box away from the target's box:
            #these sources considered
                self.neighbors_dct[(i,j)] = self.grid.obtain_box_neighbors(i,j)
            self.puff_mol_amt = puff_mol_amt
            self._ampl_const = puff_mol_amt / (8 * np.pi**3)**0.5
            self.max_puffs_per_box_est = int(np.ceil(2e5/(30)))
            # self.puff_spread_rate = puff_spread_rate
            # self.trap_locs = np.array(trap_locs)
            # self.wind_speed = wind_speed

        def compute_Gaussian(self,px,py,pz,r_sq,x,y):
            return (
                self._ampl_const / r_sq**1.5 *
                np.exp(-((x - px)**2 + (y - py)**2 + (0 - pz)**2) / (2 * r_sq))
            )

        def calc_conc_list(self, puffs, x, y, z=0):
            puffs_reshaped = puffs.reshape(-1,puffs.shape[-1])
            px, py, pz, r_sq = puffs_reshaped[~np.isnan(puffs_reshaped[:, 0]), :].T

            t1 = time.time()
            #(1) Box assignment step
            source_loc = np.array([px,py,pz]).T
            target_loc = np.array([x,y]).T
            target_values = np.zeros(len(x))
            t1a = time.time()
            source_boxes = self.grid.assign_box(source_loc[:,0:2])
            # print('find each sources box: '+str(time.time()-t1a))
            #version 0
            source_grid_dict = defaultdict(list)
            # t1aa = time.time()
            for source_box,source,source_r_sq in zip(source_boxes,source_loc,r_sq):
                #find what box it's in--value is a list: [source_x,source_y,r_sq]
                source_grid_dict[tuple(source_box)].append(
                    [source[0],source[1],source[2],source_r_sq])
            # print('option 1 time: '+str(time.time()-t1aa))


            # t1ab = time.time()
            # unique_source_boxes = np.unique(source_boxes,axis=0)
            # #print(np.shape(unique_source_boxes))
            # #print(np.shape(source_boxes))
            # boxes = self.all_boxes
            # source_grid_dict = dict.fromkeys(boxes,[])
            # # source_grid_dict1 = defaultdict(list)
            # for box in unique_source_boxes:
            #     #find what box it's in--value is a list: [source_x,source_y,r_sq]
            #     inds = np.where(np.sum((source_boxes==box),axis=1)==2)[0]
            #     source_set = np.hstack((source_loc[inds,:],r_sq[inds,np.newaxis]))
            #     source_grid_dict[tuple(box)] = source_set
            # print('option 2 time: '+str(time.time()-t1ab))
            #
            # t1ac = time.time()
            # unique_source_boxes = np.unique(source_boxes,axis=0)
            # boxes = self.all_boxes
            # source_grid_dict1 = dict.fromkeys(boxes,[])
            #
            # t1ad = time.time()
            # usb_inds,sb_inds = np.where(
            #     (unique_source_boxes[:,0].T[:,np.newaxis]==source_boxes.T[0,:])&
            #     (unique_source_boxes[:,1].T[:,np.newaxis]==source_boxes.T[1,:])
            #     )
            # print('np where time: '+str(time.time()-t1ad))
            # print('option 3a time: '+str(time.time()-t1ac))
            # for i,box in enumerate(unique_source_boxes):
            #     source_grid_dict1[tuple(box)] =  \
            #         source_boxes[sb_inds[usb_inds==i],:]
            #
            # print('option 3 time: '+str(time.time()-t1ac))



            # print(source_grid_dict.keys())
            # raw_input()
            # print(len(source_grid_dict.values()[1]))
            # print(len(source_grid_dict1.values()[1]))
            # raw_input()

            print('put each source in its box: '+str(time.time()-t1a))
            #do the same for all the targets -- also collect the target index
            target_boxes = self.grid.assign_box(target_loc)
            target_grid_dict = defaultdict(list)
            for target_box,target_loc,target_index in zip(
                target_boxes,target_loc,range(len(x))):
                target_grid_dict[tuple(target_box)].append(
                    [target_loc[0],target_loc[1],target_index])
            na = np.newaxis
            print('alg step 1 time: '+str(time.time()-t1))
            t2 = time.time()
            #(2) Within-box pairwise computation step
            #Loop through the grid boxes
            for (i,j) in list(itertools.product(range(
                self.grid.grid_max),range(self.grid.grid_max))):
            #Find all the boxes 1 box away from the target's box: these sources considered
                relevant_targets = target_grid_dict[(i,j)]
                # print(source_grid_dict.keys())
                if len(relevant_targets)>0: #only proceed if there are targets in the box
                    target_x,target_y,target_indices = np.array(relevant_targets).T
                    target_x,target_y = target_x[:,na],target_y[:,na]
                    relevant_sources = tuple(
                        source_grid_dict[neighbor] for neighbor in self.neighbors_dct[(i,j)]
                        if len(source_grid_dict[neighbor])>0)
                    #only proceed if there are sources in the neighbor set
                    if len(relevant_sources)>0:
                        source_x,source_y,source_z,r_sq = np.concatenate(relevant_sources,0).T
                        source_x,source_y,source_z,r_sq = source_x[na,:],source_y[na,:],source_z[na,:],r_sq[na,:]
                    #For targets, only those in the box, not the neighbor boxes
                        output_array =  self.compute_Gaussian(
                            source_x,source_y,source_z,r_sq,target_x,target_y).sum(1)
                        target_values[target_indices.astype(int)] = output_array
                    else:
                        pass
                else:
                    pass
            print('alg step 2 time: '+str(time.time()-t2))

            return target_values
        # def calc_conc_list(self, puffs,x, y, z=0):
        #
        #     #Trap locs: (traps x 2)
        #     na = np.newaxis
        #     puffs_reshaped = puffs.reshape(-1,puffs.shape[-1])
        #     px, py, pz, r_sq = puffs_reshaped[~np.isnan(puffs_reshaped[:, 0]), :].T
        #
        #     source_loc = np.array([px,py,pz]).T
        #     target_loc = np.array([x,y]).T
        #     target_values = np.zeros(len(x))
        #
        #     #Compute distance of each fly to each trap
        #
        #     fly_distances = np.sqrt(np.sum(np.square(
        #         target_loc.T[na,:,:]-self.trap_locs[:,:,na]),axis=1))
        #     #distances is (n traps x n flies)
        #
        #     #Send distances through a function that assigns them to a distance bin
        #     fly_distance_bins = np.floor(fly_distances/self.grid.R) #also (n traps x n flies)
        #
        #     bin_max = int(np.max(fly_distance_bins))
        #
        #     #Loop through the bins
        #     for i in range(1,bin_max):
        #     #the sources considered are the puffs with distance corresponding to bins
        #     #(i-1,i,i+1) except biggest and littlest
        #         puff_r_sq_min,puff_r_sq_max = np.array([i-1,i+1])* \
        #             (self.grid.R/self.wind_speed)*self.puff_spread_rate
        #         puffs_to_consider = (puffs[:,:,3] > puff_r_sq_min) & (
        #             puffs[:,:,3] < puff_r_sq_max) #inds of relevant puffs
        #         # print(np.shape(puffs[puffs_to_consider]))
        #
        #         #shape of puffs[puffs_to_consider] is puffs x 4
        #         source_x,source_y,source_z,r_sq = puffs[puffs_to_consider].T
        #         target_indices = (np.sum(fly_distance_bins==i,axis=0)>0)
        #         # print(np.sum(target_indices))
        #         if (np.sum(target_indices)>0) & (np.sum(puffs_to_consider)>0):
        #             # print(np.shape(target_loc[target_indices,:]))
        #             target_x,target_y = target_loc[target_indices,:].T
        #             target_x,target_y = target_x[:,na],target_y[:,na]
        #
        #             output_array =  self.compute_Gaussian(
        #                 source_x,source_y,source_z,r_sq,target_x,target_y).sum(1)
        #
        #             # print(np.shape(target_values[target_indices.astype(int)]))
        #             # print(np.shape(target_values[target_indices]))
        #             # print(np.shape(target_indices.astype(int)))
        #             # print(np.shape(output_array))
        #
        #             target_values[target_indices] = output_array
        #
        #     return target_values

        def calc_conc_display_grid(self, puffs,z=0,nx=1000,ny=1000):
            conc_array_locs_x,conc_array_locs_y = np.meshgrid(
                np.linspace(self.box_min,self.box_max,nx),
                    np.linspace(self.box_min,self.box_max,ny))
            return self.calc_conc_list(
                puffs, conc_array_locs_x.flatten(),conc_array_locs_y.flatten(), z).reshape((nx,ny))


class ConcentrationGrid(object):
    """
    A helper class to be used with ConcentrationValueFastCalculator;
    this is initialized at the beginning of the fly simulation and passed
    to the ConcentrationValueFastCalculator.

    Produces a grid on given square with unit size given by the
    largest s < R such that (1/R)*s = square width

    R, which depends on r_sq_max and epsilon, is the largest possible grid
    width such that
    in the worst case scenario, where all sources have variance r_sq_max,
    the sum total odor contribution to a given target
    from all sources outside the target's immediate neighbor boxes
    is less than epsilon.
    """
    def __init__(self,box_min,box_max,r_sq_max,epsilon,N):
        self.bounds = np.array([[box_min,box_max],[box_min,box_max]])
        self.R = np.sqrt(-1*np.log(epsilon*(r_sq_max**1.5)*np.sqrt(8*np.pi**3)/(N))*2*r_sq_max)
        n_boxes = int(math.ceil((box_max-box_min)/self.R))
        self.unit_width = 1/n_boxes
        self.grid_min = 0
        self.grid_max = n_boxes
        print('num boxes: '+str(n_boxes))


    def assign_box(self,location): #1D--combine to 2D
        return (np.floor((location-self.bounds[:,0])/self.R)).astype(int)


    def obtain_box_neighbors(self,x,y): #2d grid neighbors of a given grid coord
        if ((self.grid_min <= x <= self.grid_max-1) and (
            self.grid_min <= y <= self.grid_max-1)):
            xs,ys = np.meshgrid(
                np.array([x-1,x,x+1]),np.array([y-1,y,y+1]))
            neighbors = np.array([xs,ys])
            #this is an array that is 2 x 3 x 3
            if self.grid_max-1==x:
                neighbors = neighbors[:,:,:-1]
            if self.grid_min==x:
                neighbors = neighbors[:,:,1:]
            if self.grid_min==y:
                neighbors = neighbors[:,1:,:]
            if self.grid_max-1==y:
                neighbors = neighbors[:,:-1,:]
            neighbors = np.squeeze(np.reshape(neighbors,(2,-1,1)))
            return list(zip(neighbors[0,:],neighbors[1,:]))
        else:
            print('Grid neighbor error: provided box not in big box')
            sys.exit()


class ConcentrationArrayGenerator(object):

    """
    Produces odour concentration field arrays from puff property arrays.

    Instances of this class can take single or multiple arrays of puff
    properties outputted from a PlumeModel and process them to produce an
    array of the concentration values across the a specified region using
    a Gaussian model for the individual puff concentration distributions.

    Compared to the ConcentrationValueCalculator class, this class should be
    more efficient for calculating large concentration field arrays for
    real-time graphical display of odour concentration fields for example
    at the expense of (very) slightly less accurate values due to the
    truncation of spatial extent of each puff.

    Notes
    -----
    The returned array values correspond to the *point* concentration
    measurements across a regular grid of sampling points - i.e. the
    equivalent to convolving the true continuous concentration distribution
    with a regular 2D grid of Dirac delta / impulse functions. An improvement
    in some ways would be to instead calculate the integral of the
    concentration distribution over the (square) region around each grid point
    however this would be extremely computationally costly and due to the lack
    of a closed form solution for the integral of a Gaussian also potentially
    difficult to implement without introducing other numerical errors. An
    integrated field can be approximated with this class by generating an
    array at a higher resolution than required and then filtering with a
    suitable kernel and down-sampling.

    This implementation estimates the concentration distribution puff kernels
    with sub-grid resolution, giving improved accuracy at the cost of
    increased computational cost versus using a precomputed radial field
    aligned with the grid to compute kernel values or using a library of
    precomputed kernels.

    For cases where the array region cover the whole simulation region the
    computational cost could also be reduced by increasing the size of the
    region the array corresponds to outside of the simulation region such that
    when adding the puff concentration kernels to the concentration field
    array, checks do not need to be made to restrict to the overlapping region
    for puffs near the edges of the simulation region which have a
    concentration distribution which extends beyond its extents.
    """

    def __init__(self, array_xy_region, array_z, nx, ny, puff_mol_amount,
                 kernel_rad_mult=3):
        """
        Parameters
        ----------
        array_region : Rectangle
            Two-dimensional rectangular region defined in world coordinates
            over which to calculate the concentration field.
        array_z : float
            Height on the vertical z-axis at which to calculate the
            concentration field over.
        nx : integer
            Number of grid points to sample at across x-dimension.
        ny : integer
            Number of grid points to sample at across y-dimension.
        puff_mol_amount : float
            Molecular content of each puff (e.g. in moles or raw number of
            molecules). This is conserved as the puff is transported within
            the plume but the puff becomes increasingly diffuse as it's radius
            grows due to diffusion.
            (dimensionality:molecular amount)
        kernel_rad_mult : float
            Multiplier used to determine to within how many puff radii from
            the puff centre to truncate the concentration distribution
            kernel calculated to. The default value of 3 will truncate the
            Gaussian kernel at (or above) the point at which the concentration
            has dropped to 0.004 of the peak value at the puff centre.
        """
        self.array_xy_region = array_xy_region
        self.array_z = array_z
        self.nx = nx
        self.ny = ny
        self._dx = array_xy_region.w / nx  # calculate x grid point spacing
        self._dy = array_xy_region.h / ny  # calculate y grid point spacing
        # precompute constant used to scale Gaussian kernel amplitude
        self._ampl_const = puff_mol_amount / (8*np.pi**3)**0.5
        self.kernel_rad_mult = kernel_rad_mult

    def puff_kernel(self, shift_x, shift_y, z_offset, r_sq, even_w, even_h):
        # kernel is truncated to min +/- kernel_rad_mult * effective puff
        # radius from centre i.e. Gaussian kernel with >= kernel_rad_mult *
        # standard deviation span
        # (effective puff radius is (r_sq - (z_offset/k_r_mult)**2)**0.5 to
        # account for the cross sections of puffs with centres out of the
        # array plane being 'smaller')
        # the truncation will introduce some errors - an improvement would
        # be to use some form of windowing e.g. Hann or Hamming window
        shape = (2*(r_sq*self.kernel_rad_mult**2 - z_offset**2)**0.5 /
                 np.array([self._dx, self._dy]))
        # depending on whether centre is on grid points or grid centres
        # kernel dimensions will need to be forced to odd/even respectively
        shape[0] = self.round_up_to_next_even_or_odd(shape[0], even_w)
        shape[1] = self.round_up_to_next_even_or_odd(shape[1], even_h)
        # generate x and y grids with required shape
        [x_grid, y_grid] = 0.5 + np.mgrid[-shape[0]/2:shape[0]/2,
                                          -shape[1]/2:shape[1]/2]
        # apply shifts to correct for offset of true centre from nearest
        # grid-point / centre
        x_grid = x_grid * self._dx + shift_x
        y_grid = y_grid * self._dy + shift_y
        # compute square radial field
        r_sq_grid = x_grid**2 + y_grid**2 + z_offset**2
        # output scaled Gaussian kernel
        return self._ampl_const / r_sq**1.5 * np.exp(-r_sq_grid / (2 * r_sq))

    @staticmethod
    def round_up_to_next_even_or_odd(value, to_even):
        # Returns value rounded up to first even number >= value if
        # to_even==True and to first odd number >= value if to_even==False.
        value = math.ceil(value)
        if to_even:
            if value % 2 == 1:
                value += 1
        else:
            if value % 2 == 0:
                value += 1
        return value

    def generate_single_array(self, puffs):

        # initialise concentration array
        conc_array = np.zeros((self.nx, self.ny))
        # loop through all the puffs
        puffs_reshaped = puffs.reshape(-1,puffs.shape[-1])
        # print(np.shape(puffs_reshaped[~np.isnan(puffs_reshaped[:, 0]), :].T))
        for (puff_x, puff_y, puff_z, puff_r_sq) in \
            (puffs_reshaped[~np.isnan(puffs_reshaped[:, 0]), :]):
            # to begin with check this a real puff and not a placeholder nan
            # entry as puff arrays may have been pre-allocated with nan
            # at a fixed size for efficiency and as the number of puffs
            # existing at any time interval is variable some entries in the
            # array will be unallocated, placeholder entries should be
            # contiguous (i.e. all entries after the first placeholder will
            # also be placeholders) therefore break out of loop completely
            # if one is encountered
            if np.isnan(puff_x):
                break
            # check puff centre is within region array is being calculated
            # over otherwise skip
            if not self.array_xy_region.contains(puff_x, puff_y):
                continue
            # finally check that puff z-coordinate is within
            # kernel_rad_mult*r_sq of array evaluation height otherwise skip
            puff_z_offset = (self.array_z - puff_z)
            if abs(puff_z_offset) / puff_r_sq**0.5 > self.kernel_rad_mult:
                continue
            # calculate (float) row index corresponding to puff x coord
            p = (puff_x - self.array_xy_region.x_min) / self._dx
            # calculate (float) column index corresponding to puff y coord
            q = (puff_y - self.array_xy_region.y_min) / self._dy
            # calculate nearest integer or half-integer row index to p
            u = math.floor(2 * p + 0.5) / 2
            # calculate nearest integer or half-integer row index to q
            v = math.floor(2 * q + 0.5) / 2
            # generate puff kernel array of appropriate scale and taking
            # into account true centre offset from nearest half-grid
            # points (u,v)
            kernel = self.puff_kernel((p - u) * self._dx, (q - v) * self._dy,
                                      puff_z_offset, puff_r_sq,
                                      u % 1 == 0, v % 1 == 0)
            # compute row and column slices for source kernel array and
            # destination concentration array taking in to the account
            # the possibility of the kernel being partly outside the
            # extents of the destination array
            (w, h) = kernel.shape
            r_rng_arr = slice(int(max(0, u - w / 2.)),
                              int(max(min(u + w / 2., self.nx), 0)))
            c_rng_arr = slice(int(max(0, v - h / 2.)),
                              int(max(min(v + h / 2., self.ny), 0)))
            r_rng_knl = slice(int(max(0, -u + w / 2.)),
                              int(min(-u + w / 2. + self.nx, w)))
            c_rng_knl = slice(int(max(0, -v + h / 2.)),
                              int(min(-v + h / 2. + self.ny, h)))
            # add puff kernel values to concentration field array
            conc_array[r_rng_arr, c_rng_arr] += kernel[r_rng_knl, c_rng_knl]
        return conc_array

    def generate_multiple_arrays(self, puff_arrays):
        """
        Generates multiple concentration field arrays from a sequence of
        arrays of puff properties.
        """
        conc_arrays = []
        for puff_array in puff_arrays:
            conc_arrays.append(self.generate_single_frame(puff_array))
        return conc_arrays
