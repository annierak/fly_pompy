# -*- coding: utf-8 -*-
"""
Implementations of puff-based plume model components.
"""

from __future__ import division

__authors__ = 'Matt Graham'
__license__ = 'MIT'

import numpy as np
import scipy
import scipy.interpolate as interp
import scipy.sparse
import utility
import datetime
import h5_logger
import os
import time

class Puff(object):
    """
    Lightweight container for the properties of a single odour puff.

    Implemented with slots to improve memory management when a large number
    of puffs are being simulated. Squared radius stored rather than radius
    as puff growth model and concentration distribution models both use
    squared radius hence this minimises needless exponent operations.
    """

    __slots__ = ["x", "y", "z", "r_sq"]

    def __init__(self, x, y, z, r_sq):
        self.x = x
        self.y = y
        self.z = z

        self.r_sq = r_sq

    def __iter__(self):
        for field_name in self.__slots__:
            yield getattr(self, field_name)


class InvalidRectangleCoordinateError(Exception):
    """Raised when setting a rectangle coordinate to an invalid value."""
    pass


class Rectangle(object):

    """
    Axis-aligned rectangle defined by two points (x_min, y_min) and
    (x_max, y_max) with it required that x_max > x_min and y_max > y_min.
    """

    def __init__(self, x_min, y_min, x_max, y_max):
        """
        Parameters
        ----------
        x_min : float
            x-coordinate of bottom-left corner of rectangle.
        y_min : float
            x-coordinate of bottom-right corner of rectangle.
        x_max : float
            x-coordinate of top-right corner of rectangle.
        y_max : float
            y-coordinate of top-right corner of rectangle.
        """
        try:
            if float(x_min) >= float(x_max):
                raise InvalidRectangleCoordinateError('Rectangle x_min must \
                                                       be < x_max.')
            if float(y_min) >= float(y_max):
                raise InvalidRectangleCoordinateError('Rectangle y_min must \
                                                       be < y_max.')
        except ValueError as e:
            raise InvalidRectangleCoordinateError(
                'Rectangle coordinates must be numeric ({0}).'.format(e))
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    @property
    def w(self):
        """Width of rectangle (i.e. distance covered on x-axis)."""
        return self.x_max - self.x_min

    @property
    def h(self):
        """Height of rectangle (i.e. distance covered on y-axis)."""
        return self.y_max - self.y_min

    def as_tuple(self):
        """Tuple representation of Rectangle (x_min, y_min, x_max, y_max)."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def contains(self, x, y):
        """
        Tests whether the supplied position, an (x,y) pair, is contained
        within the region defined by this Rectangle object and returns
        True if so and False if not.
        """
        return (x >= self.x_min and x <= self.x_max and
                y >= self.y_min and y <= self.y_max)


class InvalidSourcePositionError(Exception):
    """Raised when an invalid source position is specified."""
    pass


class InvalidCentreRelDiffScaleError(Exception):
    """
    Raised when an invalid centre-line relative diffusion scale is
    specified.
    """
    pass


class PlumeModel(object):

    """
    Puff-based odour plume dispersion model from Farrell et. al. (2002).

    The odour plume is modelled as a series of odour puffs which are released
    from a fixed source position. The odour puffs are dispersed by a modelled
    2D wind velocity field plus a white noise process model of mid-scale
    puff mass diffusion relative to the plume centre line. The puffs also
    spread in size over time to model fine-scale diffusive processes.
    """

    def __init__(self, sim_region, source_pos, wind_model, simulation_time,
                model_z_disp=True,
                 centre_rel_diff_scale=2., puff_init_rad=0.03,
                 puff_spread_rate=0.001, puff_release_rate=10,
                 init_num_puffs=10, max_num_puffs=1000, prng=np.random):
        """
        Parameters
        ----------
        sim_region : Rectangle
            2D rectangular region of space over which the simulation is
            conducted. This should be the same simulation region as defined
            for the wind model.
        source_pos : float sequence
            (x,y,z) coordinates of the fixed source position within the
            simulation region from which puffs are released. If a length 2
            sequence is passed, the z coordinate will be set a default of 0.
            (dimensionality: length)
        wind_model : WindModel
            Dynamic model of the large scale wind velocity field in the
            simulation region.
        model_z_disp : boolean
            Whether to model dispersion of puffs from plume centre-line in
            z direction. If set True then the puffs will be modelled as
            dispersing in the vertical direction by a random walk process
            (the wind model is limited to 2D hence the vertical wind speed
            is assumed to be zero), if set False the puff z-coordinates will
            not be updated from their initial value of 0.
        centre_rel_diff_scale : float or float sequence
            Scaling for the stochastic process used to model the centre-line
            relative diffusive transport of puffs. Either a single float
            value of isotropic diffusion in all directions, or one of a pair
            of values specifying different scales for the x and y directions
            respectively if model_z_disp=False or a triplet of values
            specifying different scales for x, y and z scales respectively if
            model_z_disp=True.
            (dimensionality: length/time^0.5)
        puff_init_rad: float
            Initial radius of the puffs.
            (dimensionality: length)
        puff_spread_rate : float
            Constant which determines the rate at which the odour puffs
            increase in size as time progresses.
            (dimensionality: length^2/time)
        puff_release_rate : float
            Mean rate at which new puffs are released into the plume. Puff
            release is modelled as a stochastic Poisson process, with each
            puff released assumed to be independent and the mean release rate
            fixed.
            (dimensionality: count/time)
        init_num_puffs : integer
            Initial number of puffs to release at the beginning of the
            simulation.
        max_num_puffs : integer
            Maximum number of puffs to permit to be in existence
            simultaneously within model, used to limit memory and processing
            requirements of model. This parameter needs to be set carefully
            in relation to the puff release rate and simulation region size
            as if too small it will lead to a breaks in puff release when the
            number of puffs remaining in the simulation region reaches the
            limit.
        prng : RandomState
            Pseudo-random number generator to use in generating input noise
            for puff centre-line relative dispersion random walk and puff
            release Poisson processes. If no value is set (default) the
            numpy.random global generator is used however a specific
            RandomState can be set if it is desired to have reproducible
            output.
        """
        #Save a param dictionary to access creation parameters after
        #the plume model is used and stored.
        l = locals()
        self.param = dict([(k,l[k]) for k in ['model_z_disp',
                     'centre_rel_diff_scale', 'puff_init_rad',
                     'puff_spread_rate', 'puff_release_rate',
                     'init_num_puffs', 'max_num_puffs']])
        self.sim_region = sim_region
        self.wind_model = wind_model
        self.source_pos = source_pos
        self.prng = prng
        self.model_z_disp = model_z_disp
        self.unique_sources = len(source_pos[0,:])
        self._vel_dim = 3 if model_z_disp else 2
        if (model_z_disp and hasattr(centre_rel_diff_scale, '__len__') and
                len(centre_rel_diff_scale) == 2):
            raise InvalidCentreRelDiffScaleError('When model_z_disp=True, \
                                                  len(centre_rel_diff_scale) \
                                                  must be 1 or 3')
        self.centre_rel_diff_scale = centre_rel_diff_scale
        for i in range(self.unique_sources):
            if not sim_region.contains(source_pos[0,i], source_pos[1,i]):
                raise InvalidSourcePositionError('Specified source (x,y) \
                                              position must be within \
                                              simulation region.')
        # default to zero height source
        self.source_z = 0
        if len(source_pos) == 3:
            self.source_z = source_pos[2]
        self.puff_init_rad = puff_init_rad
        self._new_puff_params = (source_pos[0], source_pos[1], self.source_z,
                                 puff_init_rad**2)
        self.puff_spread_rate = puff_spread_rate
        self.puff_release_rate = puff_release_rate
        self.max_num_puffs = max_num_puffs
        # 1/25/2019: the puff list is an array of size
        # num_traps x (1.1 x simulation_time x release rate) x 4 (data points per puff)
        self.puffs = np.full((self.unique_sources,int(np.ceil(1.2*self.puff_release_rate*
            simulation_time)),4),np.nan)
        for j in range(self.unique_sources):
            self.puffs[j,0:init_num_puffs,:] = \
                np.ones((init_num_puffs,1))*np.array([[source_pos[0,j],source_pos[1,j], \
                    self.source_z,puff_init_rad**2]])
        self.last_puff_ind = init_num_puffs


    def update(self, dt):
        """Perform time-step update of plume model with Euler integration."""
        # add more puffs (stochastically) if enough capacity
        # if len(self.puffs) < self.max_num_puffs*self.unique_sources:
        #UPDATE 7/17/18: I am REMOVING the puff cap.
            # puff release modelled as Poisson process at fixed mean rate
            # with number to release clipped if it would otherwise exceed
            # the maximum allowed
        #Right now the num to release is same across traps
        num_to_release = self.prng.poisson(self.puff_release_rate*dt)
        for j in range(self.unique_sources):
            self.puffs[j,self.last_puff_ind:self.last_puff_ind+num_to_release,:] = \
                np.ones((num_to_release,1))*np.array([[self.source_pos[0,j],self.source_pos[1,j], \
                    self.source_z,self.puff_init_rad**2]])

        self.last_puff_ind +=num_to_release
        if dt ==0.01:
            noise_scale = 1
        elif dt==0.25:
            noise_scale = 0.3
        else:
            print('diffusion not programmed for this dt value')
        puffs_active = ~np.isnan(self.puffs)
        num_active = np.sum(puffs_active[:,:,0])
        print(str(num_active)+' puffs active')
        # print(np.shape(puffs_active))
        #traps by puffs
        # interpolate wind velocity at Puff positions from wind model grid
        # assuming zero wind speed in vertical direction if modelling
        # z direction dispersion
        wind_vel = self.wind_model.velocity_at_pos(
            self.puffs[:,:,0][puffs_active[:,:,0]],
                self.puffs[:,:,1][puffs_active[:,:,1]])
        # print(np.shape(puffs_active[:,:,0]))
        # print(np.shape(self.puffs[puffs_active[:,:,0]]))
        # print(np.shape(self.puffs[:,:,0][puffs_active[:,:,0]]))
        # print(wind_vel)
        # print(np.shape(wind_vel))
        if self._vel_dim>2:
            # print('here')
            wind_vel = np.hstack((wind_vel,np.zeros(num_active)[:,np.newaxis]))
        # print(np.shape(wind_vel))
        # approximate centre-line relative puff transport velocity
        # component as being a (Gaussian) white noise process scaled by
        # constants
        filament_diff_vel = noise_scale*(self.prng.normal(size=(num_active,self._vel_dim)) *
            self.centre_rel_diff_scale)
        #********** heyyyyyyyyy how about you  make this line above not dependent on the time step

        vel = wind_vel + filament_diff_vel

        # update puff position using Euler integration
        self.puffs[:,:,0][puffs_active[:,:,0]] += vel[:,0] * dt
        self.puffs[:,:,1][puffs_active[:,:,1]] += vel[:,1] * dt
        if self.model_z_disp:
            self.puffs[:,:,2][puffs_active[:,:,2]] += vel[:,2] * dt
        # update puff size using Euler integration with second puff
        # growth model described in paper
        self.puffs[:,:,3][puffs_active[:,:,3]] += self.puff_spread_rate * dt

        # num_traps  x (1.1 x simulation_time x release rate) x 4 (data points per puff)

        #set puffs that are not in the simulation region back to nans
        left_region = (np.abs(self.puffs[:,:,0])>self.sim_region.x_max) | \
            (np.abs(self.puffs[:,:,1])>self.sim_region.y_max) #size num_traps x (1.1 x simulation_time x release rate)
        self.puffs[left_region] = np.nan
        # raw_input()

    @property
    def puff_array(self):
        """
        Returns a numpy array of the properties of the simulated puffs.

        Each row corresponds to one puff with the first column containing the
        puff position x-coordinate, the second the y-coordinate, the third
        the z-coordinate and the fourth the puff squared radius.
        """
        return np.array([tuple(puff) for puff in self.puffs])


class WindModel(object):

    """
    Wind velocity model to calculate advective transport of odour.

    A 2D approximation is used as described in the paper, with the wind
    velocities calculated over a regular 2D grid of points using a finite
    difference method. The boundary conditions at the edges of the simulated
    region are for both components of the velocity field constant mean values
    plus coloured noise. For each of the field components these are calculated
    for the four corners of the simulated region and then linearly
    interpolated over the edges.


    """

    def __init__(self, sim_region, nx=15, ny=15,  Kx=2.,
                 Ky=2., noise_gain=5., noise_damp=0.2, noise_bandwidth=0.2,
                 noise_rand=np.random,EmpiricalWindField=None,diff_eq=True,
                 angle=None,mag=1):
        """
        Parameters
        ----------
        sim_region : Rectangle
            Two-dimensional rectangular region over which to model wind
            velocity field.
        nx : integer
            Number of grid points in x direction.
        ny : integer
            Number of grid points in y direction.

        Kx : float or array_like
            Diffusivity constant in x direction. Either a single scalar value
            across the whole simulated region or an array of size (nx, ny)
            defining different values for each grid point.
            (dimensionality: length^2/time)
        Ky : float or array_like
            Diffusivity constant in y direction. Either a single scalar value
            across the whole simulated region or an array of size (nx, ny)
            defining different values for each grid point.
            (dimensionality: length^2/time)
        noise_gain : float
            Input gain constant for boundary condition noise generation.
            (dimensionless)
        noise_damp : float
            Damping ratio for boundary condition noise generation.
            (dimensionless)
        noise_bandwidth : float
            Bandwidth for boundary condition noise generation.
            (dimensionality: angular measure/time)
        noise_rand : RandomState : float
            Pseudo-random number generator to use in generating input noise.
            Defaults to numpy.random global generator however a specific
            RandomState can be set if it is desired to have reproducible
            output.
        diff_eq : If false, just reduces this object to spitting out
        empirical wind values.
        """
        self.diff_eq = diff_eq
        self.EmpiricalWindField = EmpiricalWindField
        self.angle = angle
        self.mag = mag
        if (self.angle==None) and (self.EmpiricalWindField==None):
            raise ValueError('Wind model object requires either a constant wind angle or an EmpiricalWindField data object')
        # store grid parameters interally
        self._dx = abs(sim_region.w) / (nx-1)  # x grid point spacing
        self._dy = abs(sim_region.h) / (ny-1)  # y grid point spacing
        self.nx = nx
        self.ny = ny
        # precompute constant coefficients in PDE for efficiency
        self._Bx = Kx / (2.*self._dx**2)
        self._By = Ky / (2.*self._dy**2)
        self._C = 2. * (self._Bx + self._By)

        # initialise wind velocity field to mean values
        # +2s are to account for boundary grid points
        if self.EmpiricalWindField is not None:
            self.u_av,self.v_av = self.EmpiricalWindField.current_value()
        else:
            self.u_av,self.v_av = self.mag*np.cos(self.angle),self.mag*np.sin(self.angle)
        self._u = np.ones((nx+2, ny+2)) * self.u_av
        self._v = np.ones((nx+2, ny+2)) * self.v_av
        # create views on to field interiors (i.e. not including boundaries)
        # for notational ease - note this does NOT copy any data
        self._u_int = self._u[1:-1, 1:-1]
        self._v_int = self._v[1:-1, 1:-1]
        # set coloured noise generator for applying boundary condition
        # need to generate coloured noise samples at four corners of boundary
        # for both components of the wind velocity field so (2,8) state
        # vector (2 as state includes first derivative)
        self.noise_gen = ColouredNoiseGenerator(np.zeros((2, 8)), noise_damp,
                                                noise_bandwidth, noise_gain,
                                                noise_rand)
        # preassign array of corner means values
        self._corner_means = np.array([self.u_av, self.v_av]).repeat(4)
        # precompute linear ramp arrays with size of boundary edges for
        # linear interpolation of corner values
        self._rx = np.linspace(0., 1., nx+2)
        self._ry = np.linspace(0., 1., ny+2)
        # set up cubic spline interpolator for calculating off-grid wind
        # velocity field values
        self._x_points = np.linspace(sim_region.x_min, sim_region.x_max, nx)
        self._y_points = np.linspace(sim_region.y_min, sim_region.y_max, ny)
        self._set_interpolators()

    def _set_interpolators(self):
        """ Set spline interpolators using current velocity fields."""
        self._interp_u = interp.RectBivariateSpline(self.x_points,
                                                    self.y_points,
                                                    self._u_int)
        self._interp_v = interp.RectBivariateSpline(self.x_points,
                                                    self.y_points,
                                                    self._v_int)

    @property
    def x_points(self):
        """1D array of the range of x-coordinates of simulated grid points."""
        return self._x_points

    @property
    def y_points(self):
        """1D array of the range of y-coordinates of simulated grid points."""
        return self._y_points

    @property
    def velocity_field(self):
        """Current calculated velocity field across simulated grid points."""
        if self.diff_eq:
            return np.dstack((self._u_int, self._v_int))
        else:
            velocity_field = scipy.zeros((
            scipy.shape(self._u_int)[0],
            scipy.shape(self._u_int)[1],
            2))
            velocity_field[:,:,0] = self.u_av
            velocity_field[:,:,1] = self.v_av
            return velocity_field

    def velocity_at_pos(self, x, y):
        """
        Calculates the components of the velocity field at arbitrary point
        in the simulation region using a bivariate spline interpolation over
        the calculated grid point values.

        Parameters
        ----------
        x : float
            x-coordinate of the point to calculate the velocity at.
            (dimensionality: length)
        y : float
            y-coordinate of the point to calculate the velocity at.
            (dimensionality: length)

        Returns
        -------
        vel : array_like
            Velocity field (2D) values evaluated at specified point(s).
            (dimensionality: length/time)
        """
        if self.diff_eq:
            return np.array([self._interp_u.ev(x, y),
                         self._interp_v.ev(x, y)]).T
        else:
            return scipy.array([self.u_av,self.v_av]).T

    def update(self, dt):
        """
        Updates wind velocity field values using finite difference
        approximations for spatial derivatives and Euler integration for
        time-step update.

        Parameters
        ----------
        dt : float
            Simulation time-step.
            (dimensionality: time)
        """

        #update corner means to reflect changing observed wind
        if self.EmpiricalWindField is not None:
            self.u_av,self.v_av = self.EmpiricalWindField.current_value()
        else:
            self.u_av,self.v_av = np.cos(self.angle),np.sin(self.angle)
        if self.diff_eq:
            self._corner_means = np.array([self.u_av, self.v_av]).repeat(4)

            # update boundary values
            self._apply_boundary_conditions(dt)
            # initialise wind speed derivative arrays
            du_dt = np.zeros((self.nx, self.ny))
            dv_dt = np.zeros((self.nx, self.ny))
            # approximate spatial first derivatives with centred finite difference
            # equations for both components of wind field
            du_dx, du_dy = self._centred_first_derivs(self._u)
            dv_dx, dv_dy = self._centred_first_derivs(self._v)
            # calculate centred first sums i.e. sf_x = f(x+dx,y)+f(x-dx,y) and
            # sf_y = f(x,y+dy)-f(x,y-dy) as first step in approximating spatial
            # second derivatives with second order finite difference equations
            #   d2f/dx2 ~ [f(x+dx,y)-2f(x,y)+f(x-dx,y)] / (dx*dx)
            #           = [sf_x-2f(x,y)] / (dx*dx)
            #   d2f/dy2 ~ [f(x,y+dy)-2f(x,y)+f(x,y-dy)] / (dy*dy)
            #           = [sf_y-2f(x,y)] / (dy*dy)
            # second finite differences are not computed in full as the common
            # f(x,y) term in both expressions can be extracted out to reduce
            # the number of +/- operations required
            su_x, su_y = self._centred_first_sums(self._u)
            sv_x, sv_y = self._centred_first_sums(self._v)
            # use finite difference method to approximate time derivatives across
            # simulation region interior from defining PDEs
            #     du/dt = -(u*du/dx + v*du/dy) + 0.5*Kx*d2u/dx2 + 0.5*Ky*d2u/dy2
            #     dv/dt = -(u*dv/dx + v*dv/dy) + 0.5*Kx*d2v/dx2 + 0.5*Ky*d2v/dy2
            du_dt = (-self._u_int * du_dx - self._v_int * du_dy +
                     self._Bx * su_x + self._By * su_y -
                     self._C * self._u_int)
            dv_dt = (-self._u_int * dv_dx - self._v_int * dv_dy +
                     self._Bx * sv_x + self._By * sv_y -
                     self._C * self._v_int)
            # perform update with Euler integration
            self._u_int += du_dt * dt
            self._v_int += dv_dt * dt
            # update spline interpolators
            self._set_interpolators()

    def _apply_boundary_conditions(self, dt):
        """Applies boundary conditions to wind velocity field."""
        # update coloured noise generator
        self.noise_gen.update(dt)
        # extract four corner values for each of u and v fields as component
        # mean plus current noise generator output
        (u_tl, u_tr, u_bl, u_br, v_tl, v_tr, v_bl, v_br) = \
            self.noise_gen.output + self._corner_means
        # linearly interpolate along edges
        self._u[:, 0] = u_tl + self._rx * (u_tr - u_tl)   # u top edge
        self._u[:, -1] = u_bl + self._rx * (u_br - u_bl)  # u bottom edge
        self._u[0, :] = u_tl + self._ry * (u_bl - u_tl)   # u left edge
        self._u[-1, :] = u_tr + self._ry * (u_br - u_tr)  # u right edge
        self._v[:, 0] = v_tl + self._rx * (v_tr - v_tl)   # v top edge
        self._v[:, -1] = v_bl + self._rx * (v_br - v_bl)  # v bottom edge
        self._v[0, :] = v_tl + self._ry * (v_bl - v_tl)   # v left edge
        self._v[-1, :] = v_tr + self._ry * (v_br - v_tr)  # v right edge

    def _centred_first_derivs(self, f):
        """Calculates centred first difference derivative approximations."""
        return ((f[2:, 1:-1] - f[0:-2, 1:-1]) / (2 * self._dx),
                (f[1:-1, 2:] - f[1:-1, 0:-2]) / (2 * self._dy))

    def _centred_first_sums(self, f):
        """Calculates centred first sums."""
        return (f[2:, 1:-1] + f[0:-2, 1:-1]), (f[1:-1, 2:]+f[1:-1, 0:-2])





class ColouredNoiseGenerator(object):

    """
    Generates a coloured noise output via Euler integration of a state space
    system formulation.
    """

    def __init__(self, init_state, damping, bandwidth, gain,
                 prng=np.random):
        """
        Parameters
        ----------
        init_state : array_like
            The initial state of system, must be of shape (2,n) where n is
            the size of the noise vector to be produced. The first row
            sets the initial values and the second the initial first
            derivatives.
        damping : float
            Damping ratio for the system, affects system stability, values of
            <1 give an underdamped system, =1 a critically damped system and
            >1 an overdamped system.
            (dimensionless)
        bandwidth : float
            Bandwidth or equivalently undamped natural frequency of system,
            affects system reponsiveness to variations in (noise) input.
            (dimensionality = angular measure / time)
        gain : float
            Input gain of system, affects scaling of (noise) input.
            (dimensionless)
        prng : RandomState
            Pseudo-random number generator to use in generating input noise.
            Defaults to numpy.random global generator however a specific
            RandomState can be set if it is desired to have reproducible
            output.
        """
        # set up state space matrices
        self._A = np.array([[0., 1.],
                            [-bandwidth**2, -2. * damping * bandwidth]])
        self._B = np.array([[0.], [gain * bandwidth**2]])
        # initialise state
        self._x = init_state
        self.prng = prng

    @property
    def output(self):
        """Coloured noise output."""
        return self._x[0, :]

    def update(self, dt):
        """Updates state of noise generator."""
        # get normal random input
        u = self.prng.normal(size=(1, self._x.shape[1]))
        # calculate state time derivative with state space equation
        dx_dt = self._A.dot(self._x) + self._B * u
        # apply update with Euler integration
        self._x += dx_dt * dt

class EmpiricalWindField(object):
    def __init__(self,wind_data_file,wind_dt,dt,t_start):
        self.wind_dt = wind_dt
        wind_dct = utility.process_wind_data(wind_data_file,0,wind_dt=5)
        self.wind_speed_vec = scipy.array(wind_dct['wind_speed'])
        self.wind_angle_vec = scipy.array(wind_dct['wind_angle'])
        if t_start<0:
            self.update_counter = int(t_start/dt)
        else:
            self.update_counter = 0
        self.current_wind_speed = self.wind_speed_vec[0]
        self.current_wind_angle = self.wind_angle_vec[0]
    def update(self,dt):
        t=dt*self.update_counter
        print(t)
        if t>=0:
            index = int(scipy.floor(t/self.wind_dt))
        else:
            index = 0
        self.current_wind_speed = self.wind_speed_vec[index]
        self.current_wind_angle = self.wind_angle_vec[index]
        self.update_counter +=1
    def current_value(self):
        wind_vel_x = self.current_wind_speed*scipy.cos(self.current_wind_angle)
        wind_vel_y = self.current_wind_speed*scipy.sin(self.current_wind_angle)
        return wind_vel_x,wind_vel_y

class PlumeStorer(object):
    def __init__(self,plume_model,dt_store,t_stop):
        self.sim_region = plume_model.sim_region.as_tuple()
        self.dt_store = dt_store
        #store the puffs at every time step, t x puffs x 4 (infos per puff)
        n = datetime.datetime.utcnow()
        self.filename = 'puffObject{0}.{1}-{2}:{3}'.format(
        n.month,n.day,n.hour,n.minute)
        self.hdf5_filename = self.filename+'.hdf5'
        num_steps = int(t_stop/dt_store)
        run_param = {
        'num_steps': num_steps, 'dt_store':dt_store,
        'simulation_region': self.sim_region,
        'simulation_time':t_stop}
        run_param.update(plume_model.param)
        self.logger = h5_logger.H5Logger(self.hdf5_filename,param_attr=run_param)
        self.anticipated_puffs = plume_model.max_num_puffs
        self.puff_array_ends = scipy.full(
        int(scipy.ceil(t_stop/dt_store)),scipy.nan
        )
        self.num_traps = np.shape(plume_model.puffs)[0]
    def store(self,puffs):
        puffs_active = ~np.isnan(puffs)
        num_active = int(np.sum(puffs_active[:,:,0]))
        buffered_puff_array = scipy.full((
            self.num_traps,
            int(np.ceil(self.anticipated_puffs/(np.shape(puffs)[0]))),
            4),
            scipy.nan)
        # time.sleep(1)
        array_end=0
        if num_active>0:
            # print(num_active)
            # array_end = int(np.ceil(num_active/(np.shape(puffs)[0]))) #This is wrong
            array_end = np.max(np.sum(puffs_active[:,:,0],axis=1))
            array_start = np.where(np.sum(puffs_active[:,:,0],axis=0)>0)[0][0]
            buffered_puff_array[:,array_start:array_end,:] = puffs[:,array_start:array_end,:]#.reshape(
                # np.shape(buffered_puff_array[:,0:array_end,:]))
            # self.puffs[:,:,0][puffs_active[:,:,0]]

        data = {'puffs':buffered_puff_array,'array_end':array_end}
        self.logger.add(data)

class ConcentrationStorer(object):
    #A class for storing time-evolving odor concentration data, and retrieving
    #it for use with simulated flies.
    def __init__(self,initial_conc_array,image,dt_store,t_stop,vmin,vmax,
    centre_rel_diff_scale,
    puff_release_rate,
    puff_spread_rate,
    puff_init_rad,
    puff_mol_amount,
    cmap='Reds'):
        #Make sure initial_conc_array has been flipped from the pompy bug
        self.simulation_region = image.get_extent()
        self.dt_store = dt_store
        #concentration data stored here, x by y by t
        x_pixels, y_pixels = scipy.shape(initial_conc_array)
        # print((x_pixels,y_pixels,int(t_stop/dt_store)))
        n = datetime.datetime.utcnow()
        self.filename = 'concObject{0}.{1}-{2}:{3}'.format(
        n.month,n.day,n.hour,n.minute)
        self.hdf5_filename = self.filename+'.hdf5'
        num_steps = int(t_stop/dt_store)
        run_param = {
        'num_steps': num_steps, 'dt_store':dt_store,
        'simulation_region': self.simulation_region,
        'simulation_time':t_stop,
        'cmap': cmap, 'imshow_bounds':(vmin,vmax),
        'centre_rel_diff_scale':centre_rel_diff_scale,
        'puff_init_rad':        puff_init_rad,
        'puff_release_rate':         puff_release_rate,
        'puff_spread_rate':         puff_spread_rate,
        'puff_mol_amount':        puff_mol_amount
        }
        self.logger = h5_logger.H5Logger(self.hdf5_filename,param_attr=run_param)
    def store(self,conc_array):
        data = {'conc_array':conc_array}
        self.logger.add(data)


class WindStorer(object):
    #A class for storing time-evolving wind data, and retrieving
    #it for use with simulated flies.
    def __init__(self,initial_wind_array,x_points,y_points,dt_store,t_stop,
    wind_grid_density,noise_gain,noise_damp,
    noise_bandwidth,data_loc=None):
        self.u = initial_wind_array[:,:,0]
        self.v = initial_wind_array[:,:,1]
        self.dt_store = dt_store
        self.t_stop = t_stop
        self.x_points = x_points
        self.y_points = y_points
        n = datetime.datetime.utcnow()
        self.filename = 'windObject{0}.{1}-{2}:{3}'.format(
        n.month,n.day,n.hour,n.minute)
        self.hdf5_filename = self.filename+'.hdf5'
        run_param = {
        'dt_store':dt_store, 'simulation_time':t_stop,
        'x_points':x_points.tolist(),'y_points':y_points.tolist(),
        'wind_grid_density':wind_grid_density,
        'noise_gain':noise_gain,
        'noise_damp':noise_damp,
        'noise_bandwidth':noise_bandwidth,
        'data_loc':data_loc
        }
        # print(y_points)
        self.logger = h5_logger.H5Logger(self.hdf5_filename,param_attr=run_param)

    def store(self,velocity_field):
        data = {'velocity_field':velocity_field}
        self.logger.add(data)
