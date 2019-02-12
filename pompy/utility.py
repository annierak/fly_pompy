import scipy
import scipy.stats
import scipy.interpolate as interpolate
import math
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt

def rotate_vecs(x,y,angle):
    xrot = x*scipy.cos(angle) - y*scipy.sin(angle)
    yrot = x*scipy.sin(angle) + y*scipy.cos(angle)
    return xrot, yrot


def shift_and_rotate(p, shift, angle):
    p_vec = scipy.array(p)
    shift_vec = scipy.array(shift)
    rot_mat = rotation_matrix(angle)
    return scipy.dot(rot_mat, p_vec - shift_vec)


def rotation_matrix(angle):
    A = scipy.array([
        [scipy.cos(angle), -scipy.sin(angle)],
        [scipy.sin(angle),  scipy.cos(angle)]
        ])
    return A


def draw_from_inputted_distribution(data,dt,n_samples):
    #function for using empirical release data to
    #draw from the time-varying departure rate they exhibit
    #Input: the empirical release data (a list of times)
    #dt determines the bin size
    #n_samples is the number of samples to return
    t_max = max(data)
    t_min = min(data)
    bins = scipy.linspace(t_min,t_max,(t_max-t_min)/dt)
    hist, bin_edges = scipy.histogram(data, bins=bins, density=True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    r = np.random.rand(n_samples)
    return inv_cdf(r)



def create_circle_of_sources(number,radius,strength):
    location_list = []
    for i in range(number):
        angle = i*(2.0*scipy.pi)/number
        x = radius*scipy.cos(angle)
        y = radius*scipy.sin(angle)
        location_list.append((x,y))
    strength_list = [strength for x in location_list]
    return location_list, strength_list


def create_grid_of_sources(x_num, y_num, x_range, y_range,  strength):
    x_vals = scipy.linspace(x_range[0], x_range[1], x_num)
    y_vals = scipy.linspace(y_range[0], y_range[1], y_num)
    location_list = [(x,y) for x in x_vals for y in y_vals]
    strength_list = [strength for x in location_list]
    return location_list, strength_list


def distance(p,q):
    return scipy.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)


def unit_vector(x,y):
    v_mag = scipy.sqrt(x**2 + y**2)
    if type(v_mag) == scipy.ndarray:
        mask = v_mag > 0
        x_unit = scipy.zeros(x.shape)
        y_unit = scipy.zeros(y.shape)
        x_unit[mask] = x[mask]/v_mag[mask]
        y_unit[mask] = y[mask]/v_mag[mask]
    else:
        if (v_mag > 0):
            x_unit = x/v_mag
            y_unit = y/v_mag
        else:
            x_unit = 0.0
            y_unit = 0.0
    return x_unit, y_unit


def logistic(x,x0,k):
    return 1.0/(1.0 + scipy.exp(-k*(x-x0)))

def par_perp(u,v):
    #Returns the components of u parallel to and perpendicular to v, as cartesian vectors.
    if (u[0],u[1]) == (0.,0.):
        print('zero wind')
        return 0,0
    par = (scipy.inner(u,v))/(scipy.inner(v,v))*v
    if scipy.isnan(par[0]):
        print(u,v)
        raise ValueError('there is a par/perp Nan problem')
        # sys.exit()
    perp = u - par
    return par,perp

def cartesian_to_polar(x,y):
    r = math.sqrt(x**2+y**2)
    if x>0:
        theta = math.atan(y/x)
    else:
        theta = math.atan(y/x)+math.pi
    theta = theta % (2*math.pi)
    return (r,theta)


def fit_von_mises(heading_data):
    #Returns tuple (mean, kappa) of the von mises fit to inputted data.
    #Structure of input: heading data is dict with
    #key1: 'angles' : 1xn array of angles
    #key2: 'counts' : mxn array of angle counts
    angles = heading_data['angles']
    counts = heading_data['counts']
    #Create a weighted histogram where each row gets weight inversely proportional to
    counts[1,:] = 5*counts[1,:]
    counts = scipy.sum(counts,0)
    #Fit the histogram to a von mises
    draws = tuple(scipy.repeat(angles[i],counts[i]) for i in range(len(angles)))
    headings = scipy.concatenate(draws)
    #import matplotlib.pyplot as plt
    #plt.subplot(111,polar=True);plt.hist(headings);plt.show()
    (kappa_est,mu_est,scale) = scipy.stats.vonmises.fit(headings,fscale=1)
    #raw_input('Done?')
    return mu_est, kappa_est


def process_wind_data(wind_data_file,release_delay,wind_dt=None):
    #Takes in a csv file and outputs wind_angle,wind_speed,wind_dt
    wind_df = pd.read_csv('/home/annie/work/programming/odor_tracking_sim/data_files/'+wind_data_file)
    cols = list(wind_df.columns.values)

    #if a release delay is required, insert rows for the extra time into the dataframe with value of beginning wind value
    rows_to_add = int((release_delay*60)/wind_dt)
    df_to_insert = pd.DataFrame({
            cols[0]: [wind_df[cols[0]][0] for i in range(rows_to_add)],
            cols[1]: [wind_df[cols[1]][0] for i in range(rows_to_add)],
            cols[2]: [wind_df[cols[2]][0] for i in range(rows_to_add)]
            })
    wind_df = pd.concat([df_to_insert,wind_df.ix[:]]).reset_index(drop=True)

    secs,degs,mph = tuple(wind_df[col].as_matrix() for col in cols)
    #Convert min to seconds
    times = 60.*secs
    if wind_dt is None:
        wind_dt = times[1]-times[0]
    else:
        #Directly provided wind_dt in seconds
        wind_dt = wind_dt

    #Convert degrees to radians and switch to going vs coming
    wind_angle = (scipy.radians(degs)+scipy.pi)%(2*scipy.pi)
    #Convert mph to meters/sec
    wind_speed = mph*(1/3600.)*1609.34
    return {'wind_angle':wind_angle,'wind_speed': wind_speed,'wind_dt':wind_dt}


def customaxis(ax, c_left='k', c_bottom='k', c_right='none', c_top='none',
               lw=1, size=12, pad=8):

    for c_spine, spine in zip([c_left, c_bottom, c_right, c_top],
                              ['left', 'bottom', 'right', 'top']):
        if c_spine != 'none':
            ax.spines[spine].set_color(c_spine)
            ax.spines[spine].set_linewidth(lw)
        else:
            ax.spines[spine].set_color('none')
    if (c_bottom == 'none') & (c_top == 'none'): # no bottom and no top
        ax.xaxis.set_ticks_position('none')
    elif (c_bottom != 'none') & (c_top != 'none'): # bottom and top
        ax.tick_params(axis='x', direction='out', width=lw, length=7,
                      color=c_bottom, labelsize=size, pad=pad)
    elif (c_bottom != 'none') & (c_top == 'none'): # bottom but not top
        ax.xaxis.set_ticks_position('bottom')
        ax.tick_params(axis='x', direction='out', width=lw, length=7,
                       color=c_bottom, labelsize=size, pad=pad)
    elif (c_bottom == 'none') & (c_top != 'none'): # no bottom but top
        ax.xaxis.set_ticks_position('top')
        ax.tick_params(axis='x', direction='out', width=lw, length=7,
                       color=c_top, labelsize=size, pad=pad)
    if (c_left == 'none') & (c_right == 'none'): # no left and no right
        ax.yaxis.set_ticks_position('none')
    elif (c_left != 'none') & (c_right != 'none'): # left and right
        ax.tick_params(axis='y', direction='out', width=lw, length=7,
                       color=c_left, labelsize=size, pad=pad)
    elif (c_left != 'none') & (c_right == 'none'): # left but not right
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='y', direction='out', width=lw, length=7,
                       color=c_left, labelsize=size, pad=pad)
    elif (c_left == 'none') & (c_right != 'none'): # no left but right
        ax.yaxis.set_ticks_position('right')
        ax.tick_params(axis='y', direction='out', width=lw, length=7,
                       color=c_right, labelsize=size, pad=pad)

# Testing/development
# --------------------------------------------------------------------
if __name__ == '__main__':

    pass
