import time
import scipy
import matplotlib
import matplotlib.pyplot as plt
import sys
import itertools

import numpy as np
import os
import cPickle as pickle

from scipy.optimize import curve_fit

#The input values

xlim = (.1, 40.)
ylim = (-10., 10.)

array_dim_x = 999
array_dim_y = array_dim_x
conc_locs_x,conc_locs_y = np.meshgrid(
    np.linspace(xlim[0],xlim[1],array_dim_x),
    np.linspace(ylim[0],ylim[1],array_dim_y))


#The approximation function
def gauss_approx((x,y),Q,C_y,n):
    return (Q/(2*np.pi*(0.5*C_y*(x**((2-n)/2)))**2))*\
        np.exp(-1*((y**2)/(2*(0.5*C_y*(x**((2-n)/2))))))

#True parameter values
true_Q,true_C_y,true_n = (2e-6,0.4,1.)

#True output values before noise
C = gauss_approx((conc_locs_x,conc_locs_y),true_Q,true_C_y,true_n)

#Add Gaussian noise with variance 10
C_nsy = C + 10*np.random.randn(*np.shape(C))

# plt.subplot(211)
# plt.imshow(C_nsy)
# plt.subplot(212)
# plt.hist(C_nsy)
# plt.show()

p_opt,p_cov = curve_fit(gauss_approx,(conc_locs_x.flatten(),conc_locs_y.flatten()),
    C_nsy.flatten(),method='dogbox')#,p0=initial_guess)

Q_est,C_y_est,n_est = p_opt

print(p_opt)
