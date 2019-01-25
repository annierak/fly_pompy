import h5_logger
import numpy as np
import h5py
import json


#(1) creation and storing
run_param = {'a':1,'b':2}
hdf5_file = 'test_hdf5.hdf5'
logger = h5_logger.H5Logger(hdf5_file,param_attr=run_param)
data = {'data':np.array([1,2,3])}
logger.add(data)
print(logger.param_attr)


#(2) reloading stored data
data = h5py.File(hdf5_file,'r')
run_param = json.loads(data.attrs['jsonparam'])
print(run_param)
print(run_param['a'])
