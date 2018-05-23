import scipy
import scipy.stats
import math
import pandas as pd
import sys

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
