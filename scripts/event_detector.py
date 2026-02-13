import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from glob import glob
from scipy import signal
from matplotlib.ticker import FormatStrFormatter
from scipy.signal import detrend
from tqdm import tqdm
import obspy
from obspy.signal.trigger import recursive_sta_lta, plot_trigger, trigger_onset
import pickle as pkl

import copy


import pandas as pd
import rasterio
from rasterio.plot import show
from scipy.stats import gaussian_kde

from datetime import timedelta
import datetime
import numpy as np
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

import pyTMD
import numpy as np
        
import pyTMD.io
import pyTMD.predict
import pyTMD.tools
import pyTMD.utilities
import timescale.time

np.float_ = np.float64
import dascore as dc

def sintela_to_datetime(sintela_times):
    '''
    returns an array of datetime.datetime 
    ''' 
    
    days1970 = datetime.datetime.date(datetime.datetime(1970, 1, 1)).toordinal()

    # Vectorize everything
    converttime = np.vectorize(datetime.datetime.fromordinal)
    addday_lambda = lambda x : datetime.timedelta(days=x)
    adddays = np.vectorize(addday_lambda )
    
    day = days1970 + sintela_times/1e6/60/60/24
    thisDateTime = converttime(np.floor(day).astype(int))
    dayFraction = day-np.floor(day)
    thisDateTime = thisDateTime + adddays(dayFraction)

    return thisDateTime

def preprocessing_step(file):

    # Load data #
    f = h5py.File(file)
    attrs = f['Acquisition'].attrs
    data = f['Acquisition']['Raw[0]']['RawData'][:]
    this_time = f['Acquisition']['Raw[0]']['RawDataTime'][:]
    times = sintela_to_datetime(this_time)
    x = np.linspace(0,data.shape[1],data.shape[1]) * attrs['SpatialSamplingInterval']
    fs = attrs['PulseRate'] #sample rate

    if file[-9:-7] != '00':
        time_start = times[0] - datetime.timedelta(seconds=times[0].second, microseconds=times[0].microsecond)
        forward_step = np.arange(time_start, times[0], 500).shape[0]
        data_locator = np.array([int(i) for i in (this_time-this_time[0])/500]) + forward_step -1
    else:
        data_locator = np.array([int(i) for i in (this_time-this_time[0])/500])

    filled_data = np.zeros((int(fs*60),data.shape[1]))
    filled_data[:] = np.nan


    bp_top = 500
    bp_bottom = 2
    downsample_rate = int(fs/bp_top)

    #filter by freq
    sos = signal.butter(10, [bp_bottom,bp_top], 'bp', fs=fs, output='sos')
    filtered = signal.sosfiltfilt(sos, data, axis=0)

    #normalize
    normed_filtered = chan_norm(filtered)

    if data_locator[0]==0:
        filled_data[data_locator] = filtered

    else:
        first_filler = np.array([normed_filtered[0,:]]*data_locator[0])
        filled_data[:first_filler.shape[0],:] = first_filler

    mean_of_chans = np.nanmean(filled_data, axis=0)
    idx = np.where(np.isnan(filled_data))
    filled_data[idx] = np.take(mean_of_chans, idx[1])

    #FK filter
    ## rescale the times to the filled data
    if len(this_time) != int(fs*60):
        this_time = np.arange(this_time[0],int(this_time[0]+(500*fs*60)), 500)
    filled_times = sintela_to_datetime(this_time)
    new_format_times = time_fixer_4_fk(filled_times)

    dims = ('time', 'distance')
    patch = dc.Patch(data=filled_data, coords=dict(time=[np.datetime64(i) for i in new_format_times], distance=x), dims=dims)
    filt_cutoffs = np.array([500, 1500, 7000, 20000])

    fk_filtered = patch.slope_filter(filt=filt_cutoffs)
    fk_filtered_data = np.array(fk_filtered.data)

    new_attrs = dict(attrs)

    if new_attrs['MeasurementStartTime'].decode('UTF-8')[17:19] == '00' and data.shape[0] != int(new_attrs['PulseRate']*60):
        new_attrs['Status'] = 'Corrupted'
    else:
        new_attrs['Status'] = 'Good'

    ## Downsample 
    filled_times = filled_times[::downsample_rate]
    filt_filled_data = fk_filtered_data[::downsample_rate,:]

    new_attrs['PulseRate'] = new_attrs['PulseRate']/downsample_rate


    return filt_filled_data, filled_times, new_attrs

def foo(a, b):
    t = mdates.num2date(a)
    ms = str(t.microsecond)[:1]
    res = f"{t.hour:02}:{t.minute:02}:{t.second:02}.{ms}"
    return res

def chan_norm(das_data):
    data_normed = (das_data - np.mean(das_data, axis=0))/np.std(das_data, axis=0).T
    # data_normed_all_axis = (data_normed.T - np.mean(data_normed, axis=1))/np.std(data_normed, axis=1)   

    return data_normed

class DataStats:
    def __init__(self, data, attrs, times):
        self.sampling_rate = attrs["PulseRate"]
        self.npts = data.shape[0]
        self.starttime = times[0]
        # self.starttime.isoformat

class DAS:
    def __init__(self, id, data, attrs, times):
        self.id = id
        self.data = data
        self.stats = DataStats(data, attrs, times)


def obspy_stream_from_das(data, attrs,times):
    stats_default = {
        'network':'eastwind',
        'station':'',
        'location':'',
        'channel':'DAS',
        'starttime':times[0].strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
        'endtime':times[-1].strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
        'sampling_rate':attrs['PulseRate'],
        'delta':1/attrs['PulseRate'],
        'npts':0,
        'calib':1.0
    }

    streams = []
    for n,i in enumerate(data.T):
        tr = obspy.Trace(data=i,header=stats_default)
        # tr.stats.station = f'Channel {n}'
        tr.stats.npts = len(i)

        st = obspy.Stream(tr)
        streams.append(st)
    return streams

def parallel_event_finding(dummy):
    times_all,attrs,channel,all_data = dummy
    trigger_on=3.5, 
    trigger_off=1.2
    
    DAS_channel = DAS(channel, all_data, attrs, times_all)


    cft = recursive_sta_lta(DAS_channel.data, int(1 * attrs['PulseRate']), int(10 * attrs['PulseRate']))
    trigger_times = trigger_onset(cft, trigger_on, trigger_off)
    trigger_times_list = [trigger_times]     

    trigger_times_list_datetime = []

    for trigs_samp_time in trigger_times_list:
        trigger_times_list_datetime.append(times_all[trigs_samp_time])
    return {str(channel): trigger_times_list_datetime}


def time_fixer_4_fk(times):
    times_copy = times.copy()
    new_format_times = []
    for i in times_copy:
        if round(i.microsecond/500)*500 == 1000000:
            microsecond = 0
            second = i.second + 1
            minute = i.minute
            hour = i.hour
            day = i.day
            if second == 60:
                second = 0
                minute = minute+1

            if minute == 60:
                minute = 0
                hour=hour+1
            
            if hour == 24:
                hour = 0
                day = day+1
            
        else:
            microsecond = int(round(i.microsecond/500)*500)
            second = i.second
            minute = i.minute
            hour = i.hour
            day = i.day

        new_format_times.append(datetime.datetime(year=i.year,
                                                month=i.month,
                                                day=day,
                                                hour=hour,
                                                minute=minute,
                                                second=second,
                                                microsecond=microsecond))
    return new_format_times


import pathlib



import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn.mass_downloader import (
    GlobalDomain,
    Restrictions,
    MassDownloader,
)
from obspy.core import AttribDict
from pyproj import Proj
from multiprocessing import Pool

from quakemigrate import QuakeScan, Trigger
from quakemigrate.io import Archive, read_stations
from quakemigrate.lut import compute_traveltimes
from quakemigrate.signal.onsets import STALTAOnset
from quakemigrate.signal.pickers import GaussianPicker


files_list = sorted(glob('/1-fnp/petasaur/p-jbod1/antarctica2425/incoming/Eastwind_decimator_2024*'))

trigger_on = 3.5
trigger_off = 1.2
all_trigger_times = [[] for holder in range(int(351))]

current_status = 'Good'

for i in tqdm(range(len(files_list))):


    if files_list[i][-9:-7] != '00' or i == 0 or current_status == 'Corrupted':

        try:
            filt_filled_data, times, attrs = preprocessing_step(files_list[i])

            current_status = attrs['Status']


        except Exception as e: 
            print(e)
            current_status = 'Corrupted'
            continue

    else:
        first = copy.deepcopy(filt_filled_data)
        times_first = copy.deepcopy(times)
        try:
            filt_filled_data, times, attrs = preprocessing_step(files_list[i])
            current_status = attrs['Status']
        except Exception as e: 
            print(e)
            current_status = 'Corrupted'
            continue
        
        # overlap_data[:,:] = filt_filled_data

        all_data = np.concatenate((first,filt_filled_data), axis=0)
        times_all = np.concatenate((times_first,times), axis=0)

        # fig,ax = plt.subplots()
        # ax.plot(all_data[:,30])

        # fig,ax = plt.subplots()
        # ax.imshow(all_data, aspect='auto', vmin=-1, vmax=1)

        input_data_dict = {str(i):all_data[:,i] for i in range(all_data.shape[1])}

        new_iterable = [[times_all,dict(attrs),i,input_data_dict[i]] for i in input_data_dict]

        with Pool(48) as p:
            picks = p.map(parallel_event_finding,new_iterable)
        for n,e in enumerate(picks):
            try:
                all_trigger_times[n].extend(e[str(n)][0][:,0])
            except:
                continue

    # if i ==10: break

with open('../auto_picked_events_all_chans_fk_filtered.pkl', 'wb') as file:
    # Serialize and write the data to the file
    pkl.dump(all_trigger_times, file)