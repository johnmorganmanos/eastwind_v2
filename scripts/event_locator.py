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
    bp_top = 200
    bp_bottom = 1
    downsample_rate = int(fs/bp_top)

    #filter by freq
    sos = signal.butter(10, [bp_bottom,bp_top], 'bp', fs=fs, output='sos')
    filtered = signal.sosfiltfilt(sos, data, axis=0)

    #FK filter
    new_format_times = time_fixer_4_fk(times)
    dims = ('time', 'distance')
    patch = dc.Patch(data=filtered, coords=dict(time=[np.datetime64(i) for i in new_format_times], distance=x), dims=dims)
    filt_cutoffs = np.array([0, 400, 8000, np.inf])

    fk_filtered = patch.slope_filter(filt=filt_cutoffs)
    fk_filtered_data = np.array(fk_filtered.data)


    #normalize
    data_normed_filtered = chan_norm(fk_filtered_data)

    if file[-9:-7] != '00':
        time_start = times[0] - datetime.timedelta(seconds=times[0].second, microseconds=times[0].microsecond)
        forward_step = np.arange(time_start, times[0], 500).shape[0]
        data_locator = np.array([int(i) for i in (this_time-this_time[0])/500]) + forward_step -1
    else:
        data_locator = np.array([int(i) for i in (this_time-this_time[0])/500])



    # this_time = np.arange(0,int(fs*60))* 500 + this_time[0]
    # print(this_time)
    # print(filled_data.shape)
    # print(first_filler.shape)
    filled_data = np.zeros((int(fs*60),data_normed_filtered.shape[1]))
    filled_times = np.zeros((int(fs*60)), dtype=object)




    if data_locator[0]==0:
        filled_data[data_locator] = data_normed_filtered

    else:
        first_filler = np.array([data_normed_filtered[:,0]]*data_locator[0])
        filled_data[:first_filler.shape[0],:] = first_filler
    
    filt_filled_data = filled_data

    filled_times[data_locator] = times
    # filled_data[data_locator] = data_normed_filtered.T

    # filt_filled_data = filled_data

    ## Downsample 
    filled_times = filled_times[::downsample_rate]
    filt_filled_data = filt_filled_data[::downsample_rate,::5] #Skip every 5th channel
    new_dict = dict(attrs)
    new_dict['PulseRate'] = new_dict['PulseRate']/downsample_rate


    return filt_filled_data, filled_times, new_dict

def foo(a):
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

import emcee

### Priors ###
top_lim = 1400000
bottom_lim = 1360000

right_lim = 440000
left_lim = 420000

depth_lim = -1000
height_lim = 100

wavespeed_min = 100
wavespeed_max = 7000

### Functions ###
def log_prior(theta, picks):
    s_x,s_y,s_z,t_offset,c = theta

    min_offset = np.array(picks[1]).min() - 5
    max_offset = np.array(picks[1]).min() + 0
    if left_lim < s_x < right_lim and bottom_lim < s_y < top_lim and depth_lim < s_z < height_lim and min_offset < t_offset < max_offset and wavespeed_min < c < wavespeed_max:
        return 0.0
    return -np.inf

def log_likelihood(theta, picks):
    s_x,s_y,s_z,t_offset,c = theta
    chans_of_picks = np.array(picks[0], dtype=int)

    distance = np.sqrt((channel_locations[chans_of_picks,0]-s_x)**2 + (channel_locations[chans_of_picks,1]-s_y)**2 + (channel_locations[chans_of_picks,2]-s_z)**2)   
    

    model = distance/c + t_offset

    best_line_coeffs = np.poly1d(np.polyfit(chans_of_picks, picks[1], 5))
    fit_line = np.polyval(best_line_coeffs, chans_of_picks)
    pick_residual = picks[1] - fit_line
    sigma = np.std(pick_residual)

    sigma2 = sigma**2
    #model =  t_offset + np.linalg.norm(channel_locations[chans_of_picks,:] - np.array([s_x, s_y]), axis=0) / c
    ll = - 0.5 * np.sum((model - picks[1])**2 / sigma2 + np.log(sigma2))

    return ll

def log_probability(theta, picks):
    lp = log_prior(theta, picks)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, picks)


"""

The code below will locate the events given in the picks file given. Because we are not confident in the validity of each pick,
we use the inferred wavespeed to filter out events we think are real vs events picked from noisy data. If the inferred wavespeed is 
greater than 5000 m/s, we count it as a bad event

"""


### Load the cable geometry ###

channel_locations = np.array(pd.read_csv('../cable_geometry/Eastwind_channel_locations.txt', sep=' ', header=None))

### Load the picks
with open('../auto_picked_events/auto_picked_events.pkl', "rb") as f:
    picks = pkl.load(f)

all_flat_samples = {}
bad_events = {}

event_counter = 0
bad_event_counter = 0
for n,i in tqdm(enumerate(picks)):

    ### Get the event ###
    event_num = n
    event_chans, event_times = picks['Event '+str(event_num)][0], picks['Event '+str(event_num)][1]

    current_chan, current_time = event_chans[0], event_times[0]

    ### Filter out any duplicate picks within the same event ###
    event_chans_no_dupes, event_times_no_dupes = [],[]
    for chan,time in zip(event_chans[1:], event_times[1:]):
        if chan != current_chan and time != current_time:
            event_chans_no_dupes.append(chan)
            event_times_no_dupes.append(time)
        current_chan, current_time = chan,time

    this_pick = [event_chans_no_dupes, event_times_no_dupes]

    ### Make an inital guess on the time offset and velocity ###

    offset_init = np.array(this_pick[1]).min() - 0
    c_init = 2000
    s_x_init, s_y_init, s_z_init = 432000, 1.378e6, -150

    ### initial guesses ###
    num_resets = 32

    init_guesses = np.array([s_x_init, s_y_init, s_z_init, offset_init, c_init])

    initial = init_guesses + np.random.randn(num_resets, len(init_guesses)) * np.array([2e3, 5e3, 1e1, 1e-6, 1e2])

    ### Run the MCMC  on just one for a test ###

    nwalkers, ndim = initial.shape

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability,args=[this_pick]
    )
    sampler.run_mcmc(initial, 5000, progress=True);
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

    ### Check if the inferred wave speed is overr 5000 (this is likely noise) ###
    if np.mean(flat_samples[:,4]) < 5000:

        all_flat_samples['Event '+str(event_counter)] = flat_samples
        event_counter+=1
    else:
        bad_events['Event '+str(bad_event_counter)] = flat_samples
        bad_event_counter += 1 


with open('../event_locations/good_event_locations.pkl', 'wb') as file:
    # Serialize and write the data to the file
    pkl.dump(all_flat_samples, file)

with open('../event_locations/bad_event_locations.pkl', 'wb') as file:
    # Serialize and write the data to the file
    pkl.dump(bad_events, file)