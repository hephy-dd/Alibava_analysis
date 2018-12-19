# This file contains functions for noise and pedestal analysis of ALIBAVA files

__version__ = 0.1
__date__ = "13.12.2018"
__author__ = "Dominic Bloech"
__email__ = "dominic.bloech@oeaw.ac.at"

# Import statements
from utilities import *
from tqdm import tqdm
import numpy as np
from scipy.stats import norm, stats
import matplotlib.pyplot as plt
from numba import jit, prange
from scipy.interpolate import CubicSpline
from sklearn import metrics
from sklearn.cluster import KMeans


class event_analysis:
    """This class analyses measurement files per event"""

    def __init__(self, path_list = None, **kwargs):
        """

        :param path_list: List of pathes to analyse
        :param kwargs: kwargs if further data should be used, possible kwargs=calibration,noise
        """

        # Init parameters
        print("Loading event file(s): {!s}".format(path_list))
        self.data = import_h5(path_list)

        self.numchan = len(self.data[0]["events/signal"][0])
        self.numevents = len(self.data[0]["events/signal"])
        self.pedestal = np.zeros(self.numchan, dtype=np.float64)
        self.noise = np.zeros(self.numchan, dtype=np.float64)
        self.Hitmap_list = []  # Simply a fill of all channel which fires
        self.SN_cut = 1
        self.hits = 0
        self.tmin = 0
        self.tmax = 100
        self.maxcluster = 4
        self.CMN = np.zeros(self.numchan, dtype=np.float64)
        self.CMsig = np.zeros(self.numchan, dtype=np.float64)
        self.outputdata = {}

        self.material = kwargs.get("sensor_type", "n-in-p")
        self.masking = kwargs.get("masking", False)


        if "pedestal" in kwargs:
            self.pedestal = kwargs["pedestal"]

        if "SN_ratio" in kwargs:
            self.SN_cut = kwargs["SN_ratio"] # Cut for the signal to noise ratio

        if "CMN" in kwargs:
            self.CMN = kwargs["CMN"] # CMN for every channel

        if "CMsig" in kwargs:
            self.CMsig = kwargs["CMsig"] # Common mode sig for every channel

        if "Noise" in kwargs:
            self.noise = kwargs["Noise"] # Common mode sig for every channel

        if "MaxCluster" in kwargs:
            self.maxcluster = kwargs["MaxCluster"] # Common mode sig for every channel

        if "timing" in kwargs:
            self.min = kwargs["timing"][0] # timinig window
            self.max = kwargs["timing"][1] # timing maximum


        print("Processing files ...")
        # Here a loop over all files will be done to do the analysis on all imported files
        for data in tqdm(prange(len(self.data)), desc="Data files processed:"):
                events = self.data[data]["events/signal"][:]
                timing = self.data[data]["events/time"][:]
                results = self.do_analysis(events, timing)

                self.outputdata[str(self.data[data]).split('"')[1].split('.')[0]] = results



    def do_analysis(self, events, timing):
        """Does the actual event analysis"""

        # get events with good timinig only gtime and only process these events
        gtime = np.nonzero(timing>self.tmin)
        meanCMN = np.mean(self.CMN)
        meanCMsig = np.mean(self.CMsig)
        prodata = []  # List of processed data which then can be accessed
        #Warning: If you have a RS and pulseshape recognition enabled the timing window has to set accordingly
        for event in tqdm(prange(gtime[0].shape[0]), desc="Events processed:"): # Loop over all good events

            signal, SN, CMN, CMsig = self.process_event(events[event], meanCMN, meanCMsig, self.noise)

            self.clustering(signal, SN)

            prodata.append((event, signal, SN, CMN, CMsig))

        return prodata

    def clustering(self, event, SN):
        """Looks for cluster in the event"""

        numclus = 0
        channels = np.nonzero(np.abs(SN) > self.SN_cut)# Only channels which have a signal/Noise higher then the signal/Noise cut

        if self.masking:
            if self.material == "n-in-p":
                # Todo: masking of dead channels etc.
                channels = np.nonzero(np.take(event ,channels) < 0) # So only negative values are considered
            else:
                channels = np.nonzero(np.take(event, channels) > 0) # So only positive values are considered


        used_channels = np.zeros(self.numchan) # To keep track which channel have been used already for clustering and/or masking channels


    def process_event(self, event, CMN, CMsig, noise):
        """Processes single events"""

        # Calculate the common mode noise for every channel
        signal = np.single(event) - self.pedestal  # Get the signal from event and subtract pedestal

        # Remove channels which have a signal higher then 5*CMsig+CMN which are not representative
        prosignal = np.take(signal,np.nonzero(signal<(5*CMsig+CMN))) # Processed signal

        if prosignal.any():
            cmpro = np.mean(prosignal)
            sigpro = np.std(prosignal)

            corrsignal = signal - cmpro
            SN = corrsignal / noise

            return corrsignal, SN, cmpro, sigpro
        else:
            return np.zeros(self.numchan), np.zeros(self.numchan), 0, 0 # A default value return if everything fails


    def plot_data(self, single_event = -1):
        """This function plots all data processed"""

        # Plot a single event from every file
        if single_event > 0:
            for name, data in self.outputdata.items():
                self.plot_single_event(single_event, name)


    def plot_single_event(self, eventnum, file):
        """ Plots a single event and its data"""

        data = self.outputdata[file]

        fig = plt.figure("Event number {!s}, from file: {!s}".format(eventnum, file))

        # Plot signal
        channel_plot = fig.add_subplot(211)
        channel_plot.bar(np.arange(self.numchan), data[eventnum][1][:], 1., alpha=0.4, color="b")
        channel_plot.set_xlabel('channel [#]')
        channel_plot.set_ylabel('Signal [ADC]')
        channel_plot.set_title('Signal')

        # Plot signal/Noise
        SN_plot = fig.add_subplot(212)
        SN_plot.bar(np.arange(self.numchan), data[eventnum][2][:], 1., alpha=0.4, color="b")
        SN_plot.set_xlabel('channel [#]')
        SN_plot.set_ylabel('Signal/Noise [ADC]')
        SN_plot.set_title('Signal/Noise')

        fig.tight_layout()
        plt.draw()



class calibration:
    """This class handles all concerning the calibration"""

    def __init__(self, delay_path = "", charge_path = ""):
        """
        :param delay_path: Path to calibration file
        :param charge_path: Path to calibration file
        """

        self.charge_cal = None
        self.delay_cal = None
        self.delay_data = None
        self.charge_data = None

        self.charge_calibration_calc(charge_path)
        self.delay_calibration_calc(delay_path)


    def delay_calibration_calc(self, delay_path):
        # Delay scan
        print("Loading delay file: {!s}".format(delay_path))
        self.delay_data = read_file(delay_path)
        self.delay_data = get_xy_data(self.delay_data, 2)

        if self.delay_data.any():
            # Interpolate data with cubic spline interpolation
            self.delay_cal = CubicSpline(self.delay_data[:,0],self.delay_data[:,1], extrapolate=True)

    def charge_calibration_calc(self, charge_path):
        # Charge scan
        print("Loading charge calibration file: {!s}".format(charge_path))
        self.charge_data = read_file(charge_path)
        self.charge_data = get_xy_data(self.charge_data, 2)

        if self.charge_data.any():
            # Interpolate data with cubic spline interpolation
            self.charge_cal = CubicSpline(self.charge_data[:,0],self.charge_data[:,1], extrapolate=True)


    def plot_data(self):
        """Plots the processed data"""

        fig = plt.figure("Calibration")

        # Plot delay
        delay_plot = fig.add_subplot(212)
        delay_plot.bar(self.delay_data[:,0], self.delay_data[:,1], 5., alpha=0.4, color="b")
        delay_plot.plot(self.delay_data[:, 0], self.delay_cal(self.delay_data[:, 0]), "r--", color="g")
        delay_plot.set_xlabel('time [ns]')
        delay_plot.set_ylabel('Signal [ADC]')
        delay_plot.set_title('Delay plot')

        # Plot charge
        charge_plot = fig.add_subplot(211)
        charge_plot.bar(self.charge_data[:, 0], self.charge_data[:, 1], 2000., alpha=0.4, color="b")
        charge_plot.plot(self.charge_data[:, 0], self.charge_cal(self.charge_data[:, 0]), "r--", color="g")
        charge_plot.set_xlabel('Charge [e-]')
        charge_plot.set_ylabel('Signal [ADC]')
        charge_plot.set_title('Charge plot')

        fig.tight_layout()
        plt.draw()

class noise_analysis:
    """This class contains all calculations and data concerning pedestals in ALIBAVA files"""

    def __init__(self, path = ""):
        """
        :param path: Path to pedestal file
        """

        # Init parameters
        print("Loading pedestal file: {!s}".format(path))
        self.data = import_h5(path)

        if self.data:
            self.data=self.data[0]# Since I always get back a list
            self.numchan = len(self.data["header/pedestal"][0])
            self.numevents = len(self.data["events/signal"])
            self.pedestal = np.zeros(self.numchan, dtype=np.float64)
            self.noise = np.zeros(self.numchan, dtype=np.float64)
            self.goodevents = np.nonzero(self.data['/events/time'][:] >= 0)  # Only use events with good timing, here always the case
            self.CMnoise = np.zeros(len(self.goodevents[0]), dtype=np.float64)
            self.CMsig = np.zeros(len(self.goodevents[0]), dtype=np.float64)
            self.vectnoise_funct = np.vectorize(self.noise_calc, otypes=[np.float], cache=False)
            self.score = np.zeros((len(self.goodevents[0]), self.numchan), dtype=np.float64)  # Variable needed for noise calculations

            # Calculate pedestal
            print("Calculating pedestal and Noise...")
            self.pedestal = np.mean(self.data['/events/signal'][0:], axis=0)

            # Noise Calculations
            self.vectnoise_funct()
            #self.noise_calc(self.score, self.CMnoise, self.CMsig)
            self.noise = np.std(self.score, axis=0)  # Calculate the actual noise for every channel by building the mean of all noise from every event
        else:
            print("No valid file, skipping pedestal run")

    #@jit(parallel=True)
    def noise_calc(self):
        """Noise calculation, normal noise (NN) and common mode noise (CMN)
        Uses numba and numpy, can be further optimized by reducing memory access to member variables.
        But got 36k events per second.
        So fuck it."""
        events = self.data['/events/signal'][:]
        pedestal = self.pedestal[:]

        for event in tqdm(prange(self.goodevents[0].shape[0]), desc="Events processed:"): # Loop over all good events

            # Calculate the common mode noise for every channel
            cm = np.single(events[event][:]) - pedestal  # Get the signal from event and subtract pedestal
            CMNsig = np.std(cm)  # Calculate the standard deviation
            CMN = np.mean(cm)  # Now calculate the mean from the cm to get the actual common mode noise

            # Calculate the noise of channels
            cn = cm - CMN # Subtract the common mode noise --> Signal[arraylike] - pedestal[arraylike] - Common mode

            self.score[event] = cn
            # Append the common mode values per event into the data arrays
            self.CMnoise[event] = CMN
            self.CMsig[event] = CMNsig

    def plot_data(self):
        """Plots the data calculated by the framework"""

        fig = plt.figure("Noise analysis")

        #Plot noisedata
        noise_plot = fig.add_subplot(221)
        noise_plot.bar(np.arange(self.numchan), self.noise, 1., alpha=0.4, color="b")
        noise_plot.set_xlabel('Channel [#]')
        noise_plot.set_ylabel('Noise [ADC]')
        noise_plot.set_title('Noise levels per Channel')
        #noise_plot.legend()

        # Plot pedestal
        pede_plot = fig.add_subplot(222)
        pede_plot.bar(np.arange(self.numchan), self.pedestal, 1.,
                               yerr=self.noise, error_kw=dict(elinewidth=0.2, ecolor='r', ealpha=0.1), alpha=0.4, color="b")
        pede_plot.set_xlabel('Channel [#]')
        pede_plot.set_ylabel('Pedestal [ADC]')
        pede_plot.set_title('Pedestal levels per Channel with noise')
        pede_plot.set_ylim(bottom=min(self.pedestal)-50.)
        #pede_plot.legend()

        # Plot Common mode
        CM_plot = fig.add_subplot(223)
        n, bins, patches = CM_plot.hist(self.CMnoise, bins=50, density=True, alpha=0.4, color="b")
        # Calculate the mean and std
        mu, std = norm.fit(self.CMnoise)
        # Calculate the distribution for plotting in a histogram
        p = norm.pdf(bins, loc=mu, scale=std)
        CM_plot.plot(bins, p, "r--", color="g")

        CM_plot.set_xlabel('Common mode [ADC]')
        CM_plot.set_ylabel('[%]')
        CM_plot.set_title(r'$\mathrm{Common\ mode\:}\ \mu=' + str(round(mu,2)) + r',\ \sigma=' + str(round(std,2)) + r'$')
        #CM_plot.legend()

        fig.tight_layout()
        plt.draw()

if __name__ == "__main__":
    noise = noise_analysis(path = r"\\HEROS\dbloech\Alibava_measurements\VC811929\Pedestal.hdf5")