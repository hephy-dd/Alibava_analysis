# This file contains functions for noise and pedestal analysis of ALIBAVA files

__version__ = 0.1
__date__ = "13.12.2018"
__author__ = "Dominic Bloech"
__email__ = "dominic.bloech@oeaw.ac.at"

# Import statements
from utilities import import_h5
from tqdm import tqdm
import numpy as np
from scipy.stats import norm, stats
import matplotlib.pyplot as plt
from numba import jit, prange


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

        if self.data:
            pass
        else:
            print("No valid file, skipping event run")


    def plot_data(self):
        """This function plots all data processed"""


class calibration:
    """This class handles all concerning the calibration"""

    def __init__(self, path = ""):
        """
        :param path: Path to calibration file
        """
        # Init parameters
        print("Loading calibration file: {!s}".format(path))
        self.data = import_h5(path)

        if self.data:
            pass
        else:
            print("No valid file, skipping calibration run")

    def plot_data(self):
        """Plots the processed data"""


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
            self.pedestal = np.zeros(self.numchan, dtype=np.float32)
            self.noise = np.zeros(self.numchan, dtype=np.float32)
            self.goodevents = np.nonzero(self.data['/events/time'][:] >= 0)  # Only use events with good timing
            self.CMnoise = np.zeros(len(self.goodevents[0]), dtype=np.float32)
            self.CMsig = np.zeros(len(self.goodevents[0]), dtype=np.float32)
            self.vectnoise_funct = np.vectorize(self.noise_calc, otypes=[np.float], cache=False)
            self.score = np.zeros((len(self.goodevents[0]), self.numchan), dtype=np.float32)  # Variable needed for noise calculations

            # Calculate pedestal
            print("Calculating pedestal and Noise...")
            self.pedestal = np.mean(self.data['/events/signal'][0:], axis=0)

            # Noise Calculations
            self.vectnoise_funct()
            #self.noise_calc()
            self.noise = np.std(self.score, axis=0)  # Calculate the actual noise for every channel by building the mean of all noise from every event
        else:
            print("No valid file, skipping pedestal run")

    @jit(parallel=True)
    def noise_calc(self):
        """Noise calculation, normal noise (NN) and common mode noise (CMN)
        Uses numba and numpy, can be further optimized by reducing memory access to member variables.
        But got 36k events per second.
        So fuck it."""
        events = self.data['/events/signal'][:]

        for event in tqdm(prange(self.goodevents[0].shape[0]), desc="Events processed:"): # Loop over all good events

            # Calculate the common mode noise for every channel
            cm = np.single(events[event][:]) - self.pedestal  # Get the signal from event and subtract pedestal
            CMNsig = np.std(cm)  # Calculate the standard deviation
            CMN = np.mean(cm)  # Now calculate the mean from the cm to get the actual common mode noise

            # Append the common mode values per event into the data arrays
            self.CMnoise[event] = CMN
            self.CMsig[event] = CMNsig

            # Calculate the noise of channels
            cn = cm - CMN # Subtract the common mode noise --> Signal[arraylike] - pedestal[arraylike] - Common mode
            self.score[event] = cn


    def plot_data(self):
        """Plots the data calculated by the framework"""

        fig = plt.figure()

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

        #fig.tight_layout()
        plt.draw()

if __name__ == "__main__":
    noise = noise_analysis(path = r"\\HEROS\dbloech\Alibava_measurements\VC811929\Pedestal.hdf5")