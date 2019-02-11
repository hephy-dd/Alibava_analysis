"""Noise analysis of ALiBaVa files"""
# pylint: disable=C0103,R0902,C0301,R0914
import logging
from time import time
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from analysis_classes.nb_analysis_funcs import nb_noise_calc
from analysis_classes.utilities import import_h5, gaussian, read_binary_Alibava
# from analysis_classes.utilities import manage_logger

class NoiseAnalysis:
    """This class contains all calculations and data concerning pedestals in
	ALIBAVA files"""

    def __init__(self, path="", configs=None, logger=None):
        """
        :param path: Path to pedestal file
        """
        self.log = logger or logging.getLogger(__class__.__name__)

        self.log.info("Loading pedestal file: %s", path)
        if not configs["isBinary"]:
            self.data = import_h5(path)[0]
        else:
            self.data = read_binary_Alibava(path)

        if self.data:
            # Some of the declaration may seem unecessary but it clears things
            # up when you need to know how big some arrays are

            # self.data["events"]["signal"][0] -> array of signals of 0th event of all channels
            self.numchan = len(self.data["events"]["signal"][0])
            self.numevents = len(self.data["events"]["signal"])
            self.pedestal = np.zeros(self.numchan, dtype=np.float32)
            self.noise = np.zeros(self.numchan, dtype=np.float32)
            self.noise_cut = configs.get("Noise_cut", 5.)
            self.mask = configs.get("Manual_mask", [])
            # self.goodevents -> array containing indicies of events with good timing
            self.goodevents = np.nonzero(self.data["events"]["time"][:] >= 0)
            self.CMnoise = np.zeros(len(self.goodevents[0]), dtype=np.float32)
            self.CMsig = np.zeros(len(self.goodevents[0]), dtype=np.float32)
            # array for noise calculations: score per channel = signal - pedestal - CM
            self.score = np.zeros((len(self.goodevents[0]), self.numchan),
                                  dtype=np.float32)
            self.median_noise = None
            usejit = configs.get("optimize", False)

            # Calculate pedestal
            self.log.info("Calculating pedestal and Noise...")
            # mean signal per channel over all events
            self.pedestal = np.mean(self.data["events"]["signal"][0:], axis=0)
            # complete signal matrix (event vs. channel)
            self.signal = np.array(self.data["events"]["signal"][:], dtype=np.float32)

            # Noise Calculations
            if not usejit:
                start = time()
                self.noise, self.CMnoise, self.CMsig = \
                    self.noise_calc(self.signal, self.pedestal[:],
                                    self.numevents, self.numchan)
                self.noisy_strips, self.good_strips = \
                    self.detect_noisy_strips(self.noise, self.noise_cut)
                self.noise_corr, self.CMnoise, self.CMsig = \
                    self.noise_calc(self.signal[:, self.good_strips],
                                    self.pedestal[self.good_strips],
                                    self.numevents, len(self.good_strips))
                end = time()
                self.log.info("Time taken: %s seconds",
                              str((round(abs(end - start), 2))))
            else:
                self.log.info("Jit version used!!! No progress bar can be shown")
                start = time()
                self.noise, self.CMnoise, self.CMsig = \
                    nb_noise_calc(self.signal, self.pedestal)
                self.noisy_strips, self.good_strips = \
                        self.detect_noisy_strips(self.noise, self.noise_cut)
                self.noise_corr, self.CMnoise, self.CMsig, self.total_noise = \
                        nb_noise_calc(self.signal[:, self.good_strips],
                                      self.pedestal[self.good_strips], True)
                end = time()
                self.log.info("Time taken: %s seconds",
                              str(round(abs(end - start), 2)))
            # self.total_noise = np.concatenate(self.score, axis=0)
        else:
            self.log.warning("No valid file, skipping pedestal run")

    def detect_noisy_strips(self, Noise, Noise_cut):
        """Detect noisy strips (Noise > self.median_noise + Noise_cut) and
        includes the masking given by the user.
        Returns: noisy strips (np.array), good strips (np.array)"""

        good_strips = np.arange(len(Noise))
        # Calculate the median noise over all channels
        self.median_noise = np.median(Noise)
        high_noise_strips = np.nonzero(Noise > self.median_noise + Noise_cut)[0]
        high_noise_strips = np.append(high_noise_strips, self.mask)
        good_strips = np.delete(good_strips, high_noise_strips)

        return high_noise_strips, good_strips

    def noise_calc(self, events, pedestal, numevents,
                   numchannels, tot_noise=False):
        """Noise calculation of normal noise (NN) and common mode noise (CMN).
        Uses numpy, can be further optimized by reducing memory access to member variables.
        But got 36k events per second.
        So fuck it.
        This function is not numba optimized!!!"""
        score = np.zeros((numevents, numchannels), dtype=np.float32)
        noise = score
        CMnoise = np.zeros(numevents, dtype=np.float32)
        CMsig = np.zeros(numevents, dtype=np.float32)

        # Loop over all good events
        for event in tqdm(range(self.goodevents[0].shape[0]), desc="Events processed:"):
            # Calculate the common mode noise for every channel
            cm = events[event][:] - pedestal  # Get the signal from event and subtract pedestal
            CMNsig = np.std(cm)  # Calculate the standard deviation
            CMN = np.mean(cm)  # Now calculate the mean from the cm to get the actual common mode noise

            # Calculate the noise of channels
            # Subtract the common mode noise --> Signal[arraylike] - pedestal[arraylike] - Common mode
            cn = cm - CMN
            score[event] = cn
            # Append the common mode values per event to the data arrays
            CMnoise[event] = CMN
            CMsig[event] = CMNsig

        # get noise per channel from score value per channel
        noise = np.std(score, axis=0)

        if tot_noise is False:
            return noise, CMnoise, CMsig  # Return everything
        return noise, CMnoise, CMsig, np.concatenate(score, axis=0)


    def plot_data(self):
		# COMMENT: every plot needs its own method!!!
        """Plots the data calculated by the framework"""

        fig = plt.figure("Noise analysis")

        # Plot noisedata
        noise_plot = fig.add_subplot(221)
        noise_plot.bar(np.arange(self.numchan), self.noise, 1., alpha=0.4, color="b")
        # array of non masked strips
        valid_strips = np.ones(self.numchan)
        valid_strips[self.noisy_strips] = 0
        noise_plot.plot(np.arange(self.numchan), valid_strips, 1., color="r", label="Masked strips")

        # Plot the threshold for deciding a good channel
        xval = [0, self.numchan]
        # COMMENT: Again, this should be -/+ noise_cut not +/+ or I dont get it at all
        yval = [self.median_noise - self.noise_cut,
                self.median_noise + self.noise_cut]
        noise_plot.plot(xval, yval, 1., "r--", color="g", label="Threshold for noisy strips")

        noise_plot.set_xlabel('Channel [#]')
        noise_plot.set_ylabel('Noise [ADC]')
        noise_plot.set_title('Noise levels per Channel')
        noise_plot.legend()

        # Plot pedestal
        pede_plot = fig.add_subplot(222)
        pede_plot.bar(np.arange(self.numchan), self.pedestal, 1., yerr=self.noise,
                      error_kw=dict(elinewidth=0.2, ecolor='r', ealpha=0.1), alpha=0.4, color="b")
        pede_plot.set_xlabel('Channel [#]')
        pede_plot.set_ylabel('Pedestal [ADC]')
        pede_plot.set_title('Pedestal levels per Channel with noise')
        pede_plot.set_ylim(bottom=min(self.pedestal) - 50.)
        # pede_plot.legend()

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
        CM_plot.set_title(
            r'$\mathrm{Common\ mode\:}\ \mu=' + str(round(mu, 2)) + r',\ \sigma=' + str(round(std, 2)) + r'$')
        # CM_plot.legend()

        # Plot noise hist
        CM_plot = fig.add_subplot(224)
        n, bins, patches = CM_plot.hist(self.total_noise, bins=500, density=False, alpha=0.4, color="b")
        CM_plot.set_yscale("log", nonposy='clip')
        CM_plot.set_ylim(1.)

        # Cut off noise part
        cut = np.max(n) * 0.2  # Find maximum of hist and get the cut
        ind = np.concatenate(np.argwhere(n > cut))  # Finds the first element which is higher as threshold optimized

        # Calculate the mean and std
        mu, std = norm.fit(bins[ind])
        # Calculate the distribution for plotting in a histogram
        plotrange = np.arange(-35, 35)
        p = gaussian(plotrange, mu, std, np.max(n))
        CM_plot.plot(plotrange, p, "r--", color="g")

        CM_plot.set_xlabel('Noise')
        CM_plot.set_ylabel('count')
        CM_plot.set_title("Noise Histogram")

        fig.tight_layout()
        plt.draw()
        plt.show()
