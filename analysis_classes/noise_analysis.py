"""Noise analysis of ALiBaVa files"""
# pylint: disable=C0103, R0902

import logging
from time import time
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from nb_analysisFunction import nb_noise_calc
from utilities import import_h5, gaussian


class NoiseAnalysis:
    """This class contains all calculations and data concerning pedestals in
    ALIBAVA files"""

    def __init__(self, path="", usejit=False, configs=None):
        """
        :param path: Path to pedestal file
        """

        self.log = logging.getLogger(__class__.__name__)
        self.log.setLevel(logging.DEBUG)
        if self.log.hasHandlers() is False:
            format_string = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
            formatter = logging.Formatter(format_string)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.log.addHandler(console_handler)

        # Init parameters
        self.log.info("Loading pedestal file(s) from: %s", path)
        self.data = import_h5(path)

        if self.data:
            # Some of the declaration may seem unecessary but it clears things
            # up when you need to know how big some arrays are
            self.data = self.data[0]  # Since I always get back a list
            self.numchan = len(self.data["header/pedestal"][0])
            self.numevents = len(self.data["events/signal"])
            self.pedestal = np.zeros(self.numchan, dtype=np.float32)
            self.noise = np.zeros(self.numchan, dtype=np.float32)
            # Only use events with good timing, here always the case
            self.goodevents = np.nonzero(self.data['/events/time'][:] >= 0)
            self.CMnoise = np.zeros(len(self.goodevents[0]), dtype=np.float32)
            self.CMsig = np.zeros(len(self.goodevents[0]), dtype=np.float32)
            # Variable needed for noise calculations
            self.score = np.zeros((len(self.goodevents[0]), self.numchan),
                                  dtype=np.float32)
            self.configs = configs

            # Calculate pedestal
            self.log.info("Calculating pedestal and Noise...")
            self.pedestal = np.mean(self.data['/events/signal'][0:], axis=0)
            self.signal = np.array(self.data['/events/signal'][:],
                                   dtype=np.float32)

            # Noise Calculations
            if not usejit:
                start = time()
                self.score_raw, self.CMnoise, self.CMsig = self.noise_calc(\
                        self.signal, self.pedestal[:],
                        self.numevents, self.numchan)
                self.noise = np.std(self.score_raw, axis=0)
                self.noisy_strips, self.good_strips = self.detect_noisy_strips(\
                        self.noise, configs.get("Noise_cut", 5.))
                #self.noise_corr = np.std(self.score, axis=0)
                self.score_raw, self.CMnoise, self.CMsig = self.noise_calc(\
                        self.signal[:, self.good_strips],
                        self.pedestal[self.good_strips], self.numevents,
                        len(self.good_strips))
                end = time()
                self.log.info("Process time: %s seconds",
                              str(round(abs(end - start), 2)))
            else:
                self.log.warning("Jit version used!!! No progress bar "
                                 "can be shown")
                start = time()
                self.score_raw, self.CMnoise, self.CMsig = nb_noise_calc(\
                        self.signal, self.pedestal)
                # Calculate the actual noise for every channel by building the
                # mean of all noise from every event
                self.noise = np.std(self.score_raw, axis=0)
                self.noisy_strips, self.good_strips = self.detect_noisy_strips(\
                        self.noise, configs.get("Noise_cut", 5.))
                self.score, self.CMnoise, self.CMsig = nb_noise_calc(\
                        self.signal[:, self.good_strips],
                        self.pedestal[self.good_strips])
                #self.noise_corr = np.std(self.score, axis=0)
                end = time()
                self.log.info("Process time: %s seconds",
                              str(round(abs(end - start), 2)))
            self.total_noise = np.concatenate(self.score, axis=0)

        else:
            self.log.warning("No valid file, skipping pedestal run")

    def detect_noisy_strips(self, Noise, Noise_cut):
        """This function detects noisy strips and returns two arrays first
        array noisy strips, second array good strips"""

        good_strips = np.arange(len(Noise))
        # Calculate the
        self.median_noise = np.median(Noise)
        high_noise_strips = np.nonzero(Noise > self.median_noise + Noise_cut)
        good_strips = np.delete(good_strips, high_noise_strips)

        return high_noise_strips[0], good_strips

    def noise_calc(self, events, pedestal, numevents, numchannels):
        """Noise calculation, normal noise (NN) and common mode noise (CMN)
        Uses numpy, can be further optimized by reducing memory access to
        member variables.
        But got 36k events per second.
        So fuck it.
        This function is not numba optimized!!!"""
        # Variable needed for noise calculations
        score = np.zeros((numevents, numchannels),
                         dtype=np.float32)
        CMnoise = np.zeros(numevents, dtype=np.float32)
        CMsig = np.zeros(numevents, dtype=np.float32)

        # Loop over all good events
        for event in tqdm(range(self.goodevents[0].shape[0]),
                          desc="Events processed:"):

            # Calculate the common mode noise for every channel
            # Get the signal from event and subtract pedestal
            cm = events[event][:] - pedestal
            CMNsig = np.std(cm)  # Calculate the standard deviation
            # Now calculate the mean from the cm to get the actual common mode
            # noise
            CMN = np.mean(cm)

            # Calculate the noise of channels
            # Subtract the common mode noise --> Signal[arraylike] -
            # pedestal[arraylike] - Common mode
            cn = cm - CMN

            score[event] = cn
            # Append the common mode values per event into the data arrays
            CMnoise[event] = CMN
            CMsig[event] = CMNsig

        return score, CMnoise, CMsig  # Return everything

    def plot_data(self):
        # COMMENT: every plot needs its own method!!!
        """Plots the data calculated by the framework"""

        fig = plt.figure("Noise analysis")

        # Plot noisedata
        noise_plot = fig.add_subplot(221)
        noise_plot.bar(np.arange(self.numchan), self.noise, 1., alpha=0.4,
                       color="b")
        # array of non masked strips
        valid_strips = np.ones(self.numchan)
        valid_strips[self.noisy_strips] = 0
        noise_plot.plot(np.arange(self.numchan), valid_strips, 1., color="r",
                        label="Masked strips")

        # Plot the threshold for deciding a good channel
        xval = [0, self.numchan]
        yval = [
            self.median_noise +
            self.configs.get(
                "Noise_cut",
                5.),
            self.median_noise +
            self.configs.get(
                "Noise_cut",
                5.)]
        noise_plot.plot(xval, yval, 1., "r--", color="g",
                        label="Threshold for noisy strips")

        noise_plot.set_xlabel('Channel [#]')
        noise_plot.set_ylabel('Noise [ADC]')
        noise_plot.set_title('Noise levels per Channel')
        noise_plot.legend()

        # Plot pedestal
        pede_plot = fig.add_subplot(222)
        pede_plot.bar(
            np.arange(
                self.numchan),
            self.pedestal,
            1.,
            yerr=self.noise,
            error_kw=dict(
                elinewidth=0.2,
                ecolor='r',
                ealpha=0.1),
            alpha=0.4,
            color="b")
        pede_plot.set_xlabel('Channel [#]')
        pede_plot.set_ylabel('Pedestal [ADC]')
        pede_plot.set_title('Pedestal levels per Channel with noise')
        pede_plot.set_ylim(bottom=min(self.pedestal) - 50.)
        # pede_plot.legend()

        # Plot Common mode
        CM_plot = fig.add_subplot(223)
        # COMMENT: pathces unused???
        n, bins, patches = CM_plot.hist(
            self.CMnoise, bins=50, density=True, alpha=0.4, color="b")
        # Calculate the mean and std
        mu, std = norm.fit(self.CMnoise)
        # Calculate the distribution for plotting in a histogram
        p = norm.pdf(bins, loc=mu, scale=std)
        CM_plot.plot(bins, p, "r--", color="g")

        CM_plot.set_xlabel('Common mode [ADC]')
        CM_plot.set_ylabel('[%]')
        CM_plot.set_title(r'$\mathrm{Common\ mode\:}\ \mu=' +
                          str(round(mu, 2)) +
                          r',\ \sigma=' +
                          str(round(std, 2)) +
                          r'$')
        # CM_plot.legend()

        # Plot noise hist
        CM_plot = fig.add_subplot(224)
        n, bins, patches = CM_plot.hist(
            self.total_noise, bins=500, density=False, alpha=0.4, color="b")
        CM_plot.set_yscale("log", nonposy='clip')
        CM_plot.set_ylim(1.)

        # Cut off noise part
        cut = np.max(n) * 0.2  # Find maximum of hist and get the cut
        # Finds the first element which is higher as threshold optimized
        ind = np.concatenate(np.argwhere(n > cut))

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
        # plt.draw()
