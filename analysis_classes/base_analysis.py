"""This file contains the basis analysis class for the ALiBaVa analysis"""
#pylint: disable=C0103
import logging
from time import time
import gc
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from nb_analysisFunction import parallel_event_processing

class BaseAnalysis:
    """Class doc"""
    def __init__(self, main, events, timing):

        self.log = logging.getLogger(__class__.__name__)
        self.log.setLevel(logging.DEBUG)
        if self.log.hasHandlers() is False:
            format_string = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
            formatter = logging.Formatter(format_string)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.log.addHandler(console_handler)

        self.main = main
        self.events = events
        self.timing = timing

    def run(self):
        """Does the actual event analysis"""
        # get events with good timinig only gtime and only process these events
        gtime = np.nonzero(self.timing > self.main.tmin)
        self.main.numgoodevents += int(gtime[0].shape[0])
        meanCMN = np.mean(self.main.CMN)
        meanCMsig = np.mean(self.main.CMsig)
        prodata = []  # List of processed data which then can be accessed
        hitmap = np.zeros(self.main.numchan)
        # Warning: If you have a RS and pulseshape recognition enabled the
        # timing window has to be set accordingly

        if not self.main.usejit:
            # Non jitted version
            # COMMENT: start unused???
            start = time()
            iter_item = 0
            # Loop over all good events
            for event in tqdm(range(gtime[0].shape[0]),
                              desc="Events processed:"):
                # Event and Cluster Calculations
                iter_item += 1
                if iter_item == 1000:
                    gc.collect()
                    iter_item = 0
                signal, SN, CMN, CMsig = self.process_event(\
                        self.events[event], self.main.pedestal,
                        meanCMN, meanCMsig, self.main.noise, self.main.numchan)
                channels_hit, clusters, numclus, clustersize = self.clustering(\
                        signal, SN, self.main.noise)
                for channel in channels_hit:
                    hitmap[int(channel)] += 1

                prodata.append([signal, SN, CMN, CMsig, hitmap, channels_hit,
                                clusters, numclus, clustersize])

        else:
            start = time()
            # This should, in theory, use parallelization of the loop over
            # event but i did not see any performance boost, maybe you can find
            # the bug =)?
            data, automasked_hits = parallel_event_processing(\
                    gtime,
                    self.events,
                    self.main.pedestal,
                    meanCMN,
                    meanCMsig,
                    self.main.noise,
                    self.main.numchan,
                    self.main.SN_cut,
                    self.main.SN_ratio,
                    self.main.SN_cluster,
                    max_clustersize=self.main.max_clustersize,
                    masking=self.main.masking,
                    material=self.main.material,
                    poolsize=self.main.process_pool,
                    Pool=self.main.Pool,
                    noisy_strips=self.main.noise_analysis.noisy_strips)
            prodata = data
            self.main.automasked_hit = automasked_hits

        # COMMENT: end unused???
        end = time()
        return prodata

    def clustering(self, event, SN, Noise):
        """Looks for cluster in a event"""
        # Only channels which have a signal/Noise higher then the signal/Noise cut
        channels = np.nonzero(np.abs(SN) > self.main.SN_cut)[0]
        valid_ind = np.arange(len(event))

        if self.main.masking:
            if self.main.material:
                # Todo: masking of dead channels etc.
                # So only negative values are considered
                masked_ind = np.nonzero(np.take(event, channels) > 0)[0]
                # Find out which index are negative so we dont count them
                # accidently
                valid_ind = np.nonzero(event < 0)[0]
                # COMMENT: if masked_ind: would be more pythonic
                if len(masked_ind):
                    channels = np.delete(channels, masked_ind)
                    self.main.automasked_hit += len(masked_ind)
            else:
                masked_ind = np.nonzero(np.take(event, channels) < 0)[
                    0]  # So only positive values are considered
                valid_ind = np.nonzero(event > 0)[0]
                if len(masked_ind):
                    channels = np.delete(channels, masked_ind)
                    self.main.automasked_hit += len(masked_ind)

        # To keep track which channel have been used already
        used_channels = np.zeros(self.main.numchan)
        numclus = 0  # The number of found clusters
        clusters_list = []
        clustersize = np.array([])
        # Loop over all left channels which are a hit, here from "left" to "right"
        for ch in channels:
            if not used_channels[ch]:  # Make sure we dont count everything twice
                used_channels[ch] = 1  # So now the channel is used
                cluster = [ch]  # Keep track of the individual clusters
                size = 1

                right_stop = False
                left_stop = False
                # Now make a loop to find neighbouring hits of cluster, we must
                # go into both directions
                offset = int(self.main.max_clustersize * 0.5)
                # Search plus minus the channel found Todo: first entry useless
                for i in range(1, offset + 1):
                    # To exclude overrun
                    if 0 < ch - i and ch + i < self.main.numchan:
                        if np.abs(SN[ch + i]) \
                                > self.main.SN_cut * self.main.SN_ratio \
                                and not used_channels[ch + i] \
                                and ch + i in valid_ind and not right_stop:
                            cluster.append(ch + i)
                            used_channels[ch + i] = 1
                            size += 1
                        elif np.abs(SN[ch + i]) \
                                < self.main.SN_cut * self.main.SN_ratio:
                            # Prohibits search for to long clusters
                            right_stop = True

                        if np.abs(SN[ch - i]) \
                                > self.main.SN_cut * self.main.SN_ratio \
                                and not used_channels[ch - i] \
                                and ch - i in valid_ind and not left_stop:
                            cluster.append(ch - i)
                            used_channels[ch - i] = 1
                            size += 1
                        elif np.abs(SN[ch - i]) \
                                < self.main.SN_cut * self.main.SN_ratio:
                            # Prohibits search for to long clusters
                            left_stop = True

                # Now make a loop to find neighbouring hits of cluster, we must go into both directions
                # TODO huge clusters can be misinterpreted!!! Takes huge amount of cpu, vectorize
                #offset = int(self.max_clustersize / 2)
                # for i in range(ch-offset, ch+offset): # Search plus minus the channel found
                #    if 0 < i < self.numchan: # To exclude overrun
                #            if np.abs(SN[i]) > self.SN_cut * self.SN_ratio and not used_channels[i] and i in valid_ind:
                #                cluster.append(i)
                #                used_channels[i] = 1
                #                # Append the channel which is also hit after this estimation
                #                size += 1

                # Look if the cluster SN is big enough to be counted as
                # clusters
                Scluster = np.abs(np.sum(np.take(event, cluster)))
                Ncluster = np.sqrt(np.abs(np.sum(np.take(Noise, cluster))))
                SNcluster = Scluster / Ncluster  # Actual signal to noise of cluster
                if SNcluster > self.main.SN_cluster:
                    numclus += 1
                    clusters_list.append(cluster)
                    clustersize = np.append(clustersize, size)

        # warning channels are only the channels which are above SN
        return channels, clusters_list, numclus, clustersize

    def process_event(self, event, pedestal, meanCMN, meanCMsig,
                      noise, numchan=256):
        """Processes single events"""

        # Calculate the common mode noise for every channel
        signal = event - pedestal  # Get the signal from event and subtract pedestal

        # Mask noisy strips, by setting every noisy channel to 0 --> SN is
        # always 0
        signal[self.main.noise_analysis.noisy_strips] = 0

        # Remove channels which have a signal higher then 5*CMsig+CMN which are
        # not representative

        # Processed signal
        prosignal = np.take(signal,
                            np.nonzero(signal < (5 * meanCMsig + meanCMN)))

        if prosignal.any():
            cmpro = np.mean(prosignal)
            sigpro = np.std(prosignal)

            corrsignal = signal - cmpro
            SN = corrsignal / noise

            return corrsignal, SN, cmpro, sigpro
        # A default value return if everything fails
        return np.zeros(numchan), np.zeros(numchan), 0, 0

    def plot_data(self, single_event=-1):
        """This function plots all data processed"""
        # COMMENT: every plot needs its own method!!!
        for name, data in self.main.outputdata.items():
            # Plot a single event from every file
            if single_event > 0:
                self.plot_single_event(single_event, name)

            # Plot Analysis results
            fig = plt.figure("Analysis file: {!s}".format(name))

            # Plot Hitmap
            channel_plot = fig.add_subplot(211)
            channel_plot.bar(np.arange(self.main.numchan), data["base"]["Hitmap"][len(
                data["base"]["Hitmap"]) - 1], 1., alpha=0.4, color="b")
            channel_plot.set_xlabel('channel [#]')
            channel_plot.set_ylabel('Hits [#]')
            channel_plot.set_title('Hitmap from file: {!s}'.format(name))

            fig.tight_layout()

            # Plot Clustering results
            fig = plt.figure("Clustering Analysis on file: {!s}".format(name))

            # Plot Number of clusters
            numclusters_plot = fig.add_subplot(221)
            bin_index, counts = np.unique(data["base"]["Numclus"],
                                          return_counts=True)
            numclusters_plot.bar(bin_index, counts, alpha=0.4, color="b")
            numclusters_plot.set_xlabel('Number of clusters [#]')
            numclusters_plot.set_ylabel('Occurance [#]')
            numclusters_plot.set_title('Number of clusters')

            # Plot clustersizes
            clusters_plot = fig.add_subplot(222)
            # Todo: make it possible to count clusters in multihit scenarios
            bin_index, counts = np.unique(
                np.concatenate(
                    data["base"]["Clustersize"]), return_counts=True)
            clusters_plot.bar(bin_index, counts, alpha=0.4, color="b")
            clusters_plot.set_xlabel('Clustersize [#]')
            clusters_plot.set_ylabel('Occurance [#]')
            clusters_plot.set_title('Clustersizes')

            fig.suptitle('Cluster analysis from file {!s}'.format(name))
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            # plt.draw()

    def plot_single_event(self, eventnum, file):
        """ Plots a single event and its data"""

        data = self.main.outputdata[file]

        fig = plt.figure(
            "Event number {!s}, from file: {!s}".format(
                eventnum, file))

        # Plot signal
        channel_plot = fig.add_subplot(211)
        channel_plot.bar(
            np.arange(
                self.main.numchan),
            data["base"]["Signal"][eventnum],
            1.,
            alpha=0.4,
            color="b")
        channel_plot.set_xlabel('channel [#]')
        channel_plot.set_ylabel('Signal [ADC]')
        channel_plot.set_title('Signal')

        # Plot signal/Noise
        SN_plot = fig.add_subplot(212)
        SN_plot.bar(
            np.arange(
                self.main.numchan),
            data["base"]["SN"][eventnum],
            1.,
            alpha=0.4,
            color="b")
        SN_plot.set_xlabel('channel [#]')
        SN_plot.set_ylabel('Signal/Noise [ADC]')
        SN_plot.set_title('Signal/Noise')

        fig.suptitle(
            'Single event analysis from file {!s}, with event: {!s}'.format(
                file, eventnum))
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        # plt.draw()
