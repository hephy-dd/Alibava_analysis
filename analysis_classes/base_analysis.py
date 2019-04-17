"""This file contains the basis analysis class for the ALiBaVa analysis"""
#pylint: disable=C0103
import logging
import gc
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from analysis_classes.nb_analysis_funcs import parallel_event_processing
#from .utilities import manage_logger

class BaseAnalysis:

    def __init__(self, main, events, timing, logger = None):
        self.log = logger or logging.getLogger(__class__.__name__)
        #manage_logger(self.log)
        self.main = main
        self.events = events
        self.timing = timing

    def run(self):
        """Does the actual event analysis"""

        # get events with good timinig only gtime and only process these events
        gtime = np.nonzero(self.timing >= self.main.tmin)
        self.main.numgoodevents += int(gtime[0].shape[0])
        meanCMN = np.mean(self.main.CMN)
        meanCMsig = np.mean(self.main.CMsig)
        prodata = []  # List of processed data which then can be accessed
        hitmap = np.zeros(self.main.numchan)
        # Warning: If you have a RS and pulseshape recognition enabled the
        # timing window has to be set accordingly

        # This should, in theory, use parallelization of the loop over event
        # but i did not see any performance boost, maybe you can find the bug =)?
        data, automasked_hits = parallel_event_processing(gtime,
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

        return prodata

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
            channel_plot.bar(np.arange(self.main.numchan),
                             data["base"]["Hitmap"][len(data["base"]["Hitmap"]) - 1], 1.,
                             alpha=0.4, color="b")
            channel_plot.set_xlabel('channel [#]')
            channel_plot.set_ylabel('Hits [#]')
            channel_plot.set_title('Hitmap from file: {!s}'.format(name))

            fig.tight_layout()

            # Plot Clustering results
            fig = plt.figure("Clustering Analysis on file: {!s}".format(name))

            # Plot Number of clusters
            numclusters_plot = fig.add_subplot(221)
            bins, counts = np.unique(data["base"]["Numclus"], return_counts=True)
            numclusters_plot.bar(bins, counts, alpha=0.4, color="b")
            numclusters_plot.set_xlabel('Number of clusters [#]')
            numclusters_plot.set_ylabel('Occurance [#]')
            numclusters_plot.set_title('Number of clusters')
            # numclusters_plot.set_yscale("log", nonposy='clip')

            # Plot clustersizes
            clusters_plot = fig.add_subplot(222)
            bins, counts = np.unique(np.concatenate(data["base"]["Clustersize"]), return_counts=True)
            clusters_plot.bar(bins, counts, alpha=0.4, color="b")
            clusters_plot.set_xlabel('Clustersize [#]')
            clusters_plot.set_ylabel('Occurance [#]')
            clusters_plot.set_title('Clustersizes')
            # clusters_plot.set_yscale("log", nonposy='clip')

            fig.suptitle('Cluster analysis from file {!s}'.format(name))
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            # plt.draw()

    def plot_single_event(self, eventnum, file):
        """ Plots a single event and its data"""

        data = self.main.outputdata[file]

        fig = plt.figure("Event number {!s}, from file: {!s}".format(eventnum, file))

        # Plot signal
        channel_plot = fig.add_subplot(211)
        channel_plot.bar(np.arange(self.main.numchan),
                         data["base"]["Signal"][eventnum], 1.,
                         alpha=0.4, color="b")
        channel_plot.set_xlabel('channel [#]')
        channel_plot.set_ylabel('Signal [ADC]')
        channel_plot.set_title('Signal')

        # Plot signal/Noise
        SN_plot = fig.add_subplot(212)
        SN_plot.bar(np.arange(self.main.numchan),
                    data["base"]["SN"][eventnum], 1.,
                    alpha=0.4, color="b")
        SN_plot.set_xlabel('channel [#]')
        SN_plot.set_ylabel('Signal/Noise [ADC]')
        SN_plot.set_title('Signal/Noise')

        fig.suptitle('Single event analysis from file {!s}, with event: {!s}'.format(file, eventnum))
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        # plt.draw()
