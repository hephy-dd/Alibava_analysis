"""This file contains the basis analysis class for the ALiBaVa analysis"""
#pylint: disable=C0103
import logging
import numpy as np
from analysis_classes.nb_analysis_funcs import parallel_event_processing

class BaseAnalysis:
    """BaseAnalysis handles the basic clustering analysis of all passed events.
    It looks for good timing events, the timing window can be specified.

    Warning: BaseAnalysis benefits of multiprocessing and vectorization by the numba
    jit module. Therefore, no calculations are done in this script. For more
    information on how the different parts of the algorithm exactly are working,
    please refer to the nb_analysis_funcs.py file.

    That said, I will explain how in principal the algorithm is working:

    Basic Clustering algorithm:
        - Before we do anything we have to subtract the pedestal from our signal.
          This will result in a (hopefully) zero'ish signal for most channels (except
          the ones which are hit).
          Otherwise we could not distinguish between signal and the baseline signal.
          Next step is to remove events which show channels with extreme high signals
          noise. This is done by simply cutting events showing signal > 5*CMN*CMsig
          CMN - Common Mode Noise, CMsig - Common mode standard deviation.
          These events are dominated by common mode noise and would result in false
          positive hits.
          Whit basic garbage clean up we calculate the mean signal and std of every
          event. We will subtract the mean of all channels, to get rid of common mode
          in the event. We then build the ratio between the residual signal vs. the
          noise for EVERY channel. --> We have our SN for every channel.
          This is all done in the nb_preprocess_all_events function
        - We then are looking for a so called seed cut. Meaning we are looking for
          channels which show a higher Signal to Noise - SN as the specified one
          in the configs. Next in line we need to get rid of false polarized signals,
          which slipped our previous clean up. This is simply done by only considering
          correct polarized signals.
        - Now the fun begins: We are taking every hit from the SN cut and look
          left and right, to find neighbouring hits which are below the SN cut.
          Therefore, we have the parameter SN_ratio, which applies a factor to the SN_cut.
          If the neighbour strips are obove this threshold it will be considered as hit too
          and will be added to the cluster. This goes on until no strips are above this threshold.
          Finally we calculate the SN for the whole cluster which in turn has to be above
          as specified value, otherwise the cluster gets rejected.
          Warning: In reality this is not trivial to do and I therefore refer to the
          dedicated function: nb_clustering
        - Finally all data has been processed and we have finished clustering

        The data structure this algorithm returns you is as follows:

        return is of type numpy.array:
            [0] = processed signal: shape = (events, channels)
            [1] = SN: shape = (events, channels)
            [2] = CMN: shape = (events)
            [3] = CMsig: shape = (events)
            [4] = Hitmap: shape = (events, channels)
            [5] = Channels hit: shape = (hitted channels)
            [6] = Clusters: shape = (Channels hit shape = (channels in cluster))
            [7] = Number of Clusters: shape = (events)
            [8] = Clustersize: shape = (Channels hit: shape = (len(Clusters))
            [9] = Timing: shape = (events)


        # Base Analysis specific params
            - timing: [min, max] - Minimum/Maximum timing window
            - sensor_type: "n-in-p" or "p-in-n"
            - automasking: bool - if automasking of false polarized signals will be done
            - SN_cut: float - SN ratio at which it is considered a hit
            - SN_ratio: float - ratio of SN_cut to look for neighbouring hits
            - SN_cluster: float - Minimum SN of a cluster to be considered
            - numchan: int - Number of channels
            - max_cluster_size: int - maximum clustersize to look for

    Written by Dominic Bloech

    """

    def __init__(self, main, events, timing, logger = None):
        """

        :param main: MainAnalysis instance for additional paramerters if needed
        :param events: The actual events must be of shape = (events, channels)
        :param timing: an array of all timing for every event (must be the same length
                       as events parameter!!!
        :param logger: If you want to pass a specific logger you can do so
        """
        self.log = logger or logging.getLogger(__class__.__name__)
        self.main = main
        self.events = events
        self.eventtiming = timing
        self.prodata = None



    def run(self):
        """Does the actual event analysis and clustering in optimized python"""

        # Get events with good timing and only process these events
        gtime = np.nonzero(np.logical_and(self.eventtiming >= self.main.timingWindow[0],
                                          self.eventtiming <= self.main.timingWindow[1]))
        #self.eventtiming = self.eventtiming[gtime]

        # Warning: If you have a RS and pulseshape recognition enabled the
        # timing window has to be set accordingly

        # This should, in theory, use parallelization of the loop over event
        # but i did not see any performance boost, maybe you can find the bug =)?
        data, automasked_hits = parallel_event_processing(gtime,
                                                              self.eventtiming,
                                                              self.events,
                                                              self.main.pedestal,
                                                              np.mean(self.main.CMN),
                                                              np.mean(self.main.CMsig),
                                                              self.main.noise,
                                                              self.main.numChan,
                                                              self.main.SN_cut,
                                                              self.main.SN_ratio,
                                                              self.main.SN_cluster,
                                                              max_clustersize=self.main.max_cluster_size,
                                                              masking=self.main.automasking,
                                                              material=self.main.material,
                                                              poolsize=self.main.process_pool,
                                                              Pool=self.main.Pool,
                                                              noisy_strips=self.main.noise_analysis.noisy_strips)
        self.prodata = data
        self.main.automasked_hit = automasked_hits

        return self.prodata
