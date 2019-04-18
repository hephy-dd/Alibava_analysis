"""This file contains the basis analysis class for the ALiBaVa analysis"""
#pylint: disable=C0103
import logging
import numpy as np
import matplotlib.pyplot as plt
from analysis_classes.nb_analysis_funcs import parallel_event_processing
# pylint: disable=C0103
import logging

import matplotlib.pyplot as plt
import numpy as np

from analysis_classes.nb_analysis_funcs import parallel_event_processing

class BaseAnalysis:

    def __init__(self, main, events, timing, logger = None):
        self.log = logger or logging.getLogger(__class__.__name__)
        self.main = main
        self.events = events
        self.timing = timing
        self.prodata = None

    def run(self):
        """Does the actual event analysis"""

        # get events with good timinig only gtime and only process these events
        gtime = np.nonzero(np.logical_and(self.timing >= self.main.tmin, self.timing <= self.main.tmax))
        self.main.numgoodevents += int(gtime[0].shape[0])
        self.timing = self.timing[gtime]
        meanCMN = np.mean(self.main.CMN)
        meanCMsig = np.mean(self.main.CMsig)
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
        self.prodata = data
        self.main.automasked_hit = automasked_hits

        return self.prodata

