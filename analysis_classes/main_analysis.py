"""This file contains the main loop class that loops over the complete
run data"""
#pylint: disable=R0902,R0915,C0103,C0301

import logging
from multiprocessing import Pool
from time import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from .base_analysis import BaseAnalysis
from .utilities import Bdata, read_binary_Alibava, load_plugins
from .utilities import import_h5

class MainAnalysis:
    # COMMENT: the __init__ should be split up at least into 2 methods
    """This class analyses measurement files per event and conducts additional
    defined analysis"""
    def __init__(self, path, configs, logger=None):
        """
        :param path_list: List of pathes to analyse
        :param kwargs: kwargs if further data should be used, possible kwargs=calibration,noise
        """
        # Init parameters
        self.log = logger or logging.getLogger(__class__.__name__)

        self.log.info("Loading event file(s): %s", path)

        if not configs.get("isBinary", False):
            self.data = import_h5(path)
        else:
            self.data = read_binary_Alibava(path)

        self.numchan = len(self.data["events"]["signal"][0])
        self.numevents = len(self.data["events"]["signal"])
        self.pedestal = np.zeros(self.numchan, dtype=np.float32)
        # self.noise = np.zeros(self.numchan, dtype=np.float32)
        self.SN_cut = 1
        self.hits = 0
        self.tmin = 0
        self.tmax = 100
        self.maxcluster = 4
        self.CMN = np.zeros(self.numchan, dtype=np.float32)
        self.CMsig = np.zeros(self.numchan, dtype=np.float32)
        self.outputdata = {}
        self.automasked_hit = 0
        self.numgoodevents = 0
        self.total_events = self.numevents * len(self.data)
        self.additional_analysis = []
        self.start = time()
        self.noise_analysis = configs.get("noise_analysis", None)
        self.calibration = configs.get("calibration", None)

        self.pedestal = self.noise_analysis.pedestal
        self.CMN = self.noise_analysis.CMnoise
        self.CMsig = self.noise_analysis.CMsig
        self.noise = self.noise_analysis.noise
        self.SN_cut = configs["SN_cut"]  # Cut for the signal to noise ratio

        # For additional analysis
        self.add_analysis = configs.get("additional_analysis", [])
        if not self.add_analysis:
            self.add_analysis = []

        # Material decision
        self.material = configs.get("sensor_type", "n-in-p")
        if self.material == "n-in-p":
            self.material = 1
        else:
            self.material = 0  # Easier to handle

        self.masking = configs.get("automasking", False)
        self.max_clustersize = configs.get("max_cluster_size", 5)
        self.SN_ratio = configs.get("SN_ratio", 0.5)
        self.usejit = configs.get("optimize", False)
        self.SN_cluster = configs.get("SN_cluster", 6)

        # Create a pool for multiprocessing
        self.process_pool = configs.get("Processes", 1)  # How many workers
        self.Pool = Pool(processes=self.process_pool)

        if "timing" in configs:
            self.min = configs["timing"][0]  # timinig window
            self.max = configs["timing"][1]  # timing maximum

        self.log.info("Processing files ...")
        events = np.array(self.data["events"]["signal"][:], dtype=np.float32)
        timing = np.array(self.data["events"]["time"][:], dtype=np.float32)

        try:
            file = str(self.data).split('"')[1].split('.')[0]
        except:
            file = str(self.data)
        self.outputdata = {}
        # Todo: Make this loop work in a pool of processes/threads whichever is easier and better
        object = BaseAnalysis(self, events,
                              timing)  # you get back a list with events, containing the event processed data -->
                                       # np array makes it easier to slice
        results = object.run()

        self.outputdata["base"] = Bdata(results,
                                        labels=["Signal", "SN", "CMN", "CMsig",
                                                "Hitmap", "Channel_hit",
                                                "Clusters", "Numclus",
                                                "Clustersize"])

        # Now process additional analysis statet in the config file
        # Load all plugins
        plugins = load_plugins(configs.get("additional_analysis", []))
        for plugin in plugins.values():
            analysis = plugin(self, configs)
            analysis.run()
            self.outputdata[plugin] = results

        # In the end give a round up of all you have done
        print(\
            "*************************************************************************\n"
            "            Analysis report:                                             \n"
            "            ~~~~~~~~~~~~~~~~                                             \n"
            "                                                                         \n"
            "            Events processed:  {events!s}                                \n"
            "            Total events:      {total_events!s}                          \n"
            "            Time taken:        {time!s}                                  \n"
            "                                                                         \n"
            "*************************************************************************\n"\
            .format(automasked=self.automasked_hit,
                    events=self.numgoodevents,
                    total_events=self.total_events,
                    time=round((time() - self.start), 1)))

        # Add the noise results to the final dict
        self.outputdata["noise"] = {"pedestal": self.pedestal, "cmn": self.CMN, "cmnsig": self.CMsig,
                                    "noise": self.noise}

        self.Pool.close()
        self.Pool.join()
