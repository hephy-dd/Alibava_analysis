"""This file contains the main loop class that loops over the complete
run data"""
#pylint: disable=R0902,R0915,C0103

import logging
from time import time
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
from analysis_classes.base_analysis import BaseAnalysis
from utilities import import_h5, Bdata

class MainLoops:
    # COMMENT: the __init__ should be split up at least into 2 methods
    """This class analyses measurement files per event and conducts additional
    defined analysis"""

    def __init__(self, path_list=None, **kwargs):
        """

        :param path_list: List of pathes to analyse
        :param kwargs: kwargs if further data should be used,
                       possible kwargs=calibration,noise
        """
        self.log = logging.getLogger(__class__.__name__)
        self.log.setLevel(logging.DEBUG)
        if self.log.hasHandlers() is False:
            format_string = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
            formatter = logging.Formatter(format_string)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.log.addHandler(console_handler)

        self.log.info("Loading event file(s) from: %s", path_list)
        self.data = import_h5(path_list)

        # Init parameters
        self.numchan = len(self.data[0]["events/signal"][0])
        self.numevents = len(self.data[0]["events/signal"])
        self.pedestal = np.zeros(self.numchan, dtype=np.float32)
        self.noise = np.zeros(self.numchan, dtype=np.float32)
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
        self.pathes = path_list
        self.kwargs = kwargs
        self.noise_analysis = kwargs["configs"].get("noise_analysis", None)

        if "configs" in kwargs:
            # If a config was passeds it has to be a dict containig all
            # settings therefore kwargs rewritten
            kwargs = kwargs["configs"]

        # For additional analysis
        self.add_analysis = kwargs.get("additional_analysis", [])
        if not self.add_analysis:
            self.add_analysis = []

        # Material decision
        self.material = kwargs.get("sensor_type", "n-in-p")
        if self.material == "n-in-p":
            self.material = 1
        else:
            self.material = 0

        self.masking = kwargs.get("automasking", False)
        self.max_clustersize = kwargs.get("max_cluster_size", 5)
        self.SN_ratio = kwargs.get("SN_ratio", 0.5)
        self.usejit = kwargs.get("optimize", False)
        self.calibration = kwargs.get("calibration", None)
        self.SN_cluster = kwargs.get("SN_cluster", 6)
        self.process_pool = kwargs.get("Processes", 1)
        self.Pool = Pool(processes=self.process_pool)

        if "pedestal" in kwargs:
            self.pedestal = kwargs["pedestal"]

        if "SN_cut" in kwargs:
            self.SN_cut = kwargs["SN_cut"]  # Cut for the signal to noise ratio

        if "CMN" in kwargs:
            self.CMN = kwargs["CMN"]  # CMN for every channel and event

        if "CMsig" in kwargs:
            self.CMsig = kwargs["CMsig"]  # Common mode sig for every channel

        if "Noise" in kwargs:
            self.noise = kwargs["Noise"]  # Noise for every channel and event

        if "timing" in kwargs:
            self.min = kwargs["timing"][0]  # timinig window
            self.max = kwargs["timing"][1]  # timing maximum


        self.log.info("Processing files ...")
        # Here a loop over all files will be done to do the analysis on all
        # imported files
        for data in tqdm(range(len(self.data)), desc="Data files processed:"):
            events = np.array(self.data[data]["events/signal"][:],
                              dtype=np.float32)
            timing = np.array(self.data[data]["events/time"][:],
                              dtype=np.float32)

            file = str(self.data[data]).split('"')[1].split('.')[0]
            self.outputdata[file] = {}
            # Todo: Make this loop work in a pool of processes/threads
            # whichever is easier and better
            # you get back a list with events, containing the event processed
            # data --> np array makes it easier to slice
            base_ana = BaseAnalysis(self, events, timing)
            results = base_ana.run()
            # print(results[:,7])
            # make the data easy accessible: results(array) --> entries are events --> containing data eg indes 0 ist signal
            # So now order the data Dictionary --> Filename:Type of data: List of all events for specific data type ---> results[: (take all events), 0 (give me data from signal]
            # Resulting is an array containing all singal data etc.
            # self.outputdata[file] =                                             {"Signal": results[:,0],
            #                                                                     "SN": results[:, 1],
            #                                                                     "CMN": results[:, 2],
            #                                                                     "CMsig": results[:, 3],
            #                                                                     "Hitmap": results[:, 4],
            #                                                                     "Channel_hit": results[:, 5],
            #                                                                     "Clusters": results[:, 6],
            #                                                                     "Numclus": results[:, 7],
            #                                                                     "Clustersize": results[:, 8],}
            # print(self.outputdata[file]["Numclus"])
            self.outputdata[file]["base"] = Bdata(\
                    results,
                    labels=["Signal", "SN", "CMN", "CMsig", "Hitmap",
                            "Channel_hit", "Clusters", "Numclus",
                            "Clustersize"])
            # print(get_size(self.outputdata[file]))
            #a = bd["Numclus"]
            # print(a)
        # Not very pythonic, loop inside analysis (legacy)
        base_ana.plot_data(single_event=kwargs.get("Plot_single_event", 15))
        # Now process additional analysis statet in the config file
        for analysis in self.add_analysis:
            self.log.info("Starting analysis: %s", analysis)
            # Gets the total analysis class, so be aware of changes inside!!!
            # COMMENT: using eval is frowned upon in modern python...
            add_analysis = eval(analysis)(self)
            results = add_analysis.run()
            add_analysis.plot()
            if results:  # Only if results have been returned
                for file in results:
                    self.outputdata[file][str(analysis)] = results[file]

        # In the end give a round up of all you have done
        print(
"*************************************************************************\n"
"            Analysis report:                                             \n"
"            ~~~~~~~~~~~~~~~~                                             \n"
"                                                                         \n"
"            Automasked hits:   {automasked!s}                            \n"
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
        self.outputdata["noise"] = {"pedestal": self.pedestal,
                                    "cmn": self.CMN,
                                    "cmnsig": self.CMsig,
                                    "noise": self.noise}

        self.Pool.close()
        self.Pool.join()
