"""This file contains the main loop class that loops over the complete
run data"""
#pylint: disable=R0902,R0915,C0103,C0301

import logging
from multiprocessing import Pool
from time import time
import numpy as np
from .base_analysis import BaseAnalysis
from .utilities import Bdata, read_binary_Alibava, load_plugins
from .utilities import import_h5

class MainAnalysis:
    # COMMENT: the __init__ should be split up at least into 2 methods
    """MainAnalysis simply handles all logic to perform the complete analysis.
    It first conducts the BaseAnalysis - Preprocessing and Clustering
    Afterwards if conducts all analysis specified in the configs file.
    It does not have any fancy algorithms in it.

    """
    def __init__(self, path, configs, logger=None):
        """MainAnalysis simply handles all logic to perform the complete analysis.
           It first conducts the BaseAnalysis - Preprocessing and Clustering
           Afterwards if conducts all analysis specified in the configs file.
           It does not have any fancy algorithms in it.

        Config params:
            - isBinary: bool - Whether or not the input file is AliBaVa binary or HDF5
            - additional_analysis: list - containing the names of the analysises which should be done
            - Processes: int number of pool size for multiprocessing

        """

        # Init parameters
        self.log = logger or logging.getLogger(__class__.__name__)
        self.start = time()

        self.log.info("Loading event file(s): %s", path)
        if not configs.get("isBinary", False):
            self.data = import_h5(path)
        else:
            self.data = read_binary_Alibava(path)

        self.outputdata = {}
        self.results = self.outputdata
        self.configs = self.configure_configs(configs)
        self.configs_dict = configs

        # Get the objects from calibration and Noise etc.
        self.noise_analysis = configs.get("noise_analysis", None)
        self.calibration = configs.get("calibration", None)
        self.pedestal = self.noise_analysis.pedestal
        self.CMN = self.noise_analysis.CMnoise
        self.CMsig = self.noise_analysis.CMsig
        self.noise = self.noise_analysis.noise


        # Load some crucial parameters from the config
        # Get the additional anlysises which should be done
        self.add_analysis = configs.get("additional_analysis", [])
        if not self.add_analysis:
            self.add_analysis = []

        # Material decision
        self.material = configs.get("sensor_type", "n-in-p")
        if self.material == "n-in-p":
            self.material = 1
        else:
            self.material = 0  # Easier to handle

        # Create a pool for multiprocessing
        self.process_pool = configs.get("Processes", 1)  # How many workers
        self.Pool = Pool(processes=self.process_pool)

        self.log.info("Processing file ...")
        self.events = np.array(self.data["events"]["signal"][:], dtype=np.float32)
        self.timing = np.array(self.data["events"]["time"][:], dtype=np.float32)

        try:
            file = str(self.data).split('"')[1].split('.')[0]
        except:
            file = str(self.data)

        # Add the noise results to the final dict
        self.outputdata["noise"] = {"pedestal": self.pedestal, "cmn": self.CMN, "cmnsig": self.CMsig,
                                    "noise": self.noise}

        # Start the base analysis with clustering
        _object = BaseAnalysis(self, self.events, self.timing)
        results = _object.run()

        self.outputdata["base"] = Bdata(results,
                                        labels=["Signal", "SN", "CMN", "CMsig",
                                                "Hitmap", "Channel_hit",
                                                "Clusters", "Numclus",
                                                "Clustersize", "Timing"])

        # Now process additional analysis stated in the config file
        # Load all plugins
        plugins = load_plugins(configs.get("additional_analysis", []))
        for name, plugin in plugins.items():
            self.log.info("Starting analysis: {}".format(name))
            analysis = plugin(self, configs.get(name, {}))
            self.outputdata[name] = analysis.run()

        # In the end give a round up of all you have done
        print(\
            "*************************************************************************\n"
            "            Analysis report:                                             \n"
            "            ~~~~~~~~~~~~~~~~                                             \n"
            "                                                                         \n"
            "            Events processed:  {events!s}                                \n"
            "            Total events:      42                          \n"
            "            Time taken:        {time!s}                                  \n"
            "                                                                         \n"
            "*************************************************************************\n"\
            .format(automasked=42,
                    events=len(self.outputdata["base"]["Signal"]),
                    time=round((time() - self.start), 1)))

        # Close the pool
        self.Pool.close()
        self.Pool.join()


    def configure_configs(self, configs):
        """Takes every parent entry in the configs dict and makes a object for
        the main class"""

        for name, value in configs.items():
            setattr(self, name, value)
