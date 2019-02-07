# This file is the main analysis file here all processes will be started

import os
# from optparse import OptionParser
# import matplotlib.pyplot as plt
from utilities import create_dictionary
# from analysis_classes.calibration import Calibration
from analysis_classes import NoiseAnalysis
# from analysis_classes.main_analysis import MainAnalysis

CFG = create_dictionary("def_config.yml", os.path.join(os.getcwd(), "Examples"))

PED_DATA = NoiseAnalysis(os.path.join(os.getcwd(), "Examples",
                                      "pedrun_100V.h5"),
                                      # "ped_binary_RUN00251334.dat"),
                         # data_type="binary",
                         data_type="hdf5",
                         configs=CFG)
PED_DATA.plot_data()
