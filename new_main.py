# This file is the main analysis file here all processes will be started

import os
# from optparse import OptionParser
# import matplotlib.pyplot as plt
from analysis_classes.utilities import create_dictionary
from analysis_classes import Calibration
from analysis_classes import NoiseAnalysis
# from analysis_classes.main_analysis import MainAnalysis

CFG = create_dictionary("def_config.yml", os.path.join(os.getcwd(), "Examples"))

PED_DATA = NoiseAnalysis(os.path.join(os.getcwd(), "Examples",
                                      "pedrun_100V.h5"),
                         configs=CFG)
# PED_DATA = NoiseAnalysis(os.path.join(os.getcwd(), "Examples",
#                                       "ped_binary_RUN00251334.dat"),
#                          configs=CFG)
# PED_DATA.plot_data()

CAL_DATA = Calibration(charge_path=os.path.join(os.getcwd(), "Examples",
                                                "calibcharge_3.h5"),
                       Noise_calc=PED_DATA)
CAL_DATA.plot_data()
