# This file is the main analysis file here all processes will be started

import os
from analysis_classes.utilities import create_dictionary
from analysis_classes import Calibration
from analysis_classes import NoiseAnalysis
from analysis_classes import MainAnalysis
from analysis_classes.utilities import save_all_plots, save_dict

CFG = create_dictionary("config.yml", os.path.join(os.getcwd(), "Examples"))

PED_DATA = NoiseAnalysis(os.path.join(os.getcwd(), "Examples",
                                      "pedestal_h5.h5"),
                         configs=CFG)
# PED_DATA = NoiseAnalysis(os.path.join(os.getcwd(), "Examples",
#                                       "ped_binary_RUN00251334.dat"),
#                          configs=CFG)
# PED_DATA.plot_data()

CAL_DATA = Calibration(charge_path=os.path.join(os.getcwd(), "Examples",
                                                "calibration_h5.h5"),
                       Noise_calc=PED_DATA)
# CAL_DATA.plot_data()

CFG.update({"calibration": CAL_DATA,
            "noise_analysis": PED_DATA})

RUN_DATA = MainAnalysis(CFG["Measurement_file"], configs=CFG)
if CFG.get("Output_folder", "") and CFG.get("Output_name", ""):
    save_all_plots(CFG["Output_name"], CFG["Output_folder"], dpi=300)
    if CFG.get("Pickle_output", False):
        save_dict(RUN_DATA.outputdata,
                  os.path.join(CFG["Output_folder"],
                               CFG["Output_name"], ".dba"))
print(RUN_DATA)
