import os
from plot_data import PlotData
from analysis_classes.utilities import create_dictionary
from analysis_classes import Calibration
from analysis_classes import NoiseAnalysis
from analysis_classes import MainAnalysis
from analysis_classes.utilities import save_all_plots, save_dict

if __name__ == "__main__":

    CFG = create_dictionary("marius_config.yml",
                            os.path.join(os.getcwd(), "Examples"))
    PLOT = PlotData()

    PED_FILES = CFG["Pedestal_file"]
    if CFG["use_charge_cal"]:
        CAL_FILES = CFG["Charge_scan"]
    else:
        CAL_FILES = CFG["Delay_scan"]
    MAIN_FILES = CFG["Measurement_file"]

    if not len(PED_FILES) == len(MAIN_FILES) or \
            not len(CAL_FILES) == len(MAIN_FILES):
        raise ValueError("Number of pedestal, calibration and measurement files"
                         " does not match...")

    for PED, CAL, MAIN in zip(PED_FILES,
                              CAL_FILES,
                              MAIN_FILES):
        PED_DATA = NoiseAnalysis(os.path.join(os.getcwd(), "Examples", PED),
                                 configs=CFG)
        PLOT.plot_data(PED_DATA, group="pedestal")

        CAL_DATA = Calibration(CAL, Noise_calc=PED_DATA,
                               isBinary=False, configs=CFG)
        PLOT.plot_data(CAL_DATA, "calibration")

        CFG.update({"calibration": CAL_DATA,
                    "noise_analysis": PED_DATA})

        RUN_DATA = MainAnalysis(MAIN, configs=CFG)
        PLOT.plot_data(RUN_DATA, group="main")

    # if CFG.get("Output_folder", "") and CFG.get("Output_name", ""):
    #     save_all_plots(CFG["Output_name"], CFG["Output_folder"], dpi=300)
        # if CFG.get("Pickle_output", False):
        #     save_dict(RUN_DATA.outputdata,
        #               os.path.join(CFG["Output_folder"],
        #                            CFG["Output_name"], ".dba"))

    PLOT.show_plots()
