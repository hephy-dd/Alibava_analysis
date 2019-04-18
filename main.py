"""Wrapper for full alibava analysis via console"""
import os
<<<<<<< HEAD
from argparse import ArgumentParser
from plot_data import PlotData
from analysis_classes.utilities import create_dictionary
from analysis_classes import Calibration
from analysis_classes import NoiseAnalysis
from analysis_classes import MainAnalysis
from analysis_classes.utilities import save_all_plots, save_dict, read_meas_files
=======
import logging
from optparse import OptionParser
#from analysis_classes.calibration import Calibration
#from analysis_classes.noise_analysis import NoiseAnalysis
#from analysis_classes.main_analysis import MainAnalysis
from analysis_classes.utilities import *
from cmd_shell import AlisysShell
from analysis_classes.utilities import create_dictionary, save_all_plots, save_configs
from analysis_classes.utilities import save_dict
from analysis_classes import Calibration
from analysis_classes import NoiseAnalysis
from analysis_classes import MainAnalysis
import matplotlib.pyplot as plt

log = logging.getLogger("main")
#log.setLevel(logging.INFO)
#if log.hasHandlers() is False:
#    format_string = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
#    formatter = logging.Formatter(format_string)
#    console_handler = logging.StreamHandler()
#    console_handler.setFormatter(formatter)
#    log.addHandler(console_handler)


def main(args, options):
    """The main analysis which will be executed after the arguments are parsed"""

    if options.shell:
        shell = AlisysShell()
        # shell.start_shell()

    elif options.configfile and os.path.exists(os.path.normpath(options.configfile)):
        log.info("Loading file: %s", os.path.normpath(options.configfile))
        configs = create_dictionary(os.path.normpath(options.configfile), "")
        do_with_config_file(configs)
        plt.show()  # Just in case the plot show has never been shown, it gets the plot items from the env
>>>>>>> Dominic_dev

DEF = os.path.join(os.getcwd(), "Examples", "marius_config.yml")

def main(args):
    """Start analysis"""
    if args.config != "":
        cfg = create_dictionary(args.config)
    else:
        cfg = create_dictionary(DEF)
    plot = PlotData()

    for ped, cal, run in read_meas_files(cfg):
        ped_data = NoiseAnalysis(ped,
                                 configs=cfg)
        plot.plot_data(ped_data, group="pedestal")

        cal_data = Calibration(cal, Noise_calc=ped_data,
                               isBinary=False, configs=cfg)
        plot.plot_data(cal_data, "calibration")

        cfg.update({"calibration": cal_data,
                    "noise_analysis": ped_data})

        run_data = MainAnalysis(run, configs=cfg)
        plot.plot_data(run_data, group="main")
        plot.plot_data(run_data, group="single_event")

        if cfg.get("Output_folder", "") and cfg.get("Output_name", ""):
            save_all_plots(cfg["Output_name"], cfg["Output_folder"], dpi=300)
            if cfg.get("Pickle_output", False):
                save_dict(run_data.outputdata,
                          os.path.join(cfg["Output_folder"],
                                       cfg["Output_name"], ".dba"))
    plot.show_plots()

if __name__ == "__main__":

    PARSER = ArgumentParser()
    PARSER.add_argument("--config",
                        help="The path to the config file for the analysis run",
                        default="")
    main(PARSER.parse_args())
