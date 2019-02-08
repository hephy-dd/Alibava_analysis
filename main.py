"""This file is the main analysis file here all processes will be started"""

import os
import logging
# COMMENT: switch to argparse
from optparse import OptionParser
from cmd_shell import AlisysShell
from analysis_classes.utilities import create_dictionary, save_all_plots
from analysis_classes.utilities import save_dict
from analysis_classes import Calibration
from analysis_classes import NoiseAnalysis
from analysis_classes import MainAnalysis

log = logging.getLogger("main")
log.setLevel(logging.INFO)
if log.hasHandlers() is False:
    format_string = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    formatter = logging.Formatter(format_string)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)


def main(args, options):
    """The main analysis which will be executed after the arguments are parsed"""

    if options.shell:
        shell = AlisysShell()
        # shell.start_shell()

    elif options.configfile and os.path.exists(os.path.normpath(options.configfile)):
        log.info("Loading file: %s", os.path.normpath(options.configfile))
        configs = create_dictionary(os.path.normpath(options.configfile), "")
        do_with_config_file(configs)
        # COMMETN: plt not defined!!!
        # plt.show()  # Just in case the plot show has never been shown

    elif options.filepath and os.path.exists(os.path.normpath(options.filepath)):
        pass  # Todo: include the option to start the analysis with a passed file and config file

    else:
        log.error("No valid path parsed! Exiting")
        exit(1)

def do_with_config_file(config):
    """Starts analysis with a config file"""

    # Look if a pedestal file is specified
    if "Pedestal_file" in config:
        noise_data = NoiseAnalysis(config["Pedestal_file"], configs=config)
        noise_data.plot_data()

    # Look if a calibration file is specified
    if "Delay_scan" in config or "Charge_scan" in config:
        config_data = Calibration(config.get("Delay_scan", ""), config.get("Charge_scan", ""), Noise_calc=noise_data,
                                  isBinary=config.get("isBinary", False))
        config_data.plot_data()

    # Look if a pedestal file is specified
    if "Measurement_file" in config:
        # TODO: potential call before assignment error !!! with pedestal file

        config.update({"calibration": config_data,
                       "noise_analysis": noise_data})

        event_data = MainAnalysis(config["Measurement_file"],
                                  configs=config)  # Is adictionary containing all keys and values for configuration
        # Save the plots if specified
        if config.get("Output_folder", "") and config.get("Output_name", ""):
            save_all_plots(config["Output_name"], config["Output_folder"], dpi=300)
            if config.get("Pickle_output", False):
                save_dict(event_data.outputdata,
                          os.path.join(config["Output_folder"],
                                       config["Output_name"], ".dba"))
        return event_data.outputdata


if __name__ == "__main__":
    # Parse some options to the main analysis
    parser = OptionParser()
    parser.add_option("--config",
                      dest="configfile", action="store", type="string",
                      help="The path to the configfile which should be read",
                      default=""
                      )

    parser.add_option("--file",
                      dest="filepath", action="store", type="string",
                      help="Filepath of a measurement run",
                      default=""
                      )

    parser.add_option("--shell",
                      dest="shell", action="store_true", default=False,
                      help="Runs the shell interface for the anlysis",
                      )

    (options, args) = parser.parse_args()

    # Run the main routines
    main(args, options)
