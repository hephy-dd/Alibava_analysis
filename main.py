# This file is the main analysis file here all processes will be started

from analysis import *
import os
from optparse import OptionParser
import matplotlib.pyplot as plt

def main(args, options):
    """The main analysis which will be executed after the arguments are parsed"""
    if options.configfile and os.path.exists(os.path.normpath(options.configfile)):
        configs = create_dictionary(os.path.normpath(options.configfile), "")
        do_with_config_file(configs)
        plt.show()


    elif options.filepath and os.path.exists(os.path.normpath(options.filepath)):
        pass

    else:
        print ("No valid path parsed! Exiting")
        exit(1)


def do_with_config_file(config):
    """Starts analysis with a config file"""

    # Look if a calibration file is specified
    if "Delay_scan" in config or "Charge_scan" in config:
        config_data = calibration(config.get("Delay_scan",""), config.get("Charge_scan",""))
        config_data.plot_data()

    # Look if a pedestal file is specified
    if "Pedestal_file" in config:
        noise_data = noise_analysis(config["Pedestal_file"], usejit=config.get("optimize", False), configs=config)
        noise_data.plot_data()

    # Look if a pedestal file is specified
    if "Measurement_file" in config:
        # TODO: potential call before assignment error !!! with pedestal file

        config.update({"calibration": config_data,
                       "noise_analysis": noise_data})

        event_data = main_analysis(config["Measurement_file"], configs = config) # Is adictionary containing all keys and values for configuration
        # Save the plots if specified
        if config.get("Output_folder", "") and config.get("Output_name", ""):
            save_all_plots(config["Output_name"], config["Output_folder"], dpi=300)
            if config.get("Pickle_output", False):
                save_dict(event_data.outputdata, config["Output_folder"] + "\\" + config["Output_name"] + ".dba")

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
(options, args) = parser.parse_args()

if __name__ == "__main__":
    main(args, options)
