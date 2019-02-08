# This file is the main analysis file here all processes will be started


from optparse import OptionParser

from analysis_classes.Calibration import Calibration
from analysis_classes.NoiseAnalysis import NoiseAnalysis
from analysis_classes.main_loops import MainLoops
from analysis_classes.utilities import *
from cmd_shell import AlisysShell

log = logging.getLogger()
log.setLevel(logging.INFO)
if log.hasHandlers() is False:
    format_string = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    formatter = logging.Formatter(format_string)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)

def do_with_config_file(config):
    """Starts analysis with a config file"""

    # Look if a pedestal file is specified
    if "Pedestal_file" in config:
        noise_data = NoiseAnalysis(config["Pedestal_file"], usejit=config.get("optimize", False), configs=config)
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

        event_data = MainLoops(config["Measurement_file"],
                                   configs=config)  # Is adictionary containing all keys and values for configuration
        # Save the plots if specified
        if config.get("Output_folder", "") and config.get("Output_name", ""):
            save_all_plots(config["Output_name"], config["Output_folder"], dpi=300)
            if config.get("Pickle_output", False):
                save_dict(event_data.outputdata, config["Output_folder"] + "\\" + config["Output_name"] + ".dba")
        return event_data.outputdata

def main(args, options):
    """The main analysis which will be executed after the arguments are parsed"""

    if options.shell:
        AlisysShell()
        # shell.start_shell()


    elif options.configfile and os.path.exists(os.path.normpath(options.configfile)):
        configs = create_dictionary(os.path.normpath(options.configfile), "")
        do_with_config_file(configs)
        plt.show()  # Just in case the plot show has never been shown

    elif options.filepath and os.path.exists(os.path.normpath(options.filepath)):
        pass  # Todo: include the option to start the analysis with a passed file and config file

    else:
        print("No valid path parsed! Exiting")
        exit(1)


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
