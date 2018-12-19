# This file is the main analysis file here all processes will be started


from utilities import *
from calibration_analysis import *
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
        noise_data = noise_analysis(config["Pedestal_file"])
        noise_data.plot_data()

    # Look if a pedestal file is specified
    if "Measurement_file" in config:
        # TODO: potential call before assignement error !!! with pedestal file
        event_data = event_analysis(config["Measurement_file"],
                                    pedestal=noise_data.pedestal,
                                    CMN=noise_data.CMnoise,
                                    CMsig=noise_data.CMsig,
                                    Noise=noise_data.noise,
                                    SN_ratio=config.get("SN_ratio",0),
                                    MaxCluster=config.get("max_cluster_size",4),
                                    sensor_type=config.get("sensor_type","n-in-p"),
                                    masking=config.get("mask_dead_channels","False"),
                                    timing=config.get("timing",[0,100]))
        event_data.plot_data(single_event=50000)







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

#try:
main(args, options)
#except KeyError:
    #print("ERROR: I need an input file!")
#except IndexError:
    #print("ERROR: I need at least one parameter to work properly!")



