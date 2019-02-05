# This file is the main analysis file here all processes will be started

from analysis import *
import os
from optparse import OptionParser
import matplotlib.pyplot as plt
from cmd_shell import AlisysShell


def main(args, options):
    """The main analysis which will be executed after the arguments are parsed"""

    if options.shell:
        shell = AlisysShell()
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
