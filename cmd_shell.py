# This module makes the program interact with the command line

import cmd
import os
from threading import Thread
from analysis import *
from utilities import *
import matplotlib.pyplot as plt
import pprint
import numpy as np

np.set_printoptions(threshold=0, precision=2, edgeitems=2)


class AlisysShell(cmd.Cmd):
    """This class is for the commandline interface of the Alibava analysis.
    Every function stated with 'do_' will be executable in the shell.
    If you want to add additional functions, you can pass an callable object to
    self.add_cmd_command(obj) and you also can access it in the shell.
    Use list to show all available commands"""
    intro = '\n\nWelcome to the Alibava Analysis Shell.   \n' \
            'Type help or ? to list commands concerning general help.\n' \
            'This was programmed by Dominic Bloech and uses the cmd framework'
    prompt = '(Alisys)'
    file = None

    def __init__(self):
        """Initiates the cmd line interface etc"""
        super(AlisysShell, self).__init__()
        self.list_of_objects = []
        self.list_of_objects_str = []

        # results object
        self.results_obj = None

        try:
            self.cmdloop()
        except KeyboardInterrupt:
            print("^C")

    def add_cmd_command(self, object):
        """This function adds an object to the cmd prompt by calling the object with the args and kwargs"""
        self.list_of_objects.append(object)
        self.list_of_objects_str.append(object.__name__)
        setattr(self, "do_" + str(object.__name__), object)

    def do_list(self):
        """Just calls do_UniDAQ_functions"""
        self.do_functions()

    def do_functions(self):
        """This function writes back all functions added for use in the UniDAQ framework"""
        print("All functions provided by the Alisys framework:")
        for i in self.list_of_objects:
            print(str(i.__name__))
        print("==================================================")
        print("For more information to the methods type help <topic>")

    def start(self):
        "Starts the actual thread in which the shell is running"
        cmd.Cmd.__init__(self)
        self.t = Thread(target=self.start_shell)
        self.t.setDaemon(True)
        self.t.start()

    def start_shell(self):
        """This starts the shell"""
        try:
            self.cmdloop()
        except KeyboardInterrupt:
            print("^C")

    def do_bye(self):
        'Stops the Alisys shell'
        print('Thank you for using the Alisys analysis framework')
        return True

    # def precmd(self, line):
    #    """Just the pre command"""
    #    print("=====================================")
    #    return line

    # def postcmd(self, retval, line):
    #    """Just the post command"""
    #    if "list" not in line and line.split()[0] in self.list_of_objects_str:
    #        try:
    #            print("Executed command:".ljust(30) + str(line))
    #            print("Type of return value: ".ljust(30) + str(type(retval)))
    #            print(str(retval))
    #        except:
    #            pass
    #    print("=====================================")
    #    return False

    # Here all function have to be declared the Alisys shell can handle

    def do_run_config(self, config_file):
        """This function runs the analysis with the passed config file"""
        if os.path.exists(os.path.normpath(config_file)):
            configs = create_dictionary(os.path.normpath(config_file), "")
            self.results_obj = do_with_config_file(configs)
            plt.show()
        else:
            print("Please enter a valid filepath!")

    def do_plotEvent(self):
        """This function plots a Single event of all processed files"""
        plt.show()  # todo: write the cmd plot functions
        pass

    def do_hierachy(self, arg=None):
        """This function prints the hierachy of the prcessed data
        If you pass an arg the dictionary will be accessed with this arg"""

        if self.results_obj:
            pp = pprint.PrettyPrinter(indent=4, compact=True, width=20)
            if not arg:
                pp.pprint(self.results_obj)
            else:
                pp.pprint(self.results_obj[str(arg)])
        else:
            print("Please process some data first. Type ? for the how to")
