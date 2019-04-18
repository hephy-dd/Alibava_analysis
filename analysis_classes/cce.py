"""This file contains the class for analysing the charge collection
efficiency"""
import logging
import matplotlib.pyplot as plt
from .utilities import manage_logger

class CCE:
    """This function has actually plots the the CCE plot"""

    def __init__(self, main_analysis, logger=None):
        """Initialize some important parameters"""
        self.log = logger or logging.getLogger(__class__.__name__)
        #manage_logger(self.log)
        self.main = main_analysis
        self.data = self.main.outputdata.copy()

    def run(self):
        pass

    def plot(self):
        """Plots the CCE"""

        ypos = [0]  # x and y positions for the plot
        xpos = [0]
        y0 = 0

        fig = plt.figure("Charge collection efficiency (CCE)")

        # Check if the langau has been calculated
        # Loop over all processed data files
        for path in self.main.pathes:
            file = str(
                path.split("\\")[-1].split('.')[0])  # Find the filename, warning these files must have been processed
            if self.data[file]:
                ypos.append(self.data[file]["Langau"]["langau_coeff"][0])  # First value is the mpv
                if not y0:
                    y0 = ypos[-1]
                ypos[-1] = ypos[-1] / y0
                xpos.append(xpos[-1] + 1)  # Todo: make a good x axis here from the file name (regex)
            else:
                import warnings
                warnings.warn(
                    "For the CCE plot to work correctly the langau analysis has to be done prior. Suppression of output")

        plot = fig.add_subplot(111)
        plot.set_title('Charge collection efficiency from file: {!s}'.format(file))
        plot.plot(xpos, ypos, "r--", color="b")
