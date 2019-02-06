"""This file contains the class for analysing the charge collection
efficiency"""

import logging
import matplotlib.pyplot as plt

class CCE:
    """This function has actually plots the the CCE plot"""

    def __init__(self, main_analysis):
        """Initialize some important parameters"""

        self.log = logging.getLogger(__class__.__name__)
        self.log.setLevel(logging.DEBUG)
        if self.log.hasHandlers() is False:
            format_string = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
            formatter = logging.Formatter(format_string)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.log.addHandler(console_handler)

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
            # Find the filename, warning these files must have been processed
            file = str(path.split("\\")[-1].split('.')[0])
            if self.data[file]:
                # First value is the mpv
                ypos.append(self.data[file]["langau"]["langau_coeff"][0])
                if not y0:
                    y0 = ypos[-1]
                ypos[-1] = ypos[-1] / y0
                # Todo: make a good x axis here from the file name (regex)
                xpos.append(xpos[-1] + 1)
            else:
                self.log.warning("For the CCE plot to work correctly the "
                                 "langau analysis has to be done prior. "
                                 "Suppression of output")

        plot = fig.add_subplot(111)
        plot.set_title(
            'Charge collection efficiency from file: {!s}'.format(file))
        plot.plot(xpos, ypos, "r--", color="b")
