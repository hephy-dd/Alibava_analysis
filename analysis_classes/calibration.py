"""This file contains the class for the ALiBaVa calibration"""

import logging
import warnings
from time import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import pylandau
from utilities import get_xy_data, read_file

class Calibration:
    """This class handles all concerning the calibration"""
    def __init__(self, delay_path="", charge_path=""):
        """
        :param delay_path: Path to calibration file
        :param charge_path: Path to calibration file
        """

        self.log = logging.getLogger(__class__.__name__)
        self.log.setLevel(logging.DEBUG)
        if self.log.hasHandlers() is False:
            format_string = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
            formatter = logging.Formatter(format_string)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.log.addHandler(console_handler)

        #self.charge_cal = None
        self.delay_cal = None
        self.delay_data = None
        self.charge_data = None

        self.charge_calibration_calc(charge_path)
        self.delay_calibration_calc(delay_path)

    def delay_calibration_calc(self, delay_path):
        # Delay scan
        self.log.info("Loading delay file(s) from: %s", delay_path)
        self.delay_data = read_file(delay_path)
        if self.delay_data:
            self.delay_data = get_xy_data(self.delay_data, 2)

            if self.delay_data.any():
                # Interpolate data with cubic spline interpolation
                self.delay_cal = CubicSpline(self.delay_data[:, 0],
                                             self.delay_data[:, 1],
                                             extrapolate=True)

    def charge_calibration_calc(self, charge_path):
        # Charge scan
        self.log.info("Loading charge calibration file(s) from: %s", charge_path)
        self.charge_data = read_file(charge_path)
        if self.charge_data:
            self.charge_data = get_xy_data(self.charge_data, 2)

            if self.charge_data.any():
                # Interpolate and get some extrapolation data from polynomial fit (from alibava)
                # self.charge_cal =
                # PchipInterpolator(self.charge_data[:,1],self.charge_data[:,0],
                # extrapolate=False) # Test with another fit type
                self.chargecoeff = np.polyfit(self.charge_data[:, 1],
                                              self.charge_data[:, 0],
                                              deg=4, full=False)
                self.log.info("Coefficients of charge fit: %s",
                              self.chargecoeff)
                # Todo: make it possible to define these parameters in the
                # config file so everytime the same parameters are used

    def charge_cal(self, x):
        return np.polyval(self.chargecoeff, x)

    def plot_data(self):
        """Plots the processed data"""

        try:
            fig = plt.figure("Calibration")

            # Plot delay
            delay_plot = fig.add_subplot(212)
            delay_plot.bar(self.delay_data[:, 0], self.delay_data[:, 1], 5.,
                           alpha=0.4, color="b")
            delay_plot.plot(self.delay_data[:, 0],
                            self.delay_cal(self.delay_data[:, 0]),
                            "r--", color="g")
            delay_plot.set_xlabel('time [ns]')
            delay_plot.set_ylabel('Signal [ADC]')
            delay_plot.set_title('Delay plot')

            # Plot charge
            charge_plot = fig.add_subplot(211)
            charge_plot.bar(self.charge_data[:, 0], self.charge_data[:, 1],
                            2000., alpha=0.4, color="b")
            cal_range = np.array(np.arange(1., 700., 10.))
            charge_plot.plot(self.charge_cal(cal_range), cal_range, "r--",
                             color="g")
            #charge_plot.plot(self.charge_cal(self.charge_data[:, 1]), self.charge_data[:, 1], "r--", color="g")
            charge_plot.set_xlabel('Charge [e-]')
            charge_plot.set_ylabel('Signal [ADC]')
            charge_plot.set_title('Charge plot')

            fig.tight_layout()
            # plt.draw()
        except Exception as err:
            self.log.error("An error happened while trying to plot "
                           "calibration data")
            self.log.error(err)
