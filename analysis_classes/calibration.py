"""This file contains the class for the ALiBaVa calibration"""
#pylint: disable=C0103,C0301
import logging
import numpy as np
from scipy.interpolate import CubicSpline
from .utilities import read_binary_Alibava, import_h5, manage_logger

class Calibration:
    """This class handles all concerning the calibration"""
    def __init__(self, file_path="", Noise_calc={},
                 isBinary=False, logger=None, configs={}):
        """
        :param delay_path: Path to calibration file
        :param charge_path: Path to calibration file
        """
        self.log = logger or logging.getLogger(__class__.__name__)

        # self.charge_cal = None
        self.delay_cal = None
        self.delay_data = None
        self.charge_data = None
        self.pedestal = Noise_calc.pedestal
        self.noisy_channels = Noise_calc.noisy_strips
        # self.CMN = np.std(Noise_calc.CMnoise)
        self.chargecoeff = []
        self.gain = []
        self.meancoeff = None  # Mean coefficient out of all calibrations curves
        self.meansig_charge = []  # mean per pulse per channel
        self.charge_sig = None  # Standard deviation of all charge calibartions
        self.delay_cal = []
        self.meansig_delay = []  # mean per pulse per channel
        self.isBinary = isBinary
        self.ADC_sig = None
        self.configs = configs

        if not self.configs["use_charge_cal"]:
            self.delay_calibration_calc(file_path)
        elif file_path == "":
            self.use_predefined_cal_params()
        else:
            self.charge_calibration_calc(file_path)

    def use_predefined_cal_params(self):
        """Uses the predefined calibration parameters from the calibration file"""
        self.log.info("Using predefined gain parameters: %s", self.configs["Gain_params"])
        self.meancoeff = self.configs["Gain_params"]
        self.ADC_sig = 1.
        self.charge_sig = 1.
        self.chargecoeff = [np.array(self.configs["Gain_params"]) for i in range(256)]
        self.gain_calc()
        # So every strip has the same gain


    def delay_calibration_calc(self, delay_path):
        # Delay scan
        self.log.info("Loading delay file: %s", delay_path)
        if not self.isBinary:
            self.delay_data = import_h5(delay_path)
        else:
            self.delay_data = read_binary_Alibava(delay_path)

        pulses = np.array(self.delay_data["scan"]["value"][:])  # aka xdata

        # Sometime it happens, that h5py does not read correctly
        # TODO: write a more pythonic and pretty version of this
        if not len(pulses):
            self.log.error("A HDF5 read error! Loaded empty array. Restart python")

        signals = np.array(
            self.delay_data["events"]["signal"][:]) - self.pedestal  # signals per pulse, CMN is a single value
        signals = np.delete(signals, self.noisy_channels, axis=1)
        sigppulse = int(len(signals) / len(pulses))  # How many signals per pulses

        start = 0
        for i in range(sigppulse, len(signals) + sigppulse, sigppulse):
            self.meansig_delay.append(np.mean(signals[start:i], axis=1))
            start = i

        # Interpolate and get some extrapolation data from polynomial fit (from alibava)
        self.meansig_delay = np.mean(self.meansig_delay, axis=1)
        self.delay_cal = CubicSpline(pulses, self.meansig_delay)
        # print("Coefficients of charge fit: {!s}".format(self.chargecoeff))

    def charge_calibration_calc(self, charge_path):
        # Charge scan
        self.log.info("Loading charge calibration file: %s", charge_path)
        if not self.isBinary:
            self.charge_data = import_h5(charge_path)
        else:
            self.charge_data = read_binary_Alibava(charge_path)
        if self.charge_data:
            pulses = np.array(self.charge_data["scan"]["value"][:])  # aka xdata

            # Sometime it happens, that h5py does not read correctly
            #TODO: write a more pythonic and pretty version of this
            if not len(pulses):
                self.log.error("A HDF5 read error! Loaded empty array. Restart python")

            signals = np.array(self.charge_data["events"]["signal"][:]) - self.pedestal  # signals per pulse
            signals = np.delete(signals, self.noisy_channels, axis=1)

            # Sometimes it happens that alibava is not writing the value of the calibration
            # Usually happens when you do not use 32 pulses
            #if not pulses:
            #    pulses = np.arange(0, 101376, 1024) # TODO: This is not the way we do it!!! Ugly and hard coded!!!
            # Warning it seem that alibava calibrates in this order:
            # 1) Alternating pulses (pos/neg) on strips --> Strip 1-->pos, Strip2-->neg
            # 2) Next time other way round.
            # 3) Repeat until samplesize is is filled

            sigppulse = int(len(signals) / len(pulses))  # How many signals per pulses

            start = 0
            for i in range(sigppulse, len(signals) + sigppulse, sigppulse):
                # Calculate the absolute value of the difference of each strip to the pedestal and mean it
                raw = np.mean(np.abs(signals[start:i]), axis=0)
                self.meansig_charge.append(raw)
                start = i

            # Set the zero value to a real 0 size otherwise a non physical error happens (Offset corretion)
            offset = self.meansig_charge[0]
            self.meansig_charge = self.meansig_charge - offset
            if np.mean(offset) > 5:
                self.log.warning("Charge offset is greater then 5 ADC! This may be a result of bad calibration!")

            # Interpolate and get some extrapolation data from polynomial fit (from alibava)
            data = np.array(self.meansig_charge).transpose()
            # datamoffset = data-data[0]
            for pul in data:
                self.chargecoeff.append(np.polyfit(pul, pulses, deg=4, full=False))
            self.meancoeff = np.polyfit(np.mean(self.meansig_charge, axis=1), pulses, deg=4, full=False)
            self.log.info("Coefficients of charge fit: %s", self.meancoeff)
            self.ADC_sig = np.std(data, axis=0)
            self.charge_sig = np.polyval(self.meancoeff, self.ADC_sig)
            self.chargecoeff = np.array(self.chargecoeff)
            self.gain_calc()

    def charge_cal(self, x):
        return np.polyval(self.meancoeff, x)

    def gain_calc(self):
        for coeff in self.chargecoeff:
            self.gain.append(np.polyval(coeff, 100.))

    # def plot_data(self):
    #     """Plots the processed data"""
    #     if not self.configs["use_charge_cal"]:
    #         try:
    #             fig = plt.figure("Calibration")
    #
    #             # Plot delay
    #             if self.delay_data:
    #                 delay_plot = fig.add_subplot(222)
    #                 delay_plot.bar(self.delay_data["scan"]["value"][:], self.meansig_delay, 1., alpha=0.4, color="b")
    #                 #delay_plot.bar(self.delay_data["scan"]["value"][:], self.meansig_delay[:,60], 1., alpha=0.4, color="b")
    #                 delay_plot.set_xlabel('time [ns]')
    #                 delay_plot.set_ylabel('Signal [ADC]')
    #                 delay_plot.set_title('Delay plot')
    #
    #             # Plot charge
    #             if self.charge_data:
    #                 charge_plot = fig.add_subplot(221)
    #                 charge_plot.set_xlabel('Charge [e-]')
    #                 charge_plot.set_ylabel('Signal [ADC]')
    #                 charge_plot.set_title('Charge plot')
    #                 charge_plot.bar(self.charge_data["scan"]["value"][:], np.mean(self.meansig_charge, axis=1), 1000.,
    #                                 alpha=0.4, color="b", label="Mean of all gains")
    #                 cal_range = np.array(np.arange(1., 450., 10.))
    #                 charge_plot.plot(np.polyval(self.meancoeff, cal_range), cal_range, "r--", color="g")
    #                 charge_plot.errorbar(self.charge_data["scan"]["value"][:], np.mean(self.meansig_charge, axis=1),
    #                                      xerr=self.charge_sig, yerr=self.ADC_sig, fmt='o', markersize=1, color="red",
    #                                      label="Error")
    #                 charge_plot.legend()
    #
    #                 # Gain per Strip at ADC 100
    #                 gain_plot = fig.add_subplot(223)
    #                 gain_plot.set_xlabel('Channel [#]')
    #                 gain_plot.set_ylabel('Gain [e- at 100 ADC]')
    #                 gain_plot.set_title('Gain per Channel')
    #                 gain_plot.set_ylim(0, 70000)
    #                 gain = []
    #                 for coeff in self.chargecoeff:
    #                     gain.append(np.polyval(coeff, 100.))
    #
    #                 gain_plot.bar(np.arange(len(self.pedestal) - len(self.noisy_channels)), gain, alpha=0.4, color="b",
    #                               label="Only non masked channels")
    #                 gain_plot.legend()
    #
    #                 # Gain hist per Strip at ADC 100
    #                 gain_hist = fig.add_subplot(224)
    #                 gain_hist.set_ylabel('Count [#]')
    #                 gain_hist.set_xlabel('Gain [e- at 100 ADC]')
    #                 gain_hist.set_title('Gain Histogram')
    #                 gain_hist.hist(gain, alpha=0.4, bins=20, color="b", label="Only non masked channels")
    #                 gain_hist.legend()
    #
    #             fig.tight_layout()
    #             plt.draw()
    #             # plt.show() # COMMENT: Otherwise stop here
    #         except Exception as err:
    #             self.log.error("An error happened while trying to plot calibration data")
    #             self.log.error(err)
