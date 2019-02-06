"""This file contains the class for the ALiBaVa calibration"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from warnings import warn
from analysis_classes.utilities import * #import_h5, read_binary_Alibava


class calibration:
    """This class handles all concerning the calibration"""

    def __init__(self, delay_path="", charge_path="", Noise_calc={}, isBinary=False):
        """
        :param delay_path: Path to calibration file
        :param charge_path: Path to calibration file
        """

        # self.charge_cal = None
        self.delay_cal = None
        self.delay_data = None
        self.charge_data = None
        self.pedestal = Noise_calc.pedestal
        self.noisy_channels = Noise_calc.noisy_strips
        # self.CMN = np.std(Noise_calc.CMnoise)
        self.chargecoeff = []
        self.meancoeff = None  # Mean coefficient out of all calibrations curves
        self.meansig_charge = []  # mean per pulse per channel
        self.charge_sig = None  # Standard deviation of all charge calibartions
        self.delay_cal = []
        self.meansig_delay = []  # mean per pulse per channel
        self.isBinary = isBinary
        self.ADC_sig = None

        if charge_path:
            self.charge_calibration_calc(charge_path)
        if delay_path:
            self.delay_calibration_calc(delay_path)

    def delay_calibration_calc(self, delay_path):
        # Delay scan
        print("Loading delay file: {!s}".format(delay_path))
        if not self.isBinary:
            self.delay_data = import_h5(delay_path)[0]
        else:
            self.delay_data = read_binary_Alibava(delay_path)

        pulses = np.array(self.delay_data["scan"]["value"][:])  # aka xdata
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
        print("Loading charge calibration file: {!s}".format(charge_path))
        if not self.isBinary:
            self.charge_data = import_h5(charge_path)[0]
        else:
            self.charge_data = read_binary_Alibava(charge_path)
        if self.charge_data:
            pulses = np.array(self.charge_data["scan"]["value"][:])  # aka xdata
            signals = np.array(self.charge_data["events"]["signal"][:]) - self.pedestal  # signals per pulse
            signals = np.delete(signals, self.noisy_channels, axis=1)

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
                warn("Charge offset is greater then 5 ADC! This may be a result of bad calibration!")

            # Interpolate and get some extrapolation data from polynomial fit (from alibava)
            data = np.array(self.meansig_charge).transpose()
            # datamoffset = data-data[0]
            for pul in data:
                self.chargecoeff.append(np.polyfit(pul, pulses, deg=4, full=False))
            # print("Coefficients of charge fit: {!s}".format(self.chargecoeff))
            self.meancoeff = np.polyfit(np.mean(self.meansig_charge, axis=1), pulses, deg=4, full=False)
            self.ADC_sig = np.std(data, axis=0)
            self.charge_sig = np.polyval(self.meancoeff, self.ADC_sig)
            self.chargecoeff = np.array(self.chargecoeff)

    def charge_cal(self, x):
        return np.polyval(self.meancoeff, x)

    def plot_data(self):
        """Plots the processed data"""

        try:
            fig = plt.figure("Calibration")

            # Plot delay
            if self.delay_data:
                delay_plot = fig.add_subplot(222)
                delay_plot.bar(self.delay_data["scan"]["value"][:], self.meansig_delay, 1., alpha=0.4, color="b")
                #delay_plot.bar(self.delay_data["scan"]["value"][:], self.meansig_delay[:,60], 1., alpha=0.4, color="b")
                delay_plot.set_xlabel('time [ns]')
                delay_plot.set_ylabel('Signal [ADC]')
                delay_plot.set_title('Delay plot')

            # Plot charge
            if self.charge_data:
                charge_plot = fig.add_subplot(221)
                charge_plot.set_xlabel('Charge [e-]')
                charge_plot.set_ylabel('Signal [ADC]')
                charge_plot.set_title('Charge plot')
                charge_plot.bar(self.charge_data["scan"]["value"][:], np.mean(self.meansig_charge, axis=1), 1000.,
                                alpha=0.4, color="b", label="Mean of all gains")
                cal_range = np.array(np.arange(1., 450., 10.))
                charge_plot.plot(np.polyval(self.meancoeff, cal_range), cal_range, "r--", color="g")
                charge_plot.errorbar(self.charge_data["scan"]["value"][:], np.mean(self.meansig_charge, axis=1),
                                     xerr=self.charge_sig, yerr=self.ADC_sig, fmt='o', markersize=1, color="red",
                                     label="Error")
                charge_plot.legend()

                # Gain per Strip at ADC 100
                gain_plot = fig.add_subplot(223)
                gain_plot.set_xlabel('Channel [#]')
                gain_plot.set_ylabel('Gain [e- at 100 ADC]')
                gain_plot.set_title('Gain per Channel')
                gain_plot.set_ylim(0, 70000)
                gain = []
                for coeff in self.chargecoeff:
                    gain.append(np.polyval(coeff, 100.))

                gain_plot.bar(np.arange(len(self.pedestal) - len(self.noisy_channels)), gain, alpha=0.4, color="b",
                              label="Only non masked channels")
                gain_plot.legend()

                # Gain hist per Strip at ADC 100
                gain_hist = fig.add_subplot(224)
                gain_hist.set_ylabel('Count [#]')
                gain_hist.set_xlabel('Gain [e- at 100 ADC]')
                gain_hist.set_title('Gain Histogram')
                gain_hist.hist(gain, alpha=0.4, bins=20, color="b", label="Only non masked channels")
                gain_hist.legend()

            fig.tight_layout()
            # plt.draw()
        except Exception as e:
            print("An error happened while trying to plot calibration data ", e)

