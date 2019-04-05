"""This file contains the class for the ALiBaVa calibration"""
#pylint: disable=C0103,C0301,R0913,R0902
import logging
import numpy as np
from scipy.interpolate import CubicSpline
from .utilities import read_binary_Alibava, import_h5

class Calibration:
    """This class handles everything concerning the calibration.

    Charge Scan Details:
        - Each channel of the beetle chip is connected to a calibration
          capacitor which can inject a certain amount of charge (test pulse)
          into the channel. Each pulse generates a signal which is given in
          ADCs.
        - The charge scan is a sequence of injected test pulses with different
          pulse hights (charge values) for each channel.
        - The generated signal must be adjusted by subtracting the
          pedestal. Since alibava uses alternating pulses (pos/neg) in the
          course of the sequence (channel 1 -->pos, channel 2 -->neg, ...), one
          needs to calculate the absolute difference between pedestal and
          raw signal for each channel to obtain the 'real signal'.
        - The gain is the conversion factor between ADCs and electrons. It is
          defined as the gradient of the signal in ADCs vs. pulse height
          characteristic for each channel.
    """
    def __init__(self, file_path="", Noise_calc=None,
                 isBinary=False, configs=None, logger=None):
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
        self.coeff_per_ch = []
        self.mean_sig_all_ch = [] # mean signal per pulse over all channels
        self.gains_per_pulse = []
        self.pulses = [] # list of injected test pulses
        self.meancoeff = None  # Mean coefficient out of all calibrations curves
        self.meansig_charge = []  # mean signal per channel per pulse
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
        """Analyze the calibration scan and calculate conversion parameters
        for converting ADC Signals to e Signals"""
        # Charge scan
        self.log.info("Loading charge calibration file: %s", charge_path)
        if not self.isBinary:
            self.charge_data = import_h5(charge_path)
        else:
            self.charge_data = read_binary_Alibava(charge_path)
        if self.charge_data:
            # list of injected test pulse values aka x-data
            self.pulses = np.array(self.charge_data["scan"]["value"][:])

            # Sometime it happens, that h5py does not read correctly
            if not len(self.pulses):
                self.log.error("A HDF5 read error! Loaded empty array. "
                               "Restart python")

            # signals per pulse subtracted by pedestals and excluding noisy channels
            signals = np.array(self.charge_data["events"]["signal"][:]) - self.pedestal
            signals = np.delete(signals, self.noisy_channels, axis=1)
            # Calculate size of pulse group to obtain number of injected signals per pulse step
            sigppulse = int(len(signals) / len(self.pulses))

            start = 0
            # summerize signals of each pulse group by calculating the mean
            # signals of each pulse group per channel
            for i in range(sigppulse, len(signals) + sigppulse, sigppulse):
                raw = np.mean(np.abs(signals[start:i]), axis=0)
                self.meansig_charge.append(raw)
                start = i
            # For a pulse height of 0 one often finds non-zero values in meansig_charge
            # Use signals of 0 pulse as offset values and adjust rest accordingly
            offset = self.meansig_charge[0]
            self.meansig_charge = self.meansig_charge - offset
            for i, pul in enumerate(self.meansig_charge):
                for j, val in enumerate(pul):
                    if val < 0:
                        self.meansig_charge[i][j] = 0
            if np.mean(offset) > 5:
                self.log.warning("Charge offset is greater then 5 ADC! This "
                                 "may be a result of bad calibration!")

            # transpose matrix from signal per channel per pulse to signal per pulse per channel
            data = np.array(self.meansig_charge).transpose()
            self.mean_sig_all_ch = np.mean(self.meansig_charge, axis=1)
            for sig_per_pulse in data:
                # Since you want to convert ADC to e, you need to fit eSignal(ADCSignal)
                self.coeff_per_ch.append(np.polyfit(
                    sig_per_pulse, self.pulses, deg=5, full=False))
            # only consider signals that are larger than zero to improve fit in sensitive region
            fit_params = [(sig, pul) for sig, pul in \
                zip(self.mean_sig_all_ch, self.pulses) if sig > 0]
            self.meancoeff = np.polyfit([tup[0] for tup in fit_params],
                                        [tup[1] for tup in fit_params],
                                        deg=5, full=False)
            self.log.info("Mean fit coefficients over all channels: %s",
                          self.meancoeff)
            # self.coeff_test(data)

    def convert_ADC_to_e(self, x):
        """Convert an ADC signal or an array of ADC signals to electron
        signal"""
        if isinstance(x, (list, np.array)):
            return np.absolute(np.polyval(self.meancoeff, x))
        return abs(np.polyval(self.meancoeff, x))

    def gain_calc(self, cut=1.5):
        """Calculates the gain per channel per pulse. Ignores values for
        pulses below 'cut'. Beetle chip is not sensitive enough for low test
        pulse values"""
        # calculate gain of test pulses and prevent 'dividing by 0' warning
        for adc_all_ch, pulse in zip(self.meansig_charge, self.pulses):
            self.gains_per_pulse.append(
                [pulse/adc if adc > 0 else 0 for adc in adc_all_ch])
        gain_lst = []
        # concatenate gain lists
        for gain_ch in self.gains_per_pulse:
            gain_lst += gain_ch
        init_len = len(gain_lst)
        mean = np.mean(gain_lst)
        # exclude zeros and gains that are too large
        gain_lst = [gain for gain in gain_lst if 0 < gain < cut*mean]
        mean = np.mean(gain_lst)
        ex_ratio = round(1 - len(gain_lst)/init_len, 2)
        return gain_lst, mean, ex_ratio

    # def coeff_test(data):
    #     """Tests fit coefficients vs. signals generated by pulses. Compares mean
    #     vs. per channel coefficients.
    #     Args:
    #         - data (2D numpy array): signals per pulse per channel
    #         - pulses (numpy array): list of all test pulse values, e.g.
    #                                 pulses[10] = 10240,
    #                                 pulses[20] = 20480,
    #                                 pulses[30] = 30720, ...
    #     """
    #     pass
