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
                 configs=None, logger=None):
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
        self.sig_std = []  # std signal per channel per pulse
        self.charge_sig = None  # Standard deviation of all charge calibartions
        self.delay_cal = []
        self.meansig_delay = []  # mean per pulse per channel
        self.isBinary = configs.get("isBinary", "False")
        self.use_gain_per_channel = configs.get("use_gain_per_channel", True)
        self.numChan = configs.get("numChan", 256)
        self.degpoly = configs.get("charge_cal_polynom", 5)
        self.range = configs.get("range_ADC_fit", [])
        self.ADC_sig = None
        self.configs = configs
        self.mean_sig_all_ch = []
        self.mean_std_all_ch = []
        if self.configs["calibrate_gain_to"] == "negative":
            self.polarity = -1
        elif self.configs["calibrate_gain_to"] == "positive":
            self.polarity = 1
        else:
            self.log.info("No polarity for gain pulse calibration selected, using both instead."
                          "Warning: This can cause serious miscalculations when converting ADC to electrons!")
            self.polarity = 0

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
        """Analyzes delay scan"""
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

        # Loading the file------------------------------------------------------
        # Charge scan
        self.log.info("Loading charge calibration file: %s", charge_path)
        if not self.isBinary:
            self.charge_data = import_h5(charge_path)
        else:
            self.charge_data = read_binary_Alibava(charge_path)

        # Look if data is valid------------------------------------------------------
        if not self.charge_data:
            raise ValueError("Unable to read the calibration file...")
        else:
            # Process data ----------------------------------------------------------
            # list of injected test pulse values aka x-data
            self.pulses = np.array(self.charge_data["scan"]["value"][:])

            # Sometime it happens, that h5py does not read correctly
            if not len(self.pulses):
                self.log.error("A HDF5 read error! Loaded empty array. "
                               "Restart python")

            # signals per pulse subtracted by pedestals and excluding noisy channels
            signals = np.array(self.charge_data["events"]["signal"][:]) - self.pedestal
            # signals = np.delete(signals, self.noisy_channels, axis=1)
            # Calculate size of pulse group to obtain number of injected signals per pulse step
            sigppulse = int(len(signals) / len(self.pulses))

            start = 0

            # summarize signals of each pulse group by calculating the mean
            # signals of each pulse group per channel
            for i in range(sigppulse, len(signals) + sigppulse, sigppulse):
                raw_half = np.mean(signals[start:i][0::2], axis=0)
                raw_half_std = np.std(signals[start:i][0::2], axis=0)
                raw_half2 = np.mean(signals[start:i][1::2], axis=0)
                raw_half2_std = np.std(signals[start:i][1::2], axis=0)

                # Take only the correct polarity of pulses
                if self.polarity == -1:
                    meansig_neg_pulses = np.hstack(list(zip(raw_half[0::2], raw_half2[1::2])))
                    std_neg_pulses = np.hstack(list(zip(raw_half_std[0::2], raw_half2_std[1::2])))
                    self.meansig_charge.append(np.abs(meansig_neg_pulses))
                    self.sig_std.append(std_neg_pulses)
                elif self.polarity == 1:
                    meansig_pos_pulses = np.hstack(list(zip(raw_half[1::2], raw_half2[0::2])))
                    std_pos_pulses = np.hstack(list(zip(raw_half_std[1::2], raw_half2_std[0::2])))
                    self.meansig_charge.append(np.abs(meansig_pos_pulses))
                    self.sig_std.append(std_pos_pulses)
                else:
                    self.meansig_charge.append(np.mean(np.abs(signals[start:i]), axis=0))
                    self.sig_std.append(np.std(np.abs(signals[start:i]), axis=0))
                start = i
            self.meansig_charge = np.array(self.meansig_charge)
            self.sig_std = np.array(self.sig_std)

            # For a pulse height of 0 one often finds non-zero values in meansig_charge
            # Use signals of 0 pulse as offset values and adjust rest accordingly
            offset = self.meansig_charge[1] # because first value is usually garbage
            if np.mean(offset) > 5:
                self.log.warning("Charge offset is greater then 5 ADC! This "
                                 "may be a result of bad calibration! Offset: {}".format(offset))
            #self.meansig_charge = self.meansig_charge - offset
            #for i, pul in enumerate(self.meansig_charge):
            #    for j, val in enumerate(pul):
            #        if val < 0:
            #            self.meansig_charge[i][j] = 0


            # Calculate the mean over all channels for every pulse and then calc
            # a poly fit for the mean gain curve
            self.mean_sig_all_ch = np.mean(self.meansig_charge, axis=1)
            self.mean_std_all_ch = np.mean(self.sig_std, axis=1)
            if self.mean_sig_all_ch[0] <= self.range[0] and self.mean_sig_all_ch[-1] >= self.range[0]:
                xminarg = np.argwhere(self.mean_sig_all_ch <= self.range[0])[-1][0]
                xmaxarg = np.argwhere(self.mean_sig_all_ch <= self.range[1])[-1][0]
            else:
                self.log.error("Range for charge cal poorly conditioned!!!")
                xmaxarg = len(self.mean_sig_all_ch) - 1
                xminarg = 0

            # Generate a list of tuples, over signal in ADC and corresponding electrons,
            # but only in the specified range (xmin and xmax), furthermore only take these signals
            # which show a signal greater as 0
            fit_params = [(sig, pul) for sig, pul in zip(self.mean_sig_all_ch[xminarg:xmaxarg],
                                                         self.pulses[xminarg:xmaxarg]) if sig > 0]

            # Generate poly fit a cut of the offset and append a 0 instead
            self.meancoeff = np.append(np.polyfit([tup[0] for tup in fit_params],
                                        [tup[1] for tup in fit_params],
                                        deg=self.degpoly, full=False)[:-1], [0])
            self.log.info("Mean fit coefficients over all channels are: %s", self.meancoeff)

            # Calculate the gain curve for EVERY channel-------------------------------------------
            self.channel_coeff = np.zeros([self.numChan, self.degpoly+1])
            channel_offset = 0
            for i in range(self.numChan):
                if i not in self.noisy_channels:
                    #self.log.debug("Fitting channel: {}".format(i))
                    try:
                        # Taking the correct channel from the means, this has the length of pulses, and the correct
                        # polarity is already accored to. Warning first value will always be cutted away,
                        # to ensure better convergence while fitting!!!
                        mean_sig = self.meansig_charge[1:,channel_offset]
                        sig_std = self.sig_std[1:, channel_offset]
                        # Find the range for the fit
                        if not mean_sig[0] <= self.range[0] and not mean_sig[-1] >= self.range[0]:
                            self.log.error("Range for charge cal for channel {} may be poorly conditioned!!!".format(i))
                        xminarg = np.argwhere(mean_sig <= self.range[0])[-1][0]
                        xmaxarg = np.argwhere(mean_sig <= self.range[1])[-1][0]

                        # In the beginning of the pulses the error can be huge. Therefore, check if std is small enough
                        # Otherwise search for point, which has a low enough std
                        std_ok = False
                        while not std_ok:
                            if xminarg == xmaxarg:
                                # Todo: make it possible to run nontheless
                                self.log.error("Could not find satisfying std value for charge cal. This may happen"
                                               " with bad calibration. Further calculations will fail!")
                            if mean_sig[xminarg]*0.4 <= sig_std[xminarg]:
                                xminarg += 1
                            else:
                                std_ok = True
                                break




                        #    xmaxarg = len(mean_sig) - 1
                        #    xminarg = 0

                        self.channel_coeff[i] = np.append(np.polyfit(mean_sig[xminarg:xmaxarg],
                                                            self.pulses[xminarg:xmaxarg],
                                                            deg=self.degpoly, full=False)[:-1], [0])
                    except Exception as err:
                        if "SVD did not converge" in str(err):
                            self.log.error("SVD did not converge in Linear Least Squares for channel {}"
                                           " this channel will be added to noisy channels!".format(i))
                            self.noisy_channels = np.append(self.noisy_channels, i)
                    channel_offset += 1

    def convert_ADC_to_e(self, signals_adc, channels=(), use_mean=False):
        """
        Convert an array of ADC signals to electron signal
        If the parameter, use_gain_per_channel is True in the configs then channels has to be set!!!
        In this case the gain will be calculated for each channel indipendently. Otherwise the mean
        gain will be used.

        :param signals_adc: The ADC signal which should be converted to electrons
        :param channels:  Optional parameter, it defines on which channel the corresponding ADC was aquired
        :return:
        """

        # use the mean coeff out of all channels -> Fast but can lead to errors if the calibration is not good
        if not self.use_gain_per_channel or use_mean:
            return np.absolute(np.polyval(self.meancoeff, signals_adc))

        # Use gain per channel for calculations. Warning: order of results list is ordered per channel!!!
        else:
            if len(signals_adc) != len(channels):
                self.log.error("If you want to use gain_per_channel calculations please pass " \
                                                     "lists of same size. Passed lists did not have same length.")
                return np.array([])

            result = np.array([])
            unique_ch = np.unique(channels).astype(np.int)
            signals_per_channel = []
            # Order the signals per channel so calculations can be speed up
            for ch in unique_ch:
                signals_per_channel.append(np.take(signals_adc, np.nonzero(channels==ch)))

            for sig, ch in zip(signals_per_channel, unique_ch):
                result = np.append(result, np.polyval(self.channel_coeff[ch], sig))

            return np.absolute(result)


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
