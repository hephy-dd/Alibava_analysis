"""Noise analysis of ALiBaVa files"""
# pylint: disable=C0103,R0902,C0301,R0914,R0913
import logging
from time import time
import numpy as np
from tqdm import tqdm
from analysis_classes.nb_analysis_funcs import nb_noise_calc
from analysis_classes.utilities import import_h5, read_binary_Alibava

class NoiseAnalysis:
    """This class contains all calculations and data concerning pedestals in
	ALIBAVA files"""
    def __init__(self, path="", configs=None, logger=None):
        """
        :param path: Path to pedestal file
        """
        self.log = logger or logging.getLogger(__class__.__name__)

        self.log.info("Loading pedestal file: %s", path)
        if not configs["isBinary"]:
            self.data = import_h5(path)
        else:
            self.data = read_binary_Alibava(path)

        if self.data:
            # Some of the declaration may seem unecessary but it clears things
            # up when you need to know how big some arrays are

            # self.data["events"]["signal"][0] -> array of signals of 0th event of all channels
            self.numchan = len(self.data["events"]["signal"][0])
            self.numevents = len(self.data["events"]["signal"])
            self.pedestal = np.zeros(self.numchan, dtype=np.float32)
            self.noise = np.zeros(self.numchan, dtype=np.float32)
            self.noiseNCM = np.zeros(self.numchan, dtype=np.float32)

            # Only use events with good timing, here always the case
            self.noise_cut = configs.get("Noise_cut", 5.)
            self.which_strips = configs.get("Chips", (1, 2))
            self.max_channels = configs.get("numChan", 256)
            self.mask = configs.get("Manual_mask", [])
            self.goodevents = np.nonzero(self.data["events"]["time"][:] >= 0)
            self.CMnoise = np.zeros(self.numchan, dtype=np.float32)
            self.CMnoise_raw = np.zeros(self.numchan, dtype=np.float32)
            self.CMsig = np.zeros(self.numchan, dtype=np.float32)
            self.CMsig_raw = np.zeros(self.numchan, dtype=np.float32)
            self.score = np.zeros((len(self.goodevents[0]), self.numchan),
                                  dtype=np.float32)
            self.median_noise = None

            # Calculate pedestal
            self.log.info("Calculating pedestal and Noise...")
            # mean signal per channel over all events
            self.pedestal = np.mean(self.data["events"]["signal"][0:], axis=0)
            # complete signal matrix (event vs. channel)
            self.signal = np.array(self.data["events"]["signal"][:], dtype=np.float32)


            # First Noise calculation without masking to get an idea of the data
            self.noise_raw, self.noiseNCM_raw, self.CMnoise_raw, self.CMsig_raw = \
                    nb_noise_calc(self.signal, self.pedestal)
            self.noisy_strips, self.good_strips = \
                        self.detect_noisy_strips(self.noise, self.noise_cut)
            # Mask chips of alibava
            self.chip_selection, self.masked_channels = \
                    self.mask_alibava_chips(self.which_strips, self.max_channels)
            # Redefine good strips and noisy strips
            self.good_strips = np.intersect1d(self.chip_selection, self.good_strips)
            self.noisy_strips = np.append(self.noisy_strips,self.masked_channels)
            noise_corr, noiseNCM_corr, self.CMnoise, self.CMsig, self.total_noise = \
                        nb_noise_calc(self.signal[:, self.good_strips],
                                      self.pedestal[self.good_strips], True)

            # self.noise is only the non masked strips long. Make it to the full 256 strips long array so we can use it
            # Insert the correct noise for the masked strips and for all else insert np.nan --> This way it raises an error
            # if someone tries to access and claculate with it
            it = 0
            for i in range(self.numchan):
                if i in self.good_strips: # If its a good strip then add it and increase the it for tracking
                    self.noise[i] = noise_corr[it]
                    self.noiseNCM[i] = noiseNCM_corr[it]
                    it += 1
                else:
                    self.noise[i] = np.nan
                    self.noiseNCM[i] = np.nan



        else:
            self.log.warning("No valid file, skipping pedestal run")

    def mask_alibava_chips(self, chips_to_keep=(1,2), max_channels = 256):
        """Defines which chips should be considered"""
        final_channels = np.array([], dtype=np.int)
        for chip in chips_to_keep:
            start = (chip-1)*128
            to_keep = np.arange(start, start+128, dtype=np.int)
            final_channels = np.append(final_channels, to_keep)

        # No find the masked channels
        channels = np.arange(max_channels, dtype=np.int)
        return final_channels, np.setdiff1d(channels, final_channels)


    def detect_noisy_strips(self, Noise, Noise_cut):
        """Detect noisy strips (Noise > self.median_noise + Noise_cut) and
        includes the masking given by the user.
        Returns: noisy strips (np.array), good strips (np.array)"""

        good_strips = np.arange(len(Noise))
        # Calculate the median noise over all channels
        self.median_noise = np.median(Noise)
        high_noise_strips = np.nonzero(Noise > self.median_noise + Noise_cut)[0]
        high_noise_strips = np.append(high_noise_strips, self.mask)
        good_strips = np.delete(good_strips, high_noise_strips)

        return high_noise_strips.astype(np.int64), good_strips.astype(np.int64)

    def noise_calc(self, events, pedestal, numevents,
                   numchannels, tot_noise=False):
        """Noise calculation of normal noise (NN) and common mode noise (CMN).
        Uses numpy, can be further optimized by reducing memory access to member variables.
        But got 36k events per second.
        So fuck it.
        This function is not numba optimized!!!"""
        score = np.zeros((numevents, numchannels), dtype=np.float32)
        noise = score
        CMnoise = np.zeros(numevents, dtype=np.float32)
        CMsig = np.zeros(numevents, dtype=np.float32)

        # Loop over all good events
        for event in tqdm(range(self.goodevents[0].shape[0]), desc="Events processed:"):
            # Calculate the common mode noise for every channel
            cm = events[event][:] - pedestal  # Get the signal from event and subtract pedestal
            CMNsig = np.std(cm)  # Calculate the standard deviation
            CMN = np.mean(cm)  # Now calculate the mean from the cm to get the actual common mode noise

            # Calculate the noise of channels
            # Subtract the common mode noise --> Signal[arraylike] - pedestal[arraylike] - Common mode
            cn = cm - CMN
            score[event] = cn
            # Append the common mode values per event to the data arrays
            CMnoise[event] = CMN
            CMsig[event] = CMNsig

        # get noise per channel from score value per channel
        noise = np.std(score, axis=0)

        if tot_noise is False:
            return noise, CMnoise, CMsig
        # convert score matrix into an 1-d array --> np.concatenate(score, axis=0))
        return noise, CMnoise, CMsig, np.concatenate(score, axis=0)
