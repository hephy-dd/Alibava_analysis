# This files contains analysis function optimizes by numba jit capabilities

import numpy as np
from numba import jit, njit, prange, vectorize
from tqdm import tqdm
import scipy


@jit(parallel = False, nopython = True, cache=True, nogil=True)
def nb_clustering(event, SN, SN_cut, SN_ratio, numchan, max_clustersize = 5, masking=True, material=1):
    """Looks for cluster in a event"""
    channels = np.nonzero(np.abs(SN) > SN_cut)[0]  # Only channels which have a signal/Noise higher then the signal/Noise cut
    automasked_hit = 0
    used_channels = np.zeros(numchan)  # To keep track which channel have been used already
    numclus = 0  # The number of found clusters
    clusters_list = []
    clustersize = []

    if masking:
        if material:
            # Todo: masking of dead channels etc.
            masked_ind = np.nonzero(np.take(event, channels) > 0)[0]  # So only negative values are considered
            automasked_hit += len(masked_ind)
        else:
            masked_ind = np.nonzero(np.take(event, channels) < 0)[0]  # So only positive values are considered
            automasked_hit += len(masked_ind)
    else:
        masked_ind = np.array([-1])

    masked_list = list(masked_ind)


    for ch in channels:  # Loop over all left channels which are a hit, here from "left" to "right"
            if not used_channels[ch] and ch not in masked_list:  # Make sure we dont count everything twice
                used_channels[ch] = 1  # So now the channel is used
                numclus += 1
                cluster = [ch]  # Keep track of the individual clusters
                size = 1

                # Now make a loop to find neighbouring hits of cluster, we must go into both directions
                # TODO huge clusters can be misinterpreted!!! Takes huge amount of cpu, vectorize
                offset = int(max_clustersize * 0.5)
                for i in range(ch - offset, ch + offset):  # Search plus minus the channel found
                    if 0 < i < numchan:  # To exclude overrun
                        if np.abs(SN[i]) > SN_cut * SN_ratio and not used_channels[i]:
                            cluster.append(i)
                            used_channels[i] = 1
                            size += 1
                clusters_list.append(cluster)
                clustersize.append(size)  # TODO: This cost maybe to much calculation power for to less gain
    return channels, clusters_list, numclus, np.array(clustersize), automasked_hit

@jit(parallel=True, nopython = True, cache=True, nogil=True)
def nb_noise_calc(events, pedestal, numevents, numchannels):
    """Noise calculation, normal noise (NN) and common mode noise (CMN)
    Uses numba and numpy, this function uses jit for optimization"""
    score = np.zeros((numevents, numchannels), dtype=np.float64)  # Variable needed for noise calculations
    CMnoise = np.zeros(numevents, dtype=np.float64)
    CMsig = np.zeros(numevents, dtype=np.float64)

    for event in prange(numevents):  # Loop over all good events

        # Calculate the common mode noise for every channel
        cm = events[event][:] - pedestal  # Get the signal from event and subtract pedestal
        CMNsig = np.std(cm)  # Calculate the standard deviation
        CMN = np.mean(cm)  # Now calculate the mean from the cm to get the actual common mode noise

        # Calculate the noise of channels
        cn = cm - CMN  # Subtract the common mode noise --> Signal[arraylike] - pedestal[arraylike] - Common mode

        score[event] = cn
        # Append the common mode values per event into the data arrays
        CMnoise[event] = CMN
        CMsig[event] = CMNsig

    return score, CMnoise, CMsig  # Return everything

@jit(parallel=False, nopython = False, cache=True, nogil=True)
def nb_process_event(event, pedestal, meanCMN, meanCMsig, noise, numchan=256):
    """Processes single events"""

    # Calculate the common mode noise for every channel
    signal = event - pedestal  # Get the signal from event and subtract pedestal

    # Remove channels which have a signal higher then 5*CMsig+CMN which are not representative
    removed = np.nonzero(signal < (5 * meanCMsig + meanCMN))
    # TODO: np.ndarray.take wont work with nopython mode --> performance buff compared to no jit function
    prosignal = np.take(signal, removed)  # Processed signal

    if prosignal.any():
        cmpro = np.mean(prosignal)
        sigpro = np.std(prosignal)

        corrsignal = signal - cmpro
        SN = corrsignal / noise

        return corrsignal, SN, cmpro, sigpro
    else:
        return np.zeros(numchan), np.zeros(numchan), 0, 0  # A default value return if everything fails


