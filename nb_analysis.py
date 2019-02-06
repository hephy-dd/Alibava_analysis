# This files contains analysis function optimizes by numba jit capabilities

import numpy as np
from numba import jit
from tqdm import tqdm
from multiprocessing import Manager
from utilities import *

def event_process_function(start, end, events, pedestal, meanCMN,
                           meanCMsig, noise, numchan, SN_cut, SN_ratio,
                           SN_cluster, max_clustersize, masking, material,
                           noisy_strips, queue=None):
    """Necessary function to pass to the pool.map function"""
    prodata = np.zeros((np.abs(start-end), 9), dtype=np.object)
    #automasked = 0
    index = 0
    hitmap = np.zeros(numchan)
    signal, SN, CMN, CMsig = nb_process_all_events(start, end, events, pedestal,
                                                   meanCMN, meanCMsig, noise,
                                                   numchan, noisy_strips)
    for i in tqdm(range(start, end), desc="Events processed"):
        #signal, SN, CMN, CMsig = nb_process_event(events[i], pedestal, meanCMN, meanCMsig, noise, numchan, noisy_strips)
        channels_hit, clusters, numclus, clustersize, automasked_hits = \
            nb_clustering(signal[i], SN[i], noise, SN_cut, SN_ratio, SN_cluster,
                          numchan, max_clustersize=max_clustersize,
                          masking=masking, material=material)
        for channel in channels_hit:
            hitmap[channel] += 1
        #automasked += automasked_hits

        prodata[index]=np.array([
            signal[i],
            SN[i],
            CMN,
            CMsig,
            hitmap, # Todo: remove hitmap from every event Is useless info and costs memory
            channels_hit,
            clusters,
            numclus,
            clustersize])
        index += 1
    return prodata

def parallel_event_processing(goodtiming, events, pedestal, meanCMN, meanCMsig,
                              noise, numchan, SN_cut, SN_ratio, SN_cluster,
                              max_clustersize = 5, masking=True, material=1,
                              poolsize = 1, Pool=None, noisy_strips = []):
    """Parallel processing of events."""
    goodevents = goodtiming[0].shape[0]
    automasked = 0

    if poolsize > 1:

        manager = Manager()
        q = manager.Queue()
        # Split data for the pools
        splits = int(goodevents/poolsize) # you may loose the last event!!!
        paramslist = []
        start = 0
        for i in range(poolsize):
            end = splits*(i+1)
            paramslist.append((start, end, events, pedestal, meanCMN, meanCMsig, noise, numchan, SN_cut, SN_ratio, SN_cluster, max_clustersize, masking, material, noisy_strips, q))
            start=end+1

        results = Pool.starmap(event_process_function, paramslist, chunksize=1)
        #prodata = np.zeros((goodevents, 9), dtype=np.object)
        #for i in tqdm(range(len(goodevents))):
        #    prodata[i] = q.get()



        # Build the correct Hitmap which gets lost during calculations
        hitmap = np.zeros(numchan)
        for hmap in results:
            hitmap += hmap[-1][4]
        prodata = np.concatenate(results, axis=0)
        # Set the last hit with the full hitmap # I know this is pretty shitty coding style.
        prodata[-1][4] = hitmap
        return prodata, automasked

    else:
        prodata = event_process_function(0, goodevents, events, pedestal,
                                         meanCMN, meanCMsig, noise, numchan,
                                         SN_cut, SN_ratio, SN_cluster,
                                         max_clustersize, masking, material,
                                         noisy_strips)
        return np.array(prodata), automasked

@jit(nopython = True, cache=True)
def nb_clustering(event, SN, noise, SN_cut, SN_ratio, SN_cluster, numchan,
                  max_clustersize = 5, masking=True, material=1):
    """Looks for cluster in a event"""
    channels = np.nonzero(np.abs(SN) > SN_cut)[0]  # Only channels which have a signal/Noise higher then the signal/Noise cut
    automasked_hit = 0
    used_channels = np.ones(numchan)  # To keep track which channel have been used already here ones due to valid channel calculations
    numclus = 0  # The number of found clusters
    clusters_list = []
    clustersize = []
    strips = len(event)
    absSN = np.abs(SN)
    SNval = SN_cut * SN_ratio
    offset = int(max_clustersize * 0.5)

    if masking:
        if material:
            # Todo: masking of dead channels etc.
            masked_ind = np.nonzero(np.take(event, channels) > 0)[0]  # So only negative values are considered
            valid_ind = np.nonzero(event < 0)[0]
            automasked_hit += len(masked_ind)
        else:
            masked_ind = np.nonzero(np.take(event, channels) < 0)[0]  # So only positive values are considered
            valid_ind = np.nonzero(event > 0)[0]
            automasked_hit += len(masked_ind)
    else:
        valid_ind = np.arange(strips)
        #masked_ind = np.zeros([-1])

    #masked_list = list(masked_ind)
    for i in valid_ind: # Define valid index to search for
        used_channels[i] = 0

    #Todo: misinterpretation of two very close clusters
    for ch in channels:  # Loop over all left channels which are a hit, here from "left" to "right"
        if not used_channels[ch]:# and ch not in masked_list:# and not masked_ind[ch]:  # Make sure we dont count everything twice
            used_channels[ch] = 1  # So now the channel is used
            cluster = [ch]  # Keep track of the individual clusters
            size = 1

            # Now make a loop to find neighbouring hits of cluster, we must go into both directions
            right_stop = 0
            left_stop = 0
            for i in range(1, offset+1):  # Search plus minus the channel found Todo: first entry useless
                if 0 < ch-i and ch+i < numchan:  # To exclude overrun
                    chp = ch+i
                    chm = ch-i
                    if not right_stop: # right side
                        if absSN[chp] > SNval:
                            if not used_channels[chp]:
                                cluster.append(chp)
                                used_channels[chp] = 1
                                size += 1
                            else:
                                right_stop = 1
                        else:
                            right_stop = 1 # Prohibits search for to long clusters or already used channels

                    if not left_stop: #left side
                        if absSN[chm] > SNval:
                            if not used_channels[chm]:
                                cluster.append(chm)
                                used_channels[chm] = 1
                                size += 1
                            else:
                                left_stop = 1
                        else:
                            left_stop = 1 # Prohibits search for to long clusters or already used channels
            # Look if the cluster SN is big enough to be counted as clusters
            Scluster = np.abs(np.sum(np.take(event, cluster)))
            Ncluster = np.sqrt(np.abs(np.sum(np.take(noise, cluster))))
            SNcluster = np.divide(Scluster,Ncluster)  # Actual signal to noise of cluster
            if SNcluster > SN_cluster:
                numclus = numclus+1
                clusters_list.append(cluster)
                clustersize.append(size)

    return channels, clusters_list, numclus, np.array(clustersize), automasked_hit

def nb_noise_calc(events, pedestal):
    """Noise calculation, normal noise (NN) and common mode noise (CMN)
    Uses numpy"""
    # Calculate the common mode noise for every channel
    cm = np.subtract(events,pedestal, dtype=np.float32)  # Get the signal from event and subtract pedestal
    CMsig = np.std(cm, axis=1)  # Calculate the standard deviation
    CMnoise = np.mean(cm, axis=1)  # Now calculate the mean from the cm to get the actual common mode noise
    # Calculate the noise of channels
    score = np.subtract(cm,CMnoise[:,None], dtype= np.float32)  # Subtract the common mode noise --> Signal[arraylike] - pedestal[arraylike] - Common mode
    # This is a trick with the dimensions of ndarrays, score = shape[ (x,y) - x,1 ] is possible otherwise a loop is the only way

    return np.array(score, dtype=np.float32), np.array(CMnoise, dtype=np.float32), np.array(CMsig, dtype=np.float32)  # Return everything


def nb_process_event(events, pedestal, meanCMN, meanCMsig, noise, numchan,
                     noisy_strips):
    """Processes single events"""
    #TODO: some elusive error happens here when using jit and njit
    # Calculate the common mode noise for every channel
    signal = events - pedestal  # Get the signal from event and subtract pedestal

    signal[noisy_strips] = 0

    # Remove channels which have a signal higher then 5*CMsig+CMN which are not representative
    removed = np.nonzero(signal < (5. * meanCMsig + meanCMN))
    prosignal = signal[removed]

    if prosignal.any():
        cmpro = np.mean(prosignal)
        sigpro = np.std(prosignal)

        corrsignal = signal - cmpro
        SN = corrsignal / noise

        return corrsignal, SN, cmpro, sigpro
    else:
        return np.zeros(numchan), np.zeros(numchan), 0., 0.  # A default value return if everything fails

def nb_process_all_events(start, stop, events, pedestal, meanCMN, meanCMsig,
                          noise, numchan, noisy_strips):
    """Processes events"""
    #TODO: some elusive error happens here when using jit and njit
    #Calculate the common mode noise for every channel
    signal = events[start:stop] - pedestal  # Get the signal from event and subtract pedestal

    signal[:,noisy_strips] = 0

    # Remove channels which have a signal higher then 5*CMsig+CMN which are not representative
    removed = np.nonzero(signal[:,] > (5. * meanCMsig + meanCMN))
    signal[removed[0], removed[1]] = 0 # Set the signals to 0
    prosignal = signal

    if prosignal.any():
        cmpro = np.mean(prosignal, axis=1)
        sigpro = np.std(prosignal, axis=1)

        corrsignal = signal - cmpro[:,None]
        SN = corrsignal / noise

        return corrsignal, SN, cmpro, sigpro
    else:
        return np.zeros(numchan), np.zeros(numchan), 0., 0.  # A default value return if everything fails
