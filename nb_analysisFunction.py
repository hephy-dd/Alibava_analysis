# This files contains analysis function optimizes by numba jit capabilities

import numpy as np
from numba import jit, prange
from tqdm import tqdm
from multiprocessing import Manager




def event_process_function(start, end, events, pedestal, meanCMN, meanCMsig, noise, numchan, SN_cut, SN_ratio, SN_cluster, max_clustersize, masking, material, queue=None):
    """Necessary function to pass to the pool.map function"""
    prodata = np.zeros((np.abs(start-end), 9), dtype=np.object)
    automasked = 0
    index = 0
    hitmap = np.zeros(numchan)
    #worker = current_process().name.split("-")[1]
    for i in tqdm(range(start, end), desc="Events processed"):
        signal, SN, CMN, CMsig = nb_process_event(events[i], pedestal, meanCMN, meanCMsig, noise, numchan)
        channels_hit, clusters, numclus, clustersize, automasked_hits = nb_clustering(signal, SN, noise, SN_cut,
                                                                                      SN_ratio, SN_cluster, numchan,
                                                                                      max_clustersize=max_clustersize,
                                                                                      masking=masking,
                                                                                      material=material)
        for channel in channels_hit:
            hitmap[channel] += 1
        automasked += automasked_hits

        #prodata.append([
        prodata[index]=np.array([
            signal,
            SN,
            CMN,
            CMsig,
            hitmap, # Todo: remove hitmap from every event Is useless info and costs memory
            channels_hit,
            clusters,
            numclus,
            clustersize])
        index +=1

    return prodata

def parallel_event_processing(goodtiming, events, pedestal, meanCMN, meanCMsig, noise, numchan, SN_cut, SN_ratio, SN_cluster, max_clustersize = 5, masking=True, material=1, poolsize = 1, Pool=None):
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
            paramslist.append((start, end, events, pedestal, meanCMN, meanCMsig, noise, numchan, SN_cut, SN_ratio, SN_cluster, max_clustersize, masking, material, q))
            start=end+1

        results = Pool.starmap(event_process_function, paramslist, chunksize=1)
        #print("here")
        #prodata = np.zeros((goodevents, 9), dtype=np.object)
        #for i in tqdm(range(len(goodevents))):
        #    prodata[i] = q.get()

        #pool.close()
        #pool.join()

        # Build the correct Hitmap which gets lost during calculations
        hitmap = np.zeros(numchan)
        for hmap in results:
            hitmap += hmap[-1][4]
        prodata = np.concatenate(results, axis=0)
        # Set the last hit with the full hitmap # I know this is pretty shitty coding style.
        prodata[-1][4] = hitmap
        return prodata, automasked

    else:
        prodata = event_process_function(0, goodevents, events, pedestal, meanCMN, meanCMsig, noise, numchan, SN_cut, SN_ratio, SN_cluster, max_clustersize, masking, material)
        return np.array(prodata), automasked

@jit(parallel = False, nopython = False)
def nb_clustering(event, SN, noise, SN_cut, SN_ratio, SN_cluster, numchan, max_clustersize = 5, masking=True, material=1):
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

@jit(parallel=True, nopython = True, cache=False)
def nb_noise_calc(events, pedestal, numevents, numchannels):
    """Noise calculation, normal noise (NN) and common mode noise (CMN)
    Uses numba and numpy, this function uses jit for optimization"""
    score = np.zeros((numevents, numchannels), dtype=np.float32)  # Variable needed for noise calculations
    CMnoise = np.zeros(numevents, dtype=np.float32)
    CMsig = np.zeros(numevents, dtype=np.float32)

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

@jit(parallel=False, nopython = False, cache=True)
def nb_process_event(event, pedestal, meanCMN, meanCMsig, noise, numchan=256):
    """Processes single events"""

    # Calculate the common mode noise for every channel
    signal = event - pedestal  # Get the signal from event and subtract pedestal

    # Remove channels which have a signal higher then 5*CMsig+CMN which are not representative
    removed = np.nonzero(signal < (5 * meanCMsig + meanCMN))
    prosignal = np.take(signal, removed)  # Processed signal

    if prosignal.any():
        cmpro = np.mean(prosignal)
        sigpro = np.std(prosignal)

        corrsignal = signal - cmpro
        SN = corrsignal / noise

        return corrsignal, SN, cmpro, sigpro
    else:
        return np.zeros(numchan), np.zeros(numchan), 0, 0  # A default value return if everything fails


