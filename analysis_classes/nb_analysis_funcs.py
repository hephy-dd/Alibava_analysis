"""This files contains analysis function optimizes by numba jit capabilities"""
<<<<<<< HEAD
#pylint: disable=E1111,C0103
from numba import jit
from multiprocessing import Manager
=======
#pyline: disable=E1111
from numba import jit, prange
>>>>>>> Dominic_dev
import numpy as np

gil=True # Use gil or not
Fast=True # Use fastmath
parallel=True

jit(nogil=gil, cache=True, nopython=True, fastmath=Fast)
def event_process_function_multithread(args):
    """Just a small wrapper foe the multiprocessing function"""
    events, pedestal, meanCMN, meanCMsig, noise, numchan, SN_cut, SN_ratio, SN_cluster, max_clustersize, masking, material, noisy_strips = args
    return event_process_function(events, pedestal, meanCMN, meanCMsig, noise,
                           numchan, SN_cut, SN_ratio, SN_cluster, max_clustersize,
                           masking, material, noisy_strips)

jit(nogil=gil, parallel=parallel, nopython=True, fastmath=Fast)
def event_process_function(events, pedestal, meanCMN, meanCMsig, noise,
                           numchan, SN_cut, SN_ratio, SN_cluster, max_clustersize,
                           masking, material, noisy_strips, queue=None):
    """Necessary function to pass to the pool.map function"""
    prodata = np.zeros((len(events), 9), dtype=np.object)
    #automasked = 0
    index = 0
    hitmap = np.zeros(numchan)
    signal, SN, CMN, CMsig = nb_process_all_events(events, pedestal, meanCMN,
                                                   meanCMsig, noise, numchan, noisy_strips)
    for i in prange(0, len(events)):
        channels_hit, clusters, numclus, clustersize, automasked_hits = nb_clustering(signal[i], SN[i], noise, SN_cut,
                                                                                      SN_ratio, SN_cluster, numchan,
                                                                                      max_clustersize=max_clustersize,
                                                                                      masking=masking,
                                                                                      material=material)
        for channel in channels_hit:
            hitmap[channel] += 1

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
        index +=1
    return prodata

jit(nogil=gil, parallel=parallel, nopython=True, fastmath=Fast)
def parallel_event_processing(goodtiming, events, pedestal, meanCMN, meanCMsig, noise,
                              numchan, SN_cut, SN_ratio, SN_cluster, max_clustersize = 5,
                              masking=True, material=1, poolsize = 1, Pool=None, noisy_strips = []):
    """Parallel processing of events."""
    goodevents = goodtiming[0].shape[0]
    automasked = 0
    events_good = events[goodtiming[0]]

    if poolsize > 1:
        # Split data for the pools
        splits = int(goodevents/poolsize) # you may loose the last event!!!
        paramslist = []
        start = 0
        for i in range(poolsize):
            end = splits*(i+1)
            paramslist.append((events_good[start:end], pedestal, meanCMN, meanCMsig,
                               noise, numchan, SN_cut, SN_ratio, SN_cluster, max_clustersize,
                               masking, material, noisy_strips))
            start=end+1

        #results = Parallel(n_jobs=poolsize, verbose=1, backend='threading', require="sharedmem")(map(delayed(event_process_function_multithread),
        #                                                                                           paramslist))
        results = []
        for i in prange(poolsize):
            results.append(event_process_function_multithread(paramslist[i]))

        # Build the correct Hitmap which gets lost during calculations
        hitmap = np.zeros(numchan)
        for hmap in results:
            hitmap += hmap[-1][4]
        prodata = np.concatenate(results, axis=0)
        # Set the last hit with the full hitmap # I know this is pretty shitty coding style.
        prodata[-1][4] = hitmap
        return prodata, automasked

    else:
        prodata = event_process_function(events_good, pedestal, meanCMN,
                                         meanCMsig, noise, numchan, SN_cut, SN_ratio, SN_cluster,
                                         max_clustersize, masking, material, noisy_strips)
        #event_process_function.parallel_diagnostics(level=4)
        return np.array(prodata), automasked

@jit(nopython = True, cache=True, nogil=gil, fastmath=Fast)
def nb_clustering(event, SN, noise, SN_cut, SN_ratio, SN_cluster, numchan, max_clustersize = 5,
                  masking=True, material=1):
    """Looks for cluster in a event"""
    channels = np.nonzero(np.abs(SN) > SN_cut)[0]  # Only channels which have a signal/Noise higher
                                                   # then the signal/Noise cut
    automasked_hit = 0
    used_channels = np.ones(numchan)  # To keep track which channel have been used already here ones due
                                      #  to valid channel calculations
    numclus = 0  # The number of found clusters
    clusters_list = []
    clustersize = []
    strips = len(event)
    absSN = np.abs(SN)
    SNval = SN_cut * SN_ratio
    offset = int(max_clustersize * 0.5)

    if masking:
        if material:
            masked_ind = np.nonzero(np.take(event, channels) > 0)[0]  # So only negative values are considered
            valid_ind = np.nonzero(event < 0)[0]
            automasked_hit += len(masked_ind)
        else:
            masked_ind = np.nonzero(np.take(event, channels) < 0)[0]  # So only positive values are considered
            valid_ind = np.nonzero(event > 0)[0]
            automasked_hit += len(masked_ind)
    else:
        valid_ind = np.arange(strips)

    for i in valid_ind: # Define valid index to search for
        used_channels[i] = 0

    #Todo: misinterpretation of two very close clusters
    for ch in channels:  # Loop over all left channels which are a hit, here from "left" to "right"
            if not used_channels[ch]:# and ch not in masked_list:# and not masked_ind[ch]:
                                     #  Make sure we dont count everything twice
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

<<<<<<< HEAD
def nb_noise_calc(events, pedestal, tot_noise=False):
=======
jit(nogil=gil, cache=True, nopython=True)
def nb_noise_calc(events, pedestal):
>>>>>>> Dominic_dev
    """Noise calculation, normal noise (NN) and common mode noise (CMN)
    Uses numpy"""
    # Calculate the common mode noise for every channel
    cm = np.subtract(events, pedestal, dtype=np.float32)  # Get the signal from event and subtract pedestal
    CMsig = np.std(cm, axis=1)  # Calculate the standard deviation
    CMnoise = np.mean(cm, axis=1)  # Now calculate the mean from the cm to get the actual common mode noise
    # Calculate the noise of channels
    score = np.subtract(cm, CMnoise[:, None], dtype=np.float32)  # Subtract the common mode noise -->
                                                                # Signal[arraylike] - pedestal[arraylike] - Common mode
    # This is a trick with the dimensions of ndarrays, score = shape[ (x,y) - x,1 ]
    # is possible otherwise a loop is the only way
    noise = np.std(score, axis=0)
    if tot_noise is False:
        return noise, CMnoise, CMsig
    # convert score matrix into an 1-d array --> np.concatenate(score, axis=0))
    return noise, CMnoise, CMsig, np.concatenate(score, axis=0)

jit(nogil=gil, cache=True)
def nb_process_event(events, pedestal, meanCMN, meanCMsig, noise, numchan, noisy_strips):
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

jit(nogil=gil,cache=True, nopython=True, fastmath=Fast)
def nb_process_all_events(events, pedestal, meanCMN, meanCMsig, noise, numchan, noisy_strips):
    """Processes events"""
    #Calculate the common mode noise for every channel
    signal = events - pedestal  # Get the signal from event and subtract pedestal

    # Remove channels which have a signal higher then 5*CMsig+CMN which are not representative
    removed = np.nonzero(signal[:,] > (5. * meanCMsig + meanCMN))
    signal[removed[0], removed[1]] = 0 # Set the signals to 0
    prosignal = signal

    if prosignal.any():
        cmpro = np.mean(prosignal, axis=1)
        sigpro = np.std(prosignal, axis=1)

        corrsignal = signal - cmpro[:,None]
        corrsignal[:, noisy_strips] = 0
        SN = corrsignal / noise

        return corrsignal, SN, cmpro, sigpro
    else:
        return np.zeros(numchan), np.zeros(numchan), 0., 0.  # A default value return if everything fails
