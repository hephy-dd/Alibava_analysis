"""This files contains analysis function optimizes by numba jit capabilities"""
#pylint: disable=E1111,C0103
from numba import jit, prange
import numpy as np

# Some numba settings
gil = True # Use gil or not
Fast = True # Use fastmath
parallel = True # Use parallel execution

jit(nogil=gil, cache=True, nopython=True, fastmath=Fast)
def event_process_function_multithread(args):
    """Just a small wrapper foe the multiprocessing function

    Written by Dominic Bloech
    """
    events, pedestal, meanCMN, meanCMsig, noise, numchan, SN_cut, SN_ratio, SN_cluster, max_clustersize, masking, material, noisy_strips = args
    return event_process_function(events, pedestal, meanCMN, meanCMsig, noise,
                           numchan, SN_cut, SN_ratio, SN_cluster, max_clustersize,
                           masking, material, noisy_strips)

jit(nogil=gil, parallel=parallel, nopython=True, fastmath=Fast)
def event_process_function(events, pedestal, meanCMN, meanCMsig, noise,
                           numchan, SN_cut, SN_ratio, SN_cluster, max_clustersize,
                           masking, material, noisy_strips, event_timings):
    """
    This function simply handles the preprocessing of all events, like garbage clean up and then clustering
    :param events:
    :param pedestal:
    :param meanCMN:
    :param meanCMsig:
    :param noise:
    :param numchan:
    :param SN_cut:
    :param SN_ratio:
    :param SN_cluster:
    :param max_clustersize:
    :param masking:
    :param material:
    :param noisy_strips:
    :param event_timings:
    :return:

    Written by Dominic Bloech
    """

    # Generate the output array
    prodata = np.zeros((len(events), 10), dtype=np.object)
    index = 0
    # Generate the hitmap
    hitmap = np.zeros(numchan)

    # Preprocess all events for the clustering algorithm
    signal, SN, CMN, CMsig = nb_preprocess_all_events(events, pedestal, meanCMN,
                                                   meanCMsig, noise, numchan, noisy_strips)

    # Pass all events to the clustering algorithm
    for i in prange(0, len(events)):
        channels_hit, clusters, numclus, clustersize, automasked_hits = nb_clustering(signal[i], SN[i], noise, SN_cut,
                                                                                      SN_ratio, SN_cluster, numchan,
                                                                                      max_clustersize=max_clustersize,
                                                                                      masking=masking,
                                                                                      material=material)
        # Build the hitmap for this event
        for channel in channels_hit:
            hitmap[channel] += 1

        # Add the results to the results array for every event
        prodata[index]=np.array([
            signal[i],
            SN[i],
            CMN,
            CMsig,
            hitmap,
            channels_hit,
            clusters,
            numclus,
            clustersize,
            event_timings[i]])
        index +=1
    return prodata

jit(nogil=gil, parallel=parallel, nopython=True, fastmath=Fast)
def parallel_event_processing(goodtiming, timings, events, pedestal, meanCMN, meanCMsig, noise,
                              numchan, SN_cut, SN_ratio, SN_cluster, max_clustersize = 5,
                              masking=True, material=1, poolsize = 1, Pool=None, noisy_strips = []):
    """
    This function handles all logic to distribute the event processing and clustering to several cores
    to speed up the calculations. It does not do anything complicated.
    :param goodtiming: Array containing the indizes of evetns with good timing
    :param events: Array of all events: shape = (events, channels)
    :param pedestal: The pedestal: shape = (channels)
    :param meanCMN: A single value with the mean CMN of all events per channels
    :param meanCMsig: Same thing as meanCMN only the std
    :param noise: The noise of every channels: shape = (channels)
    :param numchan: Number of channels
    :param SN_cut: The SN_cut from the config
    :param SN_ratio: The SN_ratio from the config
    :param SN_cluster: The SN_cluster from the config
    :param max_clustersize: Maximum
    :param masking: A boolean of automatic masking should be applied
    :param material: Which base material the sensor is. This is needed for the signal polarity
    :param poolsize: Poolsize of the multiprocessing
    :param Pool: The actual muzltiprocessing pool
    :param noisy_strips: All noisy/masked strips from the user
    :return: The processed data

    Written by Dominic Bloech
    """

    # Get the number of how many good events there are
    goodevents = goodtiming[0].shape[0]
    automasked = 0
    # Slice out all good events
    events_good = events[goodtiming[0]].astype(np.float32)
    eventiming = timings[goodtiming[0]].astype(np.float32)

    # Do in the multiprocessed way if poolsize is greater as one
    if poolsize > 1:
        # Split data for the pools
        splits = int(goodevents/poolsize) # you may loose the last event!!!
        paramslist = []
        start = 0
        for i in range(poolsize):
            end = splits*(i+1)
            paramslist.append((events_good[start:end], pedestal, meanCMN, meanCMsig,
                               noise, numchan, SN_cut, SN_ratio, SN_cluster, max_clustersize,
                               masking, material, noisy_strips, eventiming[start:end]))
            start=end+1

        # Todo: Currently not working due to performance issues
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
        # If no multiprocessing is needed, simply call the event_process_function
        prodata = event_process_function(events_good, pedestal, meanCMN,
                                         meanCMsig, noise, numchan, SN_cut, SN_ratio, SN_cluster,
                                         max_clustersize, masking, material, noisy_strips, eventiming)
        return np.array(prodata), automasked

@jit(nopython = True, cache=True, nogil=gil, fastmath=Fast)
def nb_clustering(event, SN, noise, SN_cut, SN_ratio, SN_cluster, numchan, max_clustersize = 5,
                  masking=True, material=1):
    """
    Tries to find clusters in the event:
    It uses the three-cut algorithm: 1) Apply seed cut
                                     2) Search for neighbouring channels above the SN_ratio
                                     3) Check if cluster has higher SN as specified

    :param event: The event: shape = (channels)
    :param SN: The SN: shape = (channels)
    :param noise: The noise: shape = (channels)
    :param SN_cut: float
    :param SN_ratio: float
    :param SN_cluster: float
    :param numchan: Number of channels
    :param max_clustersize: Maximum cluster size
    :param masking: Bool, if you want masking of channels with false polarity
    :param material: The base material of the sensor, needed for polarity check
    :return:

    Written by Dominic Bloech
    """

    automasked_hit = 0
    # To keep track which channel have been used already here ones due to valid channel calculations
    used_channels = np.ones(numchan)
    numclus = 0  # The number of found clusters
    clusters_list = []
    clustersize = []
    strips = len(event)
    absSN = np.abs(SN)
    # SN for neighbours of seed cut
    SNval = SN_cut * SN_ratio
    offset = int(max_clustersize * 0.5)

    # Only channels which have a signal/Noise higher then the signal/Noise cut
    channels = np.nonzero(np.abs(SN) > SN_cut)[0]

    # Mask channels with the false polarity
    if masking:
        if material:
            # So only negative values are considered aka. p-type sensors
            masked_ind = np.nonzero(np.take(event, channels) > 0)[0]
            valid_ind = np.nonzero(event < 0)[0]
            automasked_hit += len(masked_ind)
        else:
            # So only positive values are considered aka. n-type sensors
            masked_ind = np.nonzero(np.take(event, channels) < 0)[0]
            valid_ind = np.nonzero(event > 0)[0]
            automasked_hit += len(masked_ind)
    else:
        # If none is selected then all will be used
        valid_ind = np.arange(strips)

    # Set all channels in which we search for hits to 0 to make them valid
    used_channels[valid_ind] = 0
    # Update the hitted channels #TODO: delete the ones which are automasked, delte not working with numba
    #channels = np.delete(channels, masked_ind) delete not supported by numba
    #Todo: misinterpretation of two very close clusters
    for ch in channels:  # Loop over all left channels which are a hit, here from "left" to "right"
            # Check if the channel has not been used so far
            if not used_channels[ch]:
                used_channels[ch] = 1  # Now the channel is used
                cluster = [ch]  # Size we have no a cluster init it with the channel
                size = 1 # The size of the cluster

                # Now make a loop to find neighbouring hits of cluster, we must go into both directions
                right_stop = 0
                left_stop = 0
                for i in range(1, offset+1):  # Search plus minus the channel found Todo: first entry useless
                    # Define bounderis of the chip, so we do not count outside
                    if 0 < ch-i and ch+i < numchan:
                        chp = ch+i # Right side of channel
                        chm = ch-i # Left side of channel

                        # Look if the right neighbour is above the SN_ratio from the SN_cut
                        # If absSN is nan (masked strip) loop continues
                        if not right_stop and not np.isnan(absSN[chp]):
                            if absSN[chp] > SNval:
                                if not used_channels[chp]:
                                    cluster.append(chp)
                                    used_channels[chp] = 1
                                    size += 1
                                else:
                                    right_stop = 1
                            else:
                                right_stop = 1 # Prohibits search for to long clusters or already used channels

                        # Look if the left neighbour is above the SN_ratio from the SN_cut
                        if not left_stop and not np.isnan(absSN[chm]):
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
                Scluster = np.abs(np.sum(np.take(event, cluster))) # Signal
                Ncluster = np.sqrt(np.abs(np.sum(np.take(noise, cluster)))) # Noise
                SNcluster = np.divide(Scluster,Ncluster)  # Actual signal to noise of cluster
                if SNcluster > SN_cluster:
                    numclus = numclus+1
                    clusters_list.append(cluster)
                    clustersize.append(size)

    return channels, clusters_list, numclus, np.array(clustersize), automasked_hit


jit(nogil=gil, cache=True, nopython=True)
def nb_noise_calc(events, pedestal, tot_noise=False):
    """
    Noise calculation, normal noise (NN) and common mode noise (CMN)
    Uses numpy black magic
    :param events: the events
    :param pedestal: the pedestal
    :param tot_noise: bool if you want the tot_noise
    :return:

    Written by Dominic Bloech
    """
    # Calculate the common mode noise for every channel
    # Get the signal from event and subtract pedestal
    cm = np.subtract(events, pedestal, dtype=np.float32)
    # Calculate the common mode
    keep = np.array(list(range(0,len(events))))
    for i in range(3): # Go over 3 times to get even the less dominant outliers
        CMsig = np.std(cm[keep], axis=1)
        # Now calculate the mean from the cm to get the actual common mode noise
        CMnoise = np.mean(cm[keep], axis=1)
        # Find common mode which is lower/higher than 1 sigma
        keep = np.where(np.logical_and(CMnoise<(CMnoise+2.5*CMsig), CMnoise>(CMnoise-2.5*CMsig)))[0]
    # Calculate the noise of channels
    # Subtract the common mode noise --> Signal[arraylike] - pedestal[arraylike] - Common mode
    score = np.subtract(cm[keep], CMnoise[keep][:,None], dtype=np.float32)
    # This is a trick with the dimensions of ndarrays, score = shape[ (x,y) - x,1 ]
    # is possible otherwise a loop is the only way
    noise = np.std(score, axis=0)
    noiseNC = np.std(cm[keep], axis=0)
    if tot_noise is False:
        return noise, noiseNC, CMnoise, CMsig
    # convert score matrix into an 1-d array --> np.concatenate(score, axis=0))
    return noise, noiseNC, CMnoise, CMsig, np.concatenate(score, axis=0)

#jit(nogil=gil, cache=True)
#def nb_process_event(events, pedestal, meanCMN, meanCMsig, noise, numchan, noisy_strips):
#    """Processes single events - This is an old version which has been replaces by"""
#    #TODO: some elusive error happens here when using jit and njit
#    # Calculate the common mode noise for every channel
#    signal = events - pedestal  # Get the signal from event and subtract pedestal
#
#    signal[noisy_strips] = 0
#
#    # Remove channels which have a signal higher then 5*CMsig+CMN which are not representative
#    removed = np.nonzero(signal < (5. * meanCMsig + meanCMN))
#    prosignal = signal[removed]
#
#    if prosignal.any():
#        cmpro = np.mean(prosignal)
#        sigpro = np.std(prosignal)
#
#        corrsignal = signal - cmpro
#        SN = corrsignal / noise
#
#        return corrsignal, SN, cmpro, sigpro
#    else:
#        return np.zeros(numchan), np.zeros(numchan), 0., 0.  # A default value return if everything fails

jit(nogil=gil,cache=True, nopython=True, fastmath=Fast)
def nb_preprocess_all_events(events, pedestal, meanCMN, meanCMsig, noise, numchan, noisy_strips):
    """
    Preprocesses all events and makes some clean-up on the signals.
    It calculates the SN for every events per channel and the CMN, CMNsig for every event.
    Furthermore it will return you the pure signal without pedestal, CMN etc.

    Warning: This function uses numpy black magic, if you are confused how a calculate things here,
             Please read a good introduction to numpy array operations =). But simply put I apply an
             operation usually to a whole array divisions are such a things. Every value in both
             arrays will be divided separately etc.

    :param events: All events shape = (events, channels)
    :param pedestal: The pedestal: shape = (channels)
    :param meanCMN: The mean CMN
    :param meanCMsig: The mean CMNsig
    :param noise: The noise per channel: shape = (channels)
    :param numchan: The number of channels
    :param noisy_strips: The noisy strips: shape = (channels)
    :return: corrsignal - signal without the garbage: shape = (events, channels)
             SN - Signal to noise: shape = (events)
             CMN - Common mode for every event: shape = (events)
             CMsig - Common mode std for every event: shape = (events)

    Written by Dominic Bloech
    """
    #Calculate the common mode noise for every channel
    signal = events - pedestal  # Get the signal from event and subtract pedestal

    # Remove channels which have a signal higher then 5*CMsig+CMN which are not representative
    removed = np.nonzero(signal[:,] > (5. * meanCMsig + meanCMN))
    signal[removed[0], removed[1]] = 0 # Set the signals to 0
    prosignal = signal

    if prosignal.any():
        # Calculate the mean CMN and CMNsig
        cmpro = np.mean(prosignal, axis=1)
        sigpro = np.std(prosignal, axis=1)

        # Subtract the CMN for all channels
        corrsignal = signal - cmpro[:,None]
        # Get rid of noisy strips by setting the signal to 0 which are not needed for further calculations
        corrsignal[:, noisy_strips] = 0
        # Calculate the actuall SN
        SN = corrsignal / noise

        return corrsignal, SN, cmpro, sigpro
    else:
        return np.zeros(numchan), np.zeros(numchan), 0., 0.  # A default value return if everything fails


@jit(nopython=False, nogil=True, cache=True)
def nb_process_cluster_size(args):
    """get the events with the different clustersizes its the numba optimized version

    Written by Dominic Bloech
    """
    size, valid_events_clustersize, valid_events_Signal, valid_events_clusters, noise, charge_cal, convert_ADC = args
    ClusInd = [[], []]
    for i, event in enumerate(valid_events_clustersize):
             for j, clus in enumerate(event):
                 if clus == size:
                     ClusInd[0].extend([i])
                     ClusInd[1].extend([j])

    signal_clst_event = []
    noise_clst_event = []
    for i, ind in enumerate(ClusInd[0]):
        y = ClusInd[1][i]
        # Signal calculations
        signal_clst_event.append(np.take(valid_events_Signal[ind], valid_events_clusters[ind][y]))
        # Noise Calculations
        noise_clst_event.append(
            np.take(noise, valid_events_clusters[ind][y]))  # Get the Noise of an event

    # totalE = np.sum(convert_ADC_to_e(signal_clst_event, charge_cal), axis=1)
    totalE = np.sum(convert_ADC(signal_clst_event, charge_cal), axis=1)

    # eError is a list containing electron signal noise
    # totalNoise = np.sqrt(np.sum(convert_ADC_to_e(noise_clst_event, charge_cal),
    #                             axis=1))
    totalNoise = np.sqrt(np.sum(convert_ADC(noise_clst_event, charge_cal), axis=1))

    preresults = {"signal": totalE, "noise": totalNoise}

    return preresults
