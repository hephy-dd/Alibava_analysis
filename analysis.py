# This file contains functions for noise and pedestal analysis of ALIBAVA files

__version__ = 0.1
__date__ = "13.12.2018"
__author__ = "Dominic Bloech"
__email__ = "dominic.bloech@oeaw.ac.at"

# Import statements
from utilities import *
import warnings
from scipy.stats import norm
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import pylandau
from nb_analysis import *
from time import time
from multiprocessing import Pool
import gc


class main_analysis:
    """This class analyses measurement files per event and conducts additional defined analysis"""

    def __init__(self, path_list = None, **kwargs):
        """

        :param path_list: List of pathes to analyse
        :param kwargs: kwargs if further data should be used, possible kwargs=calibration,noise
        """

        # Init parameters
        print("Loading event file(s): {!s}".format(path_list))
        self.data = import_h5(path_list)

        self.numchan = len(self.data[0]["events/signal"][0])
        self.numevents = len(self.data[0]["events/signal"])
        self.pedestal = np.zeros(self.numchan, dtype=np.float32)
        self.noise = np.zeros(self.numchan, dtype=np.float32)
        self.SN_cut = 1
        self.hits = 0
        self.tmin = 0
        self.tmax = 100
        self.maxcluster = 4
        self.CMN = np.zeros(self.numchan, dtype=np.float32)
        self.CMsig = np.zeros(self.numchan, dtype=np.float32)
        self.outputdata = {}
        self.automasked_hit = 0
        self.numgoodevents = 0
        self.total_events = self.numevents*len(self.data)
        self.additional_analysis = []
        self.start = time()
        self.pathes = path_list
        self.kwargs = kwargs
        self.noise_analysis = kwargs["configs"].get("noise_analysis", None)
        self.calibration = kwargs["configs"].get("calibration", None)
        #self.kwargs = kwargs.get("configs", {}) # If a config was passeds it has to be a dict containig all settings therefore kwargs rewritten

        self.pedestal = self.noise_analysis.pedestal
        self.CMN = self.noise_analysis.CMnoise
        self.CMsig = self.noise_analysis.CMsig
        self.noise = self.noise_analysis.noise
        self.SN_cut = self.kwargs["configs"]["SN_cut"]  # Cut for the signal to noise ratio

        # For additional analysis
        self.add_analysis = kwargs["configs"].get("additional_analysis", [])
        if not self.add_analysis:
            self.add_analysis = []

        # Material decision
        self.material = kwargs["configs"].get("sensor_type", "n-in-p")
        if self.material == "n-in-p":
            self.material = 1
        else:
            self.material = 0 # Easier to handle

        self.masking = kwargs["configs"].get("automasking", False)
        self.max_clustersize = kwargs["configs"].get("max_cluster_size", 5)
        self.SN_ratio = kwargs["configs"].get("SN_ratio", 0.5)
        self.usejit = kwargs["configs"].get("optimize", False)
        self.SN_cluster = kwargs["configs"].get("SN_cluster", 6)

        # Create a pool for multiprocessing
        self.process_pool = kwargs["configs"].get("Processes", 1)  # How many workers
        self.Pool = Pool(processes=self.process_pool)

        if "timing" in kwargs["configs"]:
            self.min = kwargs["configs"]["timing"][0] # timinig window
            self.max = kwargs["configs"]["timing"][1] # timing maximum

        print("Processing files ...")
        # Here a loop over all files will be done to do the analysis on all imported files
        for data in tqdm(range(len(self.data)), desc="Data files processed:"):
                events = np.array(self.data[data]["events/signal"][:], dtype=np.float32)
                timing = np.array(self.data[data]["events/time"][:], dtype=np.float32)


                file = str(self.data[data]).split('"')[1].split('.')[0]
                self.outputdata[file] = {}
                # Todo: Make this loop work in a pool of processes/threads whichever is easier and better
                object = base_analysis(self, events, timing) # you get back a list with events, containing the event processed data --> np array makes it easier to slice
                results = object.run()

                self.outputdata[file]["base"] = Bdata(results, labels = ["Signal", "SN", "CMN", "CMsig", "Hitmap", "Channel_hit", "Clusters", "Numclus", "Clustersize"])

        object.plot_data(single_event=kwargs["configs"].get("Plot_single_event", 15)) # Not very pythonic, loop inside analysis (legacy)
        # Now process additional analysis statet in the config file
        for analysis in self.add_analysis:
            print("Starting analysis: {!s}".format(analysis))
            add_analysis = eval(analysis)(self) # Gets the total analysis class, so be aware of changes inside!!!
            results = add_analysis.run()
            add_analysis.plot()
            if results: # Only if results have been returned
                for file in results:
                    self.outputdata[file][str(analysis)] = results[file]

        # In the end give a round up of all you have done
        print("*************************************************************************\n" 
                  "            Analysis report:                                             \n"
                  "            ~~~~~~~~~~~~~~~~                                             \n"
                  "                                                                         \n"
                  "            Automasked hits:   {automasked!s}                            \n"
                  "            Events processed:  {events!s}                                \n"
                  "            Total events:      {total_events!s}                          \n"
                  "            Time taken:        {time!s}                                  \n"
                  "                                                                         \n"
                  "*************************************************************************\n".format(
                                                                                                    automasked=self.automasked_hit,
                                                                                                    events=self.numgoodevents,
                                                                                                    total_events = self.total_events,
                                                                                                    time = round((time()-self.start), 1))
                                                                                                    )
        # Add the noise results to the final dict
        self.outputdata["noise"] = {"pedestal": self.pedestal, "cmn": self.CMN, "cmnsig": self.CMsig, "noise": self.noise}

        self.Pool.close()
        self.Pool.join()

class base_analysis:

    def __init__(self, main, events, timing):
        self.main = main
        self.events = events
        self.timing = timing


    def run(self):
        """Does the actual event analysis"""

        # get events with good timinig only gtime and only process these events
        gtime = np.nonzero(self.timing>self.main.tmin)
        self.main.numgoodevents += int(gtime[0].shape[0])
        meanCMN = np.mean(self.main.CMN)
        meanCMsig = np.mean(self.main.CMsig)
        prodata = []  # List of processed data which then can be accessed
        hitmap = np.zeros(self.main.numchan)
        #Warning: If you have a RS and pulseshape recognition enabled the timing window has to be set accordingly

        if not self.main.usejit:
            # Non jitted version
            start = time()
            iter = 0
            for event in tqdm(range(gtime[0].shape[0]), desc="Events processed:"): # Loop over all good events
            # Event and Cluster Calculations
                iter +=1
                if iter == 1000:
                    gc.collect()
                    iter = 0
                signal, SN, CMN, CMsig = self.process_event(self.events[event], self.main.pedestal, meanCMN, meanCMsig,self.main.noise, self.main.numchan)
                channels_hit, clusters, numclus, clustersize = self.clustering(signal, SN, self.main.noise)
                for channel in channels_hit:
                    hitmap[int(channel)] += 1

                prodata.append([
                    signal,
                    SN,
                    CMN,
                    CMsig,
                    hitmap,
                    channels_hit,
                    clusters,
                    numclus,
                    clustersize]
                )

        else:
            start = time()
            # This should, in theory, use parallelization of the loop over event but i did not see any performance boost, maybe you can find the bug =)?
            data, automasked_hits = parallel_event_processing(gtime,
                                                              self.events,
                                                              self.main.pedestal,
                                                              meanCMN,
                                                              meanCMsig,
                                                              self.main.noise,
                                                              self.main.numchan,
                                                              self.main.SN_cut,
                                                              self.main.SN_ratio,
                                                              self.main.SN_cluster,
                                                              max_clustersize = self.main.max_clustersize,
                                                              masking=self.main.masking,
                                                              material=self.main.material,
                                                              poolsize=self.main.process_pool,
                                                              Pool=self.main.Pool,
                                                              noisy_strips = self.main.noise_analysis.noisy_strips)
            prodata = data
            self.main.automasked_hit = automasked_hits

        end = time()
        return prodata

    def clustering(self, event, SN, Noise):
        """Looks for cluster in a event"""
        channels = np.nonzero(np.abs(SN) > self.main.SN_cut)[0]# Only channels which have a signal/Noise higher then the signal/Noise cut
        valid_ind = np.arange(len(event))

        if self.main.masking:
            if self.main.material:
                # Todo: masking of dead channels etc.
                masked_ind = np.nonzero(np.take(event, channels) > 0)[0] # So only negative values are considered
                valid_ind = np.nonzero(event < 0)[0]  # Find out which index are negative so we dont count them accidently
                if len(masked_ind):
                    channels = np.delete(channels, masked_ind)
                    self.main.automasked_hit += len(masked_ind)
            else:
                masked_ind = np.nonzero(np.take(event, channels) < 0)[0] # So only positive values are considered
                valid_ind = np.nonzero(event > 0)[0]
                if len(masked_ind):
                    channels = np.delete(channels, masked_ind)
                    self.main.automasked_hit += len(masked_ind)

        used_channels = np.zeros(self.main.numchan) # To keep track which channel have been used already
        numclus = 0 # The number of found clusters
        clusters_list = []
        clustersize = np.array([])
        for ch in channels: # Loop over all left channels which are a hit, here from "left" to "right"
            if not used_channels[ch]: # Make sure we dont count everything twice
                used_channels[ch] = 1 # So now the channel is used
                cluster = [ch] # Keep track of the individual clusters
                size = 1

                right_stop = False
                left_stop = False
                # Now make a loop to find neighbouring hits of cluster, we must go into both directions
                offset = int(self.main.max_clustersize * 0.5)
                for i in range(1, offset+1):  # Search plus minus the channel found Todo: first entry useless
                    if 0 < ch-i and ch+i < self.main.numchan:  # To exclude overrun
                        if np.abs(SN[ch+i]) > self.main.SN_cut * self.main.SN_ratio and not used_channels[ch+i] and ch+i in valid_ind and not right_stop:
                            cluster.append(ch+i)
                            used_channels[ch+i] = 1
                            size += 1
                        elif np.abs(SN[ch+i]) < self.main.SN_cut * self.main.SN_ratio:
                            right_stop = True # Prohibits search for to long clusters

                        if np.abs(SN[ch-i]) > self.main.SN_cut * self.main.SN_ratio and not used_channels[ch-i] and ch-i in valid_ind and not left_stop:
                            cluster.append(ch-i)
                            used_channels[ch-i] = 1
                            size += 1
                        elif np.abs(SN[ch-i]) < self.main.SN_cut * self.main.SN_ratio:
                            left_stop = True # Prohibits search for to long clusters


                # Now make a loop to find neighbouring hits of cluster, we must go into both directions
                # TODO huge clusters can be misinterpreted!!! Takes huge amount of cpu, vectorize
                #offset = int(self.max_clustersize / 2)
                #for i in range(ch-offset, ch+offset): # Search plus minus the channel found
                #    if 0 < i < self.numchan: # To exclude overrun
                #            if np.abs(SN[i]) > self.SN_cut * self.SN_ratio and not used_channels[i] and i in valid_ind:
                #                cluster.append(i)
                #                used_channels[i] = 1
                #                # Append the channel which is also hit after this estimation
                #                size += 1

                # Look if the cluster SN is big enough to be counted as clusters
                Scluster = np.abs(np.sum(np.take(event, cluster)))
                Ncluster = np.sqrt(np.abs(np.sum(np.take(Noise, cluster))))
                SNcluster = Scluster/Ncluster #Actual signal to noise of cluster
                if SNcluster > self.main.SN_cluster:
                    numclus += 1
                    clusters_list.append(cluster)
                    clustersize = np.append(clustersize, size)

        # warning channels are only the channels which are above SN
        return channels, clusters_list, numclus, clustersize

    def process_event(self, event, pedestal, meanCMN, meanCMsig, noise, numchan=256):
        """Processes single events"""

        # Calculate the common mode noise for every channel
        signal = event - pedestal  # Get the signal from event and subtract pedestal

        # Mask noisy strips, by setting every noisy channel to 0 --> SN is always 0
        signal[self.main.noise_analysis.noisy_strips] = 0

        # Remove channels which have a signal higher then 5*CMsig+CMN which are not representative
        prosignal = np.take(signal, np.nonzero(signal<(5*meanCMsig+meanCMN))) # Processed signal

        if prosignal.any():
            cmpro = np.mean(prosignal)
            sigpro = np.std(prosignal)

            corrsignal = signal - cmpro
            SN = corrsignal / noise

            return corrsignal, SN, cmpro, sigpro
        else:
            return np.zeros(numchan), np.zeros(numchan), 0, 0 # A default value return if everything fails

    def plot_data(self, single_event = -1):
        """This function plots all data processed"""

        for name, data in self.main.outputdata.items():
            # Plot a single event from every file
            if single_event > 0:
                self.plot_single_event(single_event, name)

            # Plot Analysis results
            fig = plt.figure("Analysis file: {!s}".format(name))

            # Plot Hitmap
            channel_plot = fig.add_subplot(211)
            channel_plot.bar(np.arange(self.main.numchan), data["base"]["Hitmap"][len(data["base"]["Hitmap"])-1], 1., alpha=0.4, color="b")
            channel_plot.set_xlabel('channel [#]')
            channel_plot.set_ylabel('Hits [#]')
            channel_plot.set_title('Hitmap from file: {!s}'.format(name))

            fig.tight_layout()

            # Plot Clustering results
            fig = plt.figure("Clustering Analysis on file: {!s}".format(name))

            # Plot Number of clusters
            numclusters_plot = fig.add_subplot(221)
            bin, counts = np.unique(data["base"]["Numclus"], return_counts=True)
            numclusters_plot.bar(bin , counts, alpha=0.4, color="b")
            numclusters_plot.set_xlabel('Number of clusters [#]')
            numclusters_plot.set_ylabel('Occurance [#]')
            numclusters_plot.set_title('Number of clusters')

            # Plot clustersizes
            clusters_plot = fig.add_subplot(222)
            # Todo: make it possible to count clusters in multihit scenarios
            bin, counts = np.unique(np.concatenate(data["base"]["Clustersize"]), return_counts=True)
            clusters_plot.bar(bin, counts, alpha=0.4, color="b")
            clusters_plot.set_xlabel('Clustersize [#]')
            clusters_plot.set_ylabel('Occurance [#]')
            clusters_plot.set_title('Clustersizes')

            fig.suptitle('Cluster analysis from file {!s}'.format(name))
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            #plt.draw()

    def plot_single_event(self, eventnum, file):
        """ Plots a single event and its data"""

        data = self.main.outputdata[file]

        fig = plt.figure("Event number {!s}, from file: {!s}".format(eventnum, file))

        # Plot signal
        channel_plot = fig.add_subplot(211)
        channel_plot.bar(np.arange(self.main.numchan), data["base"]["Signal"][eventnum], 1., alpha=0.4, color="b")
        channel_plot.set_xlabel('channel [#]')
        channel_plot.set_ylabel('Signal [ADC]')
        channel_plot.set_title('Signal')

        # Plot signal/Noise
        SN_plot = fig.add_subplot(212)
        SN_plot.bar(np.arange(self.main.numchan), data["base"]["SN"][eventnum], 1., alpha=0.4, color="b")
        SN_plot.set_xlabel('channel [#]')
        SN_plot.set_ylabel('Signal/Noise [ADC]')
        SN_plot.set_title('Signal/Noise')

        fig.suptitle('Single event analysis from file {!s}, with event: {!s}'.format(file, eventnum))
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        #plt.draw()

class calibration:
    """This class handles all concerning the calibration"""

    def __init__(self, delay_path = "", charge_path = ""):
        """
        :param delay_path: Path to calibration file
        :param charge_path: Path to calibration file
        """

        #self.charge_cal = None
        self.delay_cal = None
        self.delay_data = None
        self.charge_data = None

        self.charge_calibration_calc(charge_path)
        self.delay_calibration_calc(delay_path)


    def delay_calibration_calc(self, delay_path):
        # Delay scan
        print("Loading delay file: {!s}".format(delay_path))
        self.delay_data = read_file(delay_path)
        if self.delay_data:
            self.delay_data = get_xy_data(self.delay_data, 2)

            if self.delay_data.any():
                # Interpolate data with cubic spline interpolation
                self.delay_cal = CubicSpline(self.delay_data[:,0],self.delay_data[:,1], extrapolate=True)

    def charge_calibration_calc(self, charge_path):
        # Charge scan
        print("Loading charge calibration file: {!s}".format(charge_path))
        self.charge_data = read_file(charge_path)
        if self.charge_data:
            self.charge_data = get_xy_data(self.charge_data, 2)

            if self.charge_data.any():
                # Interpolate and get some extrapolation data from polynomial fit (from alibava)
                #self.charge_cal = PchipInterpolator(self.charge_data[:,1],self.charge_data[:,0], extrapolate=False) # Test with another fit type
                self.chargecoeff = np.polyfit(self.charge_data[:,1],self.charge_data[:,0], deg=4, full=False)
                print("Coefficients of charge fit: {!s}".format(self.chargecoeff))
                #Todo: make it possible to define these parameters in the config file so everytime the same parameters are used


    def charge_cal(self, x):
        return np.polyval(self.chargecoeff, x)

    def plot_data(self):
        """Plots the processed data"""

        try:
            fig = plt.figure("Calibration")

            # Plot delay
            delay_plot = fig.add_subplot(212)
            delay_plot.bar(self.delay_data[:,0], self.delay_data[:,1], 5., alpha=0.4, color="b")
            delay_plot.plot(self.delay_data[:, 0], self.delay_cal(self.delay_data[:, 0]), "r--", color="g")
            delay_plot.set_xlabel('time [ns]')
            delay_plot.set_ylabel('Signal [ADC]')
            delay_plot.set_title('Delay plot')

            # Plot charge
            charge_plot = fig.add_subplot(211)
            charge_plot.bar(self.charge_data[:, 0], self.charge_data[:, 1], 2000., alpha=0.4, color="b")
            cal_range = np.array(np.arange(1., 700., 10.))
            charge_plot.plot(self.charge_cal(cal_range), cal_range, "r--", color="g")
            #charge_plot.plot(self.charge_cal(self.charge_data[:, 1]), self.charge_data[:, 1], "r--", color="g")
            charge_plot.set_xlabel('Charge [e-]')
            charge_plot.set_ylabel('Signal [ADC]')
            charge_plot.set_title('Charge plot')

            fig.tight_layout()
            #plt.draw()
        except Exception as e:
            print("An error happened while trying to plot calibration data ", e)

class noise_analysis:
    """This class contains all calculations and data concerning pedestals in ALIBAVA files"""

    def __init__(self, path = "", usejit=False, configs=None):
        """
        :param path: Path to pedestal file
        """

        # Init parameters
        print("Loading pedestal file: {!s}".format(path))
        self.data = import_h5(path)

        if self.data:
            # Some of the declaration may seem unecessary but it clears things up when you need to know how big some arrays are
            self.data=self.data[0]# Since I always get back a list
            self.numchan = len(self.data["header/pedestal"][0])
            self.numevents = len(self.data["events/signal"])
            self.pedestal = np.zeros(self.numchan, dtype=np.float32)
            self.noise = np.zeros(self.numchan, dtype=np.float32)
            self.goodevents = np.nonzero(self.data['/events/time'][:] >= 0)  # Only use events with good timing, here always the case
            self.CMnoise = np.zeros(len(self.goodevents[0]), dtype=np.float32)
            self.CMsig = np.zeros(len(self.goodevents[0]), dtype=np.float32)
            self.score = np.zeros((len(self.goodevents[0]), self.numchan), dtype=np.float32)  # Variable needed for noise calculations
            self.configs = configs

            # Calculate pedestal
            print("Calculating pedestal and Noise...")
            self.pedestal = np.mean(self.data['/events/signal'][0:], axis=0)
            self.signal = np.array(self.data['/events/signal'][:], dtype=np.float32)

            # Noise Calculations
            if not usejit:
                start = time()
                self.score_raw, self.CMnoise, self.CMsig = self.noise_calc(self.signal, self.pedestal[:], self.numevents, self.numchan)
                self.noise = np.std(self.score_raw, axis=0)
                self.noisy_strips, self.good_strips = self.detect_noisy_strips(self.noise, configs.get("Noise_cut", 5.))
                #self.noise_corr = np.std(self.score, axis=0)
                self.score_raw, self.CMnoise, self.CMsig = self.noise_calc(self.signal[:, self.good_strips],
                                                                           self.pedestal[self.good_strips], self.numevents,
                                                                           len(self.good_strips))
                end = time()
                print("Time taken: {!s} seconds".format(round(abs(end - start), 2)))
            else:
                print("Jit version used!!! No progress bar can be shown")
                start = time()
                self.score_raw, self.CMnoise, self.CMsig = nb_noise_calc(self.signal, self.pedestal)
                self.noise = np.std(self.score_raw,axis=0)  # Calculate the actual noise for every channel by building the mean of all noise from every event
                self.noisy_strips, self.good_strips = self.detect_noisy_strips(self.noise, configs.get("Noise_cut", 5.))
                self.score, self.CMnoise, self.CMsig = nb_noise_calc(self.signal[:,self.good_strips], self.pedestal[self.good_strips])
                self.noise_corr = np.std(self.score, axis=0)
                end = time()
                print("Time taken: {!s} seconds".format(round(abs(end-start), 2)))
            self.total_noise = np.concatenate(self.score, axis=0)

            
        else:
            print("No valid file, skipping pedestal run")

    def detect_noisy_strips(self, Noise, Noise_cut):
        """This function detects noisy strips and returns two arrays first array noisy strips, second array good strips"""

        good_strips = np.arange(len(Noise))
        # Calculate the
        self.median_noise = np.median(Noise)
        high_noise_strips = np.nonzero(Noise > self.median_noise+Noise_cut)
        good_strips = np.delete(good_strips, high_noise_strips)

        return high_noise_strips[0], good_strips



    def noise_calc(self, events, pedestal, numevents, numchannels):
        """Noise calculation, normal noise (NN) and common mode noise (CMN)
        Uses numpy, can be further optimized by reducing memory access to member variables.
        But got 36k events per second.
        So fuck it.
        This function is not numba optimized!!!"""
        score = np.zeros((numevents, numchannels), dtype=np.float32)  # Variable needed for noise calculations
        CMnoise = np.zeros(numevents, dtype=np.float32)
        CMsig = np.zeros(numevents, dtype=np.float32)

        for event in tqdm(range(self.goodevents[0].shape[0]), desc="Events processed:"): # Loop over all good events

            # Calculate the common mode noise for every channel
            cm = events[event][:] - pedestal  # Get the signal from event and subtract pedestal
            CMNsig = np.std(cm)  # Calculate the standard deviation
            CMN = np.mean(cm)  # Now calculate the mean from the cm to get the actual common mode noise

            # Calculate the noise of channels
            cn = cm - CMN # Subtract the common mode noise --> Signal[arraylike] - pedestal[arraylike] - Common mode

            score[event] = cn
            # Append the common mode values per event into the data arrays
            CMnoise[event] = CMN
            CMsig[event] = CMNsig

        return score, CMnoise, CMsig # Return everything

    def plot_data(self):
        """Plots the data calculated by the framework"""

        fig = plt.figure("Noise analysis")

        #Plot noisedata
        noise_plot = fig.add_subplot(221)
        noise_plot.bar(np.arange(self.numchan), self.noise, 1., alpha=0.4, color="b")
        # array of non masked strips
        valid_strips = np.ones(self.numchan)
        valid_strips[self.noisy_strips] = 0
        noise_plot.plot(np.arange(self.numchan), valid_strips, 1., color="r", label= "Masked strips")

        # Plot the threshold for deciding a good channel
        xval = [0,self.numchan]
        yval = [self.median_noise + self.configs.get("Noise_cut", 5.), self.median_noise + self.configs.get("Noise_cut", 5.)]
        noise_plot.plot(xval, yval, 1., "r--", color="g", label="Threshold for noisy strips")

        noise_plot.set_xlabel('Channel [#]')
        noise_plot.set_ylabel('Noise [ADC]')
        noise_plot.set_title('Noise levels per Channel')
        noise_plot.legend()

        # Plot pedestal
        pede_plot = fig.add_subplot(222)
        pede_plot.bar(np.arange(self.numchan), self.pedestal, 1., yerr=self.noise, error_kw=dict(elinewidth=0.2, ecolor='r', ealpha=0.1), alpha=0.4, color="b")
        pede_plot.set_xlabel('Channel [#]')
        pede_plot.set_ylabel('Pedestal [ADC]')
        pede_plot.set_title('Pedestal levels per Channel with noise')
        pede_plot.set_ylim(bottom=min(self.pedestal)-50.)
        #pede_plot.legend()

        # Plot Common mode
        CM_plot = fig.add_subplot(223)
        n, bins, patches = CM_plot.hist(self.CMnoise, bins=50, density=True, alpha=0.4, color="b")
        # Calculate the mean and std
        mu, std = norm.fit(self.CMnoise)
        # Calculate the distribution for plotting in a histogram
        p = norm.pdf(bins, loc=mu, scale=std)
        CM_plot.plot(bins, p, "r--", color="g")

        CM_plot.set_xlabel('Common mode [ADC]')
        CM_plot.set_ylabel('[%]')
        CM_plot.set_title(r'$\mathrm{Common\ mode\:}\ \mu=' + str(round(mu,2)) + r',\ \sigma=' + str(round(std,2)) + r'$')
        #CM_plot.legend()

        # Plot noise hist
        CM_plot = fig.add_subplot(224)
        n, bins, patches = CM_plot.hist(self.total_noise, bins=500, density=False, alpha=0.4, color="b")
        CM_plot.set_yscale("log", nonposy='clip')
        CM_plot.set_ylim(1.)

        # Cut off noise part
        cut = np.max(n) * 0.2  # Find maximum of hist and get the cut
        ind = np.concatenate(np.argwhere(n > cut))  # Finds the first element which is higher as threshold optimized

        # Calculate the mean and std
        mu, std = norm.fit(bins[ind])
        # Calculate the distribution for plotting in a histogram
        plotrange = np.arange(-35,35)
        p = gaussian(plotrange, mu, std, np.max(n))
        CM_plot.plot(plotrange, p, "r--", color="g")

        CM_plot.set_xlabel('Noise')
        CM_plot.set_ylabel('count')
        CM_plot.set_title("Noise Histogram")

        fig.tight_layout()
        #plt.draw()

class langau:
    """This class calculates the langau distribution and returns the best values for landau and Gauss fit to the data
    """

    def __init__(self, main_analysis):
        """Gets the main analysis class and imports all things needed for its calculations"""

        import pylandau # imports the necessary class for its calculations
        from scipy.optimize import curve_fit

        self.main = main_analysis
        self.data = self.main.outputdata.copy()
        self.results_dict = {} # Containing all data processed
        self.pedestal = self.main.pedestal
        self.pool = self.main.Pool
        self.poolsize = self.main.process_pool

    def run(self):
        """Calculates the langau for the specified data"""

        # Which clusters need to be considered
        clustersize_list = self.main.kwargs["configs"].get("langau",{}).get("clustersize",[-1])
        if type(clustersize_list) != list:
            clustersize_list = list(clustersize_list)

        if clustersize_list[0] == -1:
            clustersize_list = list(range(1, self.main.kwargs["configs"]["max_cluster_size"]+1)) # If nothing is specified


        # Go over all datafiles
        for data in tqdm(self.data, desc="(langau) Processing file:"):
            self.results_dict[data] = {}
            indizes = self.get_num_clusters(self.data[data], 1)[0] # Here events with only one cluster are choosen
            valid_events_clustersize = np.take(self.data[data]["base"]["Clustersize"], indizes) # Get the clustersizes of valid events
            valid_events_clusters = np.take(self.data[data]["base"]["Clusters"], indizes)
            valid_events_Signal = np.take(self.data[data]["base"]["Signal"], indizes)  # Get the clustersizes of valid events
            # Get events which show only cluster in its data
            charge_cal, noise = self.main.calibration.charge_cal, self.main.noise
            self.results_dict[data]["Clustersize"] = []

            # General langau, where all clustersizes are considered
            if self.main.usejit and self.poolsize > 1:
                paramslist = []
                for size in clustersize_list:
                    cls_ind = np.nonzero(valid_events_clustersize == size)[0]
                    paramslist.append((cls_ind, valid_events_Signal, valid_events_clusters, self.main.calibration.charge_cal, self.main.noise))

                results = self.pool.starmap(langau_cluster, paramslist, chunksize=1) # Here multiple cpu calculate the energy of the events per clustersize

                self.results_dict[data]["Clustersize"] = results

            else:
                for size in tqdm(clustersize_list, desc="(langau) Processing clustersize"):
                    # get the events with the different clustersizes
                    cls_ind = np.nonzero(valid_events_clustersize == size)[0]
                    # indizes_to_search = np.take(valid_events_clustersize, cls_ind) # TODO: veeeeery ugly implementation
                    #totalE = np.zeros(len(cls_ind))
                    #totalNoise = np.zeros(len(cls_ind))
                    # Loop over the clustersize to get total deposited energy
                    #incrementor = 0
                    signal_clst_event = []
                    noise_clst_event = []
                    for ind in tqdm(cls_ind, desc="(langau) Processing event"):
                        # TODO: make this work for multiple cluster in one event
                        # Signal calculations
                        signal_clst_event.append(np.take(valid_events_Signal[ind], valid_events_clusters[ind][0]))
                        # Noise Calculations
                        noise_clst_event.append(np.take(noise, valid_events_clusters[ind][0]))  # Get the Noise of an event

                    totalE = np.sum(convert_ADC_to_e(signal_clst_event, charge_cal), axis=1)
                    totalNoise = np.sqrt(np.sum(convert_ADC_to_e(noise_clst_event, charge_cal), axis=1))  # eError is a list containing electron signal noise

                        #incrementor += 1

                    preresults = {}
                    preresults["signal"] = totalE
                    preresults["noise"] = totalNoise

                    self.results_dict[data]["Clustersize"].append(preresults)

            # With all the data from every clustersize add all together and fit the langau to it
            finalE = np.zeros(0)
            finalNoise = np.zeros(0)
            for cluster in self.results_dict[data]["Clustersize"]:
                finalE = np.append(finalE, cluster["signal"])
                finalNoise = np.append(finalNoise, cluster["noise"])

            # Fit the langau to it
            coeff, pcov, hist, error_bins = self.fit_langau(finalE, finalNoise)
            self.results_dict[data]["signal"] = finalE
            self.results_dict[data]["noise"] = finalNoise
            self.results_dict[data]["langau_coeff"] = coeff
            self.results_dict[data]["langau_data"] = [np.arange(1., 100000., 1000.), pylandau.langau(np.arange(1., 100000., 1000.), *coeff)]  # aka x and y data
            self.results_dict[data]["data_error"] = error_bins

            # Consider now only the seedcut hits for the langau,
            if self.main.kwargs["configs"].get("langau",{}).get("seed_cut_langau",False):
                seed_cut_channels = self.data[data]["base"]["Channel_hit"]
                signals = self.data[data]["base"]["Signal"]
                finalE = []
                seedcutADC = []
                for i, signal in enumerate(tqdm(signals, desc="(langau SC) Processing events")):
                    if signal[seed_cut_channels[i]].any():
                        seedcutADC.append(signal[seed_cut_channels[i]])

                print("Converting ADC to electrons...")
                converted = convert_ADC_to_e(seedcutADC, charge_cal)
                for conv in converted:
                    finalE.append(sum(conv))
                finalE = np.array(finalE, dtype=np.float32)

                # get rid of 0 events
                indizes = np.nonzero(finalE > 0)[0]
                nogarbage = finalE[indizes]
                indizes = np.nonzero(nogarbage < 120000)[0] # ultra_high_energy_cut
                coeff, pcov, hist, error_bins = self.fit_langau(nogarbage[indizes], bins=500)
                self.results_dict[data]["signal_SC"] = nogarbage[indizes]
                self.results_dict[data]["langau_coeff_SC"] = coeff
                self.results_dict[data]["langau_data_SC"] = [np.arange(1., 100000., 1000.), pylandau.langau(np.arange(1., 100000., 1000.), *coeff)]  # aka x and y data

        return self.results_dict.copy()

    def fit_landau_migrad(x, y, p0, limit_mpv, limit_eta, limit_sigma, limit_A):
        #TODO make it possible with error calculation

        def minimizeMe(mpv, eta, sigma, A):
            chi2 = np.sum(np.square(y - langau(x, mpv, eta, sigma, A).astype(float)) / np.square(yerr.astype(float)))
            return chi2 / (x.shape[0] - 5)  # devide by NDF

        # Prefit to get correct errors
        yerr = np.sqrt(y)  # Assume error from measured data
        yerr[y < 1] = 1
        m = iminuit.Minuit(minimizeMe,
                           mpv=p0[0],
                           limit_mpv=limit_mpv,
                           error_mpv=1,
                           eta=p0[1],
                           error_eta=0.1,
                           limit_eta=limit_eta,
                           sigma=p0[2],
                           error_sigma=0.1,
                           limit_sigma=limit_sigma,
                           A=p0[3],
                           error_A=1,
                           limit_A=limit_A,
                           errordef=1,
                           print_level=2)
        m.migrad()

        if not m.get_fmin().is_valid:
            raise RuntimeError('Fit did not converge')

        # Main fit with model errors
        yerr = np.sqrt(langau(x,
                              mpv=m.values['mpv'],
                              eta=m.values['eta'],
                              sigma=m.values['sigma'],
                              A=m.values['A']))  # Assume error from measured data
        yerr[y < 1] = 1

        m = iminuit.Minuit(minimizeMe,
                           mpv=m.values['mpv'],
                           limit_mpv=limit_mpv,
                           error_mpv=1,
                           eta=m.values['eta'],
                           error_eta=0.1,
                           limit_eta=limit_eta,
                           sigma=m.values['sigma'],
                           error_sigma=0.1,
                           limit_sigma=limit_sigma,
                           A=m.values['A'],
                           error_A=1,
                           limit_A=limit_A,
                           errordef=1,
                           print_level=2)
        m.migrad()

        fit_values = m.values

        values = np.array([fit_values['mpv'],
                           fit_values['eta'],
                           fit_values['sigma'],
                           fit_values['A']])

        m.hesse()

        m.minos()
        minos_errors = m.get_merrors()

        if not minos_errors['mpv'].is_valid:
            print('Warning: MPV error determination with Minos failed! You can still use Hesse errors.')

        errors = np.array([(minos_errors['mpv'].lower, minos_errors['mpv'].upper),
                           (minos_errors['eta'].lower, minos_errors['eta'].upper),
                           (minos_errors['sigma'].lower, minos_errors['sigma'].upper),
                           (minos_errors['A'].lower, minos_errors['A'].upper)])

        return values, errors, m

    def fit_langau(self, x, errors=np.array([]), bins=500):
        """Fits the langau to data"""
        hist, edges = np.histogram(x, bins=bins)
        if errors.any():
            binerror = self.calc_hist_errors(x, errors, edges)
        else:
            binerror = np.array([])

        # Cut off noise part
        lancut = np.max(hist) * 0.33  # Find maximum of hist and get the cut
        #TODO: Bug when using optimized vs non optimized !!!
        try:
            ind_xmin = np.argwhere(hist > lancut)[0][0]  # Finds the first element which is higher as threshold optimized
        except:
            ind_xmin = np.argwhere(hist > lancut)[0]  # Finds the first element which is higher as threshold non optimized

        mpv, eta, sigma, A = 18000, 1500, 5000, np.max(hist)

        # Fit with constrains
        converged = False
        iter = 0
        oldmpv = 0
        diff = 100
        while not converged:
            iter += 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # create a text trap and redirect stdout
                # Warning: astype(float) is importanmt somehow, otherwise funny error happens one some machines where it tells you double_t and float are not possible
                coeff, pcov = curve_fit(pylandau.langau, edges[ind_xmin:-1].astype(float), hist[ind_xmin:].astype(float), absolute_sigma=True, p0=(mpv, eta, sigma, A), bounds=(1, 500000))
            if abs(coeff[0]-oldmpv) > diff:
                mpv, eta, sigma, A = coeff
                oldmpv = mpv
            else:
                converged = True
            if iter > 50:
                converged = True
                warnings.warn("Langau has not converged after 50 attempts!")


        return coeff, pcov, hist, binerror

    def get_num_clusters(self, data, num_cluster=1):
        """
        Get all clusters which seem important- Here custers with numclus will be returned
        :param data: data file which should be searched
        :param num_cluster: number of cluster which should be considered 1 is default and minimum. 0 makes no sense
        :return: list of data indizes after cluster consideration (so basically eventnumbers which are good)
        """
        return np.nonzero(data["base"]["Numclus"] == num_cluster) # Indizes of events with the desired clusternumbers

    def calc_hist_errors(self, x, errors, bins):
        """Calculates the errors for the bins in a histogram if error of simple point is known"""
        errorBins = np.zeros(len(bins)-1)
        binsize = bins[1]-bins[0]

        iter = 0
        for ind in bins:
            if ind != bins[-1]:
                ind_where_bin = np.where((x >= ind) & (x < (binsize+ind)))[0]
                #mu, std = norm.fit(self.CMnoise)
                if ind_where_bin.any():
                    errorBins[iter] = np.mean(np.take(errors, ind_where_bin))
                iter+=1

        return errorBins

    def plot(self):
        """Plots the data calculated so the energy data and the langau"""

        for file, data in self.results_dict.items():
            fig = plt.figure("Langau from file: {!s}".format(file))

            # Plot delay
            plot = fig.add_subplot(111)
            hist, edges = np.histogram(data["signal"], bins=500)
            plot.hist(data["signal"], bins=500, density=False, alpha=0.4, color="b", label= "All clusters")
            plot.errorbar(edges[:-1], hist, xerr=data["data_error"], fmt='o', markersize=1, color="red")
            plot.plot(data["langau_data"][0], data["langau_data"][1], "r--", color="g", label = "Langau: \n mpv: {mpv!s} \n eta: {eta!s} \n sigma: {sigma!s} \n A: {A!s} \n".format(mpv=data["langau_coeff"][0],eta=data["langau_coeff"][1],sigma=data["langau_coeff"][2],A=data["langau_coeff"][3]))
            plot.set_xlabel('electrons [#]')
            plot.set_ylabel('Count [#]')
            plot.set_title('All clusters Langau from file: {!s}'.format(file))
            #plot.legend(["Langau: \n mpv: {mpv!s} \n eta: {eta!s} \n sigma: {sigma!s} \n A: {A!s} \n".format(mpv=data["langau_coeff"][0],eta=data["langau_coeff"][1],sigma=data["langau_coeff"][2],A=data["langau_coeff"][3])])


            # Plot the different clustersizes as well into the langau plot
            colour = ['green', 'red', 'orange', 'cyan', 'black', 'pink', 'magenta']
            for i, cls in enumerate(data["Clustersize"]):
                if i<7:
                    plot.hist(cls["signal"], bins=500, density=False, alpha=0.3, color=colour[i], label="Clustersize: {!s}".format(i+1))
                else:
                    warnings.warn("To many histograms for this plot. Colorsheme only supports seven different histograms. Extend if need be!")
                    continue

            plot.set_xlim(0,100000)
            plot.legend()
            fig.tight_layout()
            #plt.draw()

            if self.main.kwargs["configs"].get("langau", {}).get("seed_cut_langau", False):
                fig = plt.figure("Seed cut langau from file: {!s}".format(file))

                # Plot Seed cut langau
                plot = fig.add_subplot(111)
                #indizes = np.nonzero(data["signal_SC"] > 0)[0]
                plot.hist(data["signal_SC"], bins=500, density=False, alpha=0.4, color="b", label="Seed clusters")
                plot.plot(data["langau_data_SC"][0], data["langau_data_SC"][1], "r--", color="g",
                          label="Langau: \n mpv: {mpv!s} \n eta: {eta!s} \n sigma: {sigma!s} \n A: {A!s} \n".format(
                              mpv=data["langau_coeff_SC"][0], eta=data["langau_coeff_SC"][1], sigma=data["langau_coeff_SC"][2],
                              A=data["langau_coeff_SC"][3]))
                plot.set_xlabel('electrons [#]')
                plot.set_ylabel('Count [#]')
                plot.set_title('Seed cut Langau from file: {!s}'.format(file))
                plot.set_xlim(0, 100000)
                plot.legend()

class chargesharing:
    """ A class calculating the charge sharing between two strip clusters and plotting it into a histogram and a eta plot"""

    def __init__(self, main_analysis):
        """Initialize some important parameters"""
        self.main = main_analysis
        self.clustersize = 2 # Other thing would not make sense for interstrip analysis
        self.data = self.main.outputdata.copy()
        self.results_dict = {}  # Containing all data processed

    def run(self):
        """Runs the analysis"""
        for data in tqdm(self.data, desc="(chargesharing) Processing file:"):
            self.results_dict[data] = {}
            # Get clustersizes of 2 and only events which show only one cluster in its data (just to be sure
            indizes_clusters = np.nonzero(self.data[data]["base"]["Numclus"] == 1) # Indizes of events with the desired clusternumbers
            clusters_raw = np.take(self.data[data]["base"]["Clustersize"], indizes_clusters)
            clusters_flattend = np.concatenate(clusters_raw).ravel() # so that they are easy accessible
            indizes_clustersize = np.nonzero(clusters_flattend == 2) # Indizes of events with the desired clusternumbers
            indizes = np.take(indizes_clusters, indizes_clustersize)[0]


            # Data containing the al and ar values as list entries data[0] --> al
            raw = np.take(self.data[data]["base"]["Signal"], indizes)
            raw = np.reshape(np.concatenate(raw), (len(raw),self.main.numchan))
            hits = np.concatenate(np.take(self.data[data]["base"]["Clusters"], indizes))
            al = np.zeros(len(indizes)) # Amplitude left and right
            ar = np.zeros(len(indizes))
            il = np.min(hits, axis=1)  # Indizes of left and right
            ir = np.max(hits, axis=1)
            #final_data = np.zeros((len(indizes), 2))

            for i, event, ali, ari, l, r in zip(range(len(al)), raw, al, ar, il, ir):
                al[i] = event[l] # So always the left strip is choosen
                ar[i] = event[r] # Same with the right strip

            # Convert ADC to actual energy
            al = convert_ADC_to_e(al, self.main.calibration.charge_cal)
            ar = convert_ADC_to_e(ar, self.main.calibration.charge_cal)

            final_data = np.array([al,ar])
            eta = ar/(al+ar)
            theta = np.arctan(ar/al)

            # Calculate the gauss distributions

            # Cut the eta in two halves and fit gaussian to it
            bins = 200
            etahist, edges = np.histogram(eta, bins=bins)
            length = len(etahist)
            mul, stdl = norm.fit(etahist[:int(length/2)])
            mur, stdr = norm.fit(etahist[int(length/2):])


            self.results_dict[data]["data"] = final_data
            self.results_dict[data]["eta"] = eta
            self.results_dict[data]["theta"] = theta
            self.results_dict[data]["fits"] = ((mul,stdl), (mur, stdr), edges, bins)

        return self.results_dict.copy()

    def plot(self):
        """Plots all results"""

        for file, data in self.results_dict.items():
            fig = plt.figure("Charge sharing from file: {!s}".format(file))

            # Plot delay
            plot = fig.add_subplot(221)
            counts, xedges, yedges, im = plot.hist2d(data["data"][0,:], data["data"][1,:], bins=400, range=[[0,50000],[0,50000]])
            plot.set_xlabel('A_left (electrons)')
            plot.set_ylabel('A_right (electrons)')
            fig.colorbar(im)
            plot.set_title('Charge distribution interstrip')

            plot = fig.add_subplot(222)
            counts, edges, im = plot.hist(data["eta"], bins=300, range=(0,1), alpha=0.4, color="b")
            #left = stats.norm.pdf(data["fits"][2][:100], loc=data["fits"][0][0], scale=data["fits"][0][1])
            #right = stats.norm.pdf(data["fits"][2], loc=data["fits"][1][0], scale=data["fits"][1][1])
            #plot.plot(data["fits"][2][:100], left,"r--", color="r")
            #plot.plot(data["fits"][2], right,"r--", color="r")
            plot.set_xlabel('eta')
            plot.set_ylabel('entries')
            plot.set_title('Eta distribution')

            plot = fig.add_subplot(223)
            counts, edges, im = plot.hist(data["theta"]/np.pi, bins=300, alpha=0.4, color="b", range=(0, 0.5))
            plot.set_xlabel('theta/Pi')
            plot.set_ylabel('entries')
            plot.set_title('Theta distribution')

            fig.suptitle('Charge sharing analysis from file {!s}'.format(file))
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            #plt.draw()

class CCE:
    """This function has actually plots the the CCE plot"""

    def __init__(self, main_analysis):
        """Initialize some important parameters"""
        self.main = main_analysis
        self.data = self.main.outputdata.copy()

    def run(self):
        pass

    def plot(self):
        """Plots the CCE"""

        ypos = [0] # x and y positions for the plot
        xpos = [0]
        y0 = 0

        fig = plt.figure("Charge collection efficiency (CCE)")

        # Check if the langau has been calculated
        # Loop over all processed data files
        for path in self.main.pathes:
            file = str(path.split("\\")[-1].split('.')[0])  # Find the filename, warning these files must have been processed
            if self.data[file]:
                ypos.append(self.data[file]["langau"]["langau_coeff"][0]) # First value is the mpv
                if not y0:
                    y0 = ypos[-1]
                ypos[-1] = ypos[-1]/y0
                xpos.append(xpos[-1]+1) # Todo: make a good x axis here from the file name (regex)
            else:
                import warnings
                warnings.warn("For the CCE plot to work correctly the langau analysis has to be done prior. Suppression of output")

        plot = fig.add_subplot(111)
        plot.set_title('Charge collection efficiency from file: {!s}'.format(file))
        plot.plot(xpos, ypos, "r--", color="b")


