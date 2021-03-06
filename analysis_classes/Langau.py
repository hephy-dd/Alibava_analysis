"""This file contains the class for the Landau-Gauss calculation"""
# pylint: disable=C0103,E1101,R0913,C0301,E0401
import logging
import warnings
import time
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
import pylandau
from joblib import Parallel, delayed
from .utilities import set_attributes


class Langau:
    """Langau calculates the Landau-Gauss convolution for the energy deposition in sensors.

    How does it work:
        - First get the events with the desired number of clusters per event
        - Next add from every cluster the energy together and put it in a histogram

        In this analysis you have several options of getting more data out or constrain it
        You have the chance to get the the seed cut langau, where only seed cut hits are considered.
        For the overall langau you have the possibility to restrain which  clustersizes you want to consider
        See the possible parameters to pass!!!

     # Langau Analysis specific params
            - clustersize: list[int] - List of clustersizes the langau should be calculated of ([1,2,3])
            - seed_cut_langau: bool - Whether or not to calculate the seed cut langau (True)
            - fitLangau: bool - Try to fit a Langau to data or not (True)
            - energyCutOff: int - High energy cut of for calculations (100 000)
            - numClus: int - How many clusters per event should be considered (1)
            - bins: int - Bin count for langau (200)

    Written by Dominic Bloech
    """
    def __init__(self, main_analysis, configs, logger=None):
        """
        Init for the Langau analysis class

        :param main_analysis: The main analysis with all its parameters
        :param configs: The dictionary with the langau specific parameters
        :param logger: A specific logger if you want
        """

        # Makes the entries of the dict to member object of the class
        set_attributes(self, configs)

        self.log = logger or logging.getLogger(__class__.__name__)
        self.main = main_analysis
        self.data = self.main.outputdata.copy()
        self.results_dict = {}  # Containing all data processed
        self.pedestal = self.main.pedestal
        self.pool = self.main.Pool
        self.poolsize = self.main.process_pool
        self.numClusters = self.numClus
        self.Ecut = self.energyCutOff
        self.plotfit = self.fitLangau
        self.cluster_size_list = self.clustersize
        self.results_dict = {"bins": self.bins}
        self.seed_cut_langau = self.seed_cut_langau


    def run(self):
        """Runs the routines to generate all langau specific data"""

        # Here events with only one cluster are choosen or two, you decide
        indNumClus = self.get_num_clusters(self.data, self.numClusters)
        indizes = np.concatenate(indNumClus)

        # Slice the data from the bas analysis for further calculations
        valid_events_clustersize = np.take(self.data["base"]["Clustersize"], indizes)
        valid_events_clusters = np.take(self.data["base"]["Clusters"], indizes)
        valid_events_Signal = np.take(self.data["base"]["Signal"],indizes)
        self.results_dict["Clustersize"] = []

        # TODO: here a non numba optimized version is used. We should use numba here!
        # Calculate the energy deposition PER Clustersize and add it to self.results_dict["Clustersize"]
        self.cluster_analysis(valid_events_Signal,
                              valid_events_clusters,
                              valid_events_clustersize)

        # With all the data from every clustersize add all together and fit the main langau to it
        finalE = np.zeros(0)
        finalNoise = np.zeros(0)
        for cluster in self.results_dict["Clustersize"]:
            # Clean up and extra energy cut
            indi = np.nonzero(cluster["signal"] > 0)[0]
            nogarbage = cluster["signal"][indi]
            # ultra_high_energy_cut
            indi = np.nonzero(nogarbage < self.Ecut)[0]
            cluster["signal"] = cluster["signal"][indi]
            finalE = np.append(finalE, cluster["signal"])
            finalNoise = np.append(finalNoise, cluster["noise"])

        # Fit the langau to the summ of all individual clusters
        coeff, _, _, error_bins, edges= self.fit_langau(finalE,
                                                  finalNoise,
                                                  bins=self.results_dict["bins"],
                                                  cut = self.ClusterCut)

        self.results_dict["signal"] = finalE
        self.results_dict["noise"] = finalNoise
        self.results_dict["langau_coeff"] = coeff
        plotxrange = np.arange(0., edges[-1], edges[-1] / 1000.)
        self.results_dict["langau_data"] = [
            plotxrange,pylandau.langau(plotxrange,*coeff)
        ]  # aka x and y data
        self.results_dict["data_error"] = error_bins

        # Seed cut langau, taking only the bare hit channels which are above seed cut levels
        if self.seed_cut_langau:
            seed_cut_channels = self.data["base"]["Channel_hit"]
            signals = self.data["base"]["Signal"]
            seedcutADC = []
            seedcutChannels = []
            for i, signal in enumerate(tqdm(signals, desc="(langau SC) Processing events")):
                if signal[seed_cut_channels[i]].any():
                    seedcutADC.append(signal[seed_cut_channels[i]])
                    seedcutChannels.append(seed_cut_channels[i])


            if self.Charge_scale:
                self.log.info("Converting ADC to electrons for SC Langau...")
                converted = self.main.calibration.convert_ADC_to_e(np.concatenate(seedcutADC), np.concatenate(seedcutChannels))
            else:
                converted = np.absolute(np.concatenate(seedcutADC))
            finalE = np.array(converted, dtype=np.float32)

            # get rid of 0 events
            indizes = np.nonzero(finalE > 0)[0]
            nogarbage = finalE[indizes]
            indizes = np.nonzero(nogarbage < self.Ecut)[0]  # ultra_high_energy_cut
            coeff, _, _, error_bins, edges = self.fit_langau(
                nogarbage[indizes], bins=self.results_dict["bins"], cut = self.SCCut)
            self.results_dict["signal_SC"] = nogarbage[indizes]
            self.results_dict["langau_coeff_SC"] = coeff
            plotxrange = np.arange(0., edges[-1], edges[-1]/1000.)
            self.results_dict["langau_data_SC"] = [
                plotxrange,pylandau.langau(plotxrange,*coeff)
            ]  # aka x and y data

#         Old attempts for multiprocessing, no speed up seen here
#             # Try joblib
#             #start = time()
#             #arg_instances = [(size, valid_events_clustersize,
#             #                  valid_events_Signal, valid_events_clusters,
#             #                  noise, charge_cal) for size in clustersize_list]
#             #results = Parallel(n_jobs=4, backend="threading")(map(delayed(self.process_cluster_size),
#             #                                                           arg_instances))
#             #for res in results:
#             #    self.results_dict[data]["Clustersize"].append(res)
        #
        # !!!!!!!!!!!!!!! NO SPEED BOOST HERE!!!!!!!!!!!!!!!!!!!!
        # General langau, where all clustersizes are considered
        # if self.poolsize > 1:
        #    paramslist = []
        #    for size in self.cluster_size_list:
        #        cls_ind = np.nonzero(valid_events_clustersize == size)[0]
        #        paramslist.append((cls_ind, valid_events_Signal,
        #                           valid_events_clusters,
        #                           self.main.calibration.convert_ADC_to_e,
        #                           self.main.noise))

        # COMMENT: lagau_cluster not defined!!!!
        # Here multiple cpu calculate the energy of the events per clustersize
        #    results = self.pool.starmap(self.langau_cluster, paramslist,
        #                                chunksize=1)

        #    self.results_dict["Clustersize"] = results


        return self.results_dict.copy()

    def cluster_analysis(self, valid_events_Signal,
                         valid_events_clusters, valid_events_clustersize):
        """Calculates the energies for different cluster sizes
         (like a Langau per clustersize) - non optimized version """
        #TODO: Formerly in the numba function!!! We should use numba here not native python!!!
        for size in tqdm(self.cluster_size_list, desc="(langau) Processing clustersize"):
            # get the events with the different clustersizes
            ClusInd = [[], []]
            for i, event in enumerate(valid_events_clustersize):
                for j, clus in enumerate(event):
                    if clus == size:
                        ClusInd[0].extend([i])
                        ClusInd[1].extend([j])

            signal_clst_event = np.zeros([len(ClusInd[0]), size])
            noise_clst_event = np.zeros([len(ClusInd[0]), size])
            channels_hit_event = np.zeros([len(ClusInd[0]), size])
            #for i, ind in enumerate(tqdm(ClusInd[0], desc="(langau) Processing event")):
            for i, ind in enumerate(ClusInd[0]):
                y = ClusInd[1][i]
                # Signal calculations
                signal_clst_event[i] = np.take(valid_events_Signal[ind], valid_events_clusters[ind][y])
                # Noise Calculations
                noise_clst_event[i] = np.take(self.main.noise, valid_events_clusters[ind][y])  # Get the Noise of an event
                # Save channels at which the hit happend
                channels_hit_event[i] = np.array(valid_events_clusters[ind][y])

            # Todo: Due to the sum of all channels prior to conversion a need to choose a channel for
            # the gain. Therefore, in the future it would be good to separately calculate the gain for
            # every channel and then build the sum. But the error should be minimal.

            try:
                totalE = np.zeros([len(channels_hit_event[0]), len(channels_hit_event)])
                for part in range(len(channels_hit_event[0])):
                    if self.Charge_scale:
                        totalE[part] = self.main.calibration.convert_ADC_to_e(signal_clst_event[:,part], channels_hit_event[:,part])
                    else:
                        totalE[part] = np.absolute(signal_clst_event[:,part])
                totalE = np.sum(totalE, axis=0)


                # eError is a list containing electron signal noise
                if self.Charge_scale:
                    totalNoise = np.sqrt(
                        self.main.calibration.convert_ADC_to_e(np.sum(noise_clst_event,axis=1), channels_hit_event[:,0]))
                else:
                    totalNoise = np.sqrt(np.sum(noise_clst_event,axis=1))

                preresults = {"signal": totalE, "noise": totalNoise}
            except:
                self.log.critical("Clustersize analysis of size: {} seems to have no entries skipping this clustersize. "
                                  "Warning this is VERY uncommon please make sure the other data is correct!!!".format(size))
                preresults = {"signal": np.zeros(1), "noise": np.zeros(0)}

            self.results_dict["Clustersize"].append(preresults)

    def fit_langau(self, x, errors=np.array([]), bins=500, cut=0.33):
        """Fits the langau to data"""
        hist, edges = np.histogram(x, bins=bins)
        # If no histogram could be made
        if not hist.any():
            self.log.critical("Insufficient data to make a histogram, langau fit aborted! ")
            return [1, 1, 1, 1], None, [0], [0], [0,1]
        if errors.any():
            binerror = self.calc_hist_errors(x, errors, edges)
        else:
            binerror = np.array([])

        # Cut off noise part
        lancut = np.max(hist) * cut  # Find maximum of hist and get the cut
        # TODO: Bug when using optimized vs non optimized !!!
        try:
            ind_xmin = np.argwhere(hist > lancut)[0][0]
            # Finds the first element which is higher as threshold optimized
        except:
            ind_xmin = np.argwhere(hist > lancut)
            if ind_xmin.any():
                ind_xmin[0]
            else:
                ind_xmin = 0
            # Finds the first element which is higher as threshold non optimized

        sigma = np.std(hist)
        data_min = np.argwhere(hist > 100) # Finds the minimum bound for the fit
        #print(data_min[-1])
        mpv, eta, sigma, A = edges[ind_xmin], sigma, sigma, np.max(hist)
        self.log.debug("Langau first guess: {} {} {} {}".format(mpv, eta, sigma, A))

        # Fit with constrains
        converged = False
        it = 0
        oldmpv = 0
        diff = 100
        while not converged:
            it += 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # create a text trap and redirect stdout
                # Warning: astype(float) is important somehow, otherwise funny error happens one
                # some machines where it tells you double_t and float are not possible
                try:
                    coeff, pcov = curve_fit(pylandau.langau, edges[ind_xmin:-1].astype(float),
                                            hist[ind_xmin:].astype(float), absolute_sigma=False, p0=(mpv, eta, sigma, A),
                                            #bounds=([150,1,1, np.max(hist)*0.5], [300,sigma*5,sigma*5, np.max(hist)*1.5]))
                                            bounds=([data_min[-1],1,1, np.max(hist)*0.5], [edges[-1],sigma*2,sigma*2, np.max(hist)*1.5]))
                except Exception as err:
                    self.log.error("Langau fit did not converge with error: {}".format(err))
                    return [1,1,1,1], None, hist, binerror, edges
                self.log.debug("Langau coeff: {}".format(coeff))
            if abs(coeff[0] - oldmpv) > diff:
                mpv, eta, sigma, A = coeff
                oldmpv = mpv
            else:
                converged = True
            if it > 50:
                converged = True
                warnings.warn("Langau has not converged after 50 attempts!")

        return coeff, pcov, hist, binerror, edges

    def get_num_clusters(self, data, num_cluster):
        """
        Get all clusters which seem important- Here custers with numclus will be returned
        :param data: data file which should be searched
        :param num_cluster: number of cluster which should be considered. 0 makes no sense
        :return: list of data indizes after cluster consideration (so basically eventnumbers which are good)
        """
        events = []
        for clus in num_cluster:
            events.append(
                # Indizes of events with the desired clusternumbers
                np.nonzero(data["base"]["Numclus"] == clus)[0])
        return events

    def calc_hist_errors(self, x, errors, bins):
        """Calculates the errors for the bins in a histogram if error of simple point is known"""
        errorBins = np.zeros(len(bins) - 1)
        binsize = bins[1] - bins[0]

        it = 0
        for ind in bins:
            if ind != bins[-1]:
                ind_where_bin = np.where((x >= ind) & (x < (binsize + ind)))[0]
                # mu, std = norm.fit(self.CMnoise)
                if ind_where_bin.any():
                    errorBins[it] = np.mean(np.take(errors, ind_where_bin))
                it += 1

        return errorBins

    # depricated from multiprocessing
    def langau_cluster(self, cls_ind, valid_events_Signal, valid_events_clusters,
                       charge_cal, noise):
        """Calculates the energy of events, clustersize independently"""
        # for size in tqdm(clustersize_list, desc="(langau) Processing clustersize"):
        totalE = np.zeros(len(cls_ind))
        totalNoise = np.zeros(len(cls_ind))
        # Loop over the clustersize to get total deposited energy
        incrementor = 0
        start = time()
        #for ind in tqdm(cls_ind, desc="(langau) Processing event"):
        def collector(ind, incrementor):
            # Signal calculations
            signal_clst_event = np.take(valid_events_Signal[ind],
                                        valid_events_clusters[ind][0])
            totalE[incrementor] = np.sum(
                self.main.calibration.convert_ADC_to_e(signal_clst_event,
                                                       charge_cal))

            # Noise Calculations

            # Get the Noise of an event
            noise_clst_event = np.take(noise, valid_events_clusters[ind][0])
            # eError is a list containing electron signal noise
            totalNoise[incrementor] = np.sqrt(np.sum(
                self.main.calibration.convert_ADC_to_e(noise_clst_event,
                                                       charge_cal)))

            incrementor += 1

        Parallel(n_jobs=2, require='sharedmem')(delayed(collector)(ind, 0) for ind in cls_ind)

        print("*********************************************" + time()-start)

        preresults = {"signal": totalE, "noise": totalNoise}
        return preresults
