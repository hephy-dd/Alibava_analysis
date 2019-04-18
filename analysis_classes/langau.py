"""This file contains the class for the Landau-Gauss calculation"""
# pylint: disable=C0103,E1101,R0913,C0301,E0401
import logging
import warnings
import time
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
import pylandau
<<<<<<< HEAD
from joblib import Parallel, delayed
=======
from analysis_classes.utilities import convert_ADC_to_e, manage_logger
#from joblib import Parallel, delayed
from time import time
#from numba import jit

>>>>>>> Dominic_dev

class Langau:
    """This class calculates the langau distribution and returns the best
    values for landau and Gauss fit to the data"""
    def __init__(self, main_analysis, configs, logger=None):
        """Gets the main analysis class and imports all things needed for its calculations"""

        self.log = logger or logging.getLogger(__class__.__name__)
        self.main = main_analysis
        self.data = self.main.outputdata.copy()
        self.results_dict = {}  # Containing all data processed
        self.pedestal = self.main.pedestal
        self.pool = self.main.Pool
        self.poolsize = self.main.process_pool
<<<<<<< HEAD
        self.numClusters = configs.get("langau", {}).get("numClus", 1)
        self.Ecut = configs.get("langau", {}).get("energyCutOff", 150000)
        self.plotfit = configs.get("langau", {}).get("fitLangau", True)
        self.max_cluster_size = configs.get("max_cluster_size", 3)
        self.cluster_size_list = configs.get("langau", {}).get("clustersize", [1, 2, 3])
        self.results_dict = {"bins": configs.get("langau", {}).get("bins", 500)}
        self.seed_cut_langau = configs.get("langau", {}).get("seed_cut_langau", False)
=======
        self.numClusters = self.main.kwargs["configs"].get("langau", {}).get("numClus", 1)
        self.bins = self.main.kwargs["configs"].get("langau", {}).get("bins", 500)
        self.Ecut = self.main.kwargs["configs"].get("langau", {}).get("energyCutOff", 150000)
        self.plotfit = self.main.kwargs["configs"].get("langau", {}).get("fitLangau", False)
>>>>>>> Dominic_dev

    def run(self):
        """Calculates the langau for the specified data"""
        indNumClus = self.get_num_clusters(self.data,
                                           self.numClusters)  # Here events with only one cluster are choosen
        indizes = np.concatenate(indNumClus)
        valid_events_clustersize = np.take(self.data["base"]["Clustersize"], indizes)
        valid_events_clusters = np.take(self.data["base"]["Clusters"], indizes)
        # Get the clustersizes of valid events
        valid_events_Signal = np.take(self.data["base"]["Signal"],
                                      indizes)
        # Get events which show only cluster in its data
        self.results_dict["Clustersize"] = []

        # General langau, where all clustersizes are considered
        if self.main.usejit and self.poolsize > 1:
            paramslist = []
            for size in self.cluster_size_list:
                cls_ind = np.nonzero(valid_events_clustersize == size)[0]
                paramslist.append((cls_ind, valid_events_Signal,
                                   valid_events_clusters,
                                   self.main.calibration.convert_ADC_to_e,
                                   self.main.noise))

            # COMMENT: lagau_cluster not defined!!!!
            # Here multiple cpu calculate the energy of the events per clustersize
            results = self.pool.starmap(self.langau_cluster, paramslist,
                                        chunksize=1)

            self.results_dict["Clustersize"] = results

<<<<<<< HEAD
        else:
            self.cluster_analysis(valid_events_Signal,
                                  valid_events_clusters,
                                  valid_events_clustersize)

        # With all the data from every clustersize add all together and fit the langau to it
        finalE = np.zeros(0)
        finalNoise = np.zeros(0)
        for cluster in self.results_dict["Clustersize"]:
            indi = np.nonzero(cluster["signal"] > 0)[0]  # Clean up and extra energy cut
            nogarbage = cluster["signal"][indi]
            indi = np.nonzero(nogarbage < self.Ecut)[0]  # ultra_high_energy_cut
            cluster["signal"] = cluster["signal"][indi]
            finalE = np.append(finalE, cluster["signal"])
            finalNoise = np.append(finalNoise, cluster["noise"])

        # Fit the langau to it
        coeff, _, _, error_bins = self.fit_langau(
            finalE, finalNoise, bins=self.results_dict["bins"])
        self.results_dict["signal"] = finalE
        self.results_dict["noise"] = finalNoise
        self.results_dict["langau_coeff"] = coeff
        self.results_dict["langau_data"] = [
            np.arange(1., 100000., 1000.),
            pylandau.langau(np.arange(1., 100000., 1000.),
                            *coeff)]  # aka x and y data
        self.results_dict["data_error"] = error_bins

        # Consider now only the seedcut hits for the langau,
        if self.seed_cut_langau:
            seed_cut_channels = self.data["base"]["Channel_hit"]
            signals = self.data["base"]["Signal"]
            finalE = []
            seedcutADC = []
            for i, signal in enumerate(tqdm(signals, desc="(langau SC) Processing events")):
                if signal[seed_cut_channels[i]].any():
                    seedcutADC.append(signal[seed_cut_channels[i]])

            self.log.info("Converting ADC to electrons for SC Langau...")
            converted = self.main.calibration.convert_ADC_to_e(seedcutADC)
            for conv in converted:
                finalE.append(sum(conv))
            finalE = np.array(finalE, dtype=np.float32)

            # get rid of 0 events
            indizes = np.nonzero(finalE > 0)[0]
            nogarbage = finalE[indizes]
            indizes = np.nonzero(nogarbage < self.Ecut)[0]  # ultra_high_energy_cut
            coeff, _, _, error_bins = self.fit_langau(
                nogarbage[indizes], bins=self.results_dict["bins"])
            # coeff, pcov, hist, error_bins = self.fit_langau(nogarbage, bins=500)
            self.results_dict["signal_SC"] = nogarbage[indizes]
            self.results_dict["langau_coeff_SC"] = coeff
            self.results_dict["langau_data_SC"] = [
                np.arange(1., 100000., 1000.),
                pylandau.langau(np.arange(1., 100000., 1000.),
                                *coeff)]  # aka x and y data
=======
        # Which clusters need to be considered
        clustersize_list = self.main.kwargs["configs"].get("langau", {}).get("clustersize", [-1])
        if isinstance(clustersize_list, list):
            clustersize_list = list(clustersize_list)

        if clustersize_list[0] == -1:
            clustersize_list = list(
                range(1, self.main.kwargs["configs"]["max_cluster_size"] + 1))  # If nothing is specified

        # Go over all datafiles
        #self.log.info("Doing Langau over all files:")
        for i, data in enumerate(self.data):
            print("") # Otherwise progress bar will be crippled
            #self.log.info("Processing file {!s} of {!s}".format(i+1, len(self.data)))
            self.results_dict[data] = {}
            indNumClus = self.get_num_clusters(self.data[data],
                                               self.numClusters)  # Here events with only one cluster are choosen
            indizes = np.concatenate(indNumClus)
            valid_events_clustersize = np.take(self.data[data]["base"]["Clustersize"], indizes)
            valid_events_clusters = np.take(self.data[data]["base"]["Clusters"], indizes)
            valid_events_Signal = np.take(self.data[data]["base"]["Signal"],
                                          indizes)  # Get the clustersizes of valid events
            # Get events which show only cluster in its data
            charge_cal, noise = self.main.calibration.charge_cal, self.main.noise
            self.results_dict[data]["Clustersize"] = []


            # Try joblib
            #start = time()
            #arg_instances = [(size, valid_events_clustersize,
            #                  valid_events_Signal, valid_events_clusters,
            #                  noise, charge_cal) for size in clustersize_list]
            #results = Parallel(n_jobs=4, backend="threading")(map(delayed(self.process_cluster_size),
            #                                                           arg_instances))
            #for res in results:
            #    self.results_dict[data]["Clustersize"].append(res)
            #print(time()-start)

            #start = time()
            for size in tqdm(clustersize_list, desc="(langau) Processing clustersize"):
                # get the events with the different clustersizes
                ClusInd = [[], []]
                for i, event in enumerate(valid_events_clustersize):
                    # cls_ind = np.nonzero(valid_events_clustersize == size)[0]
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

                # COMMENT: axis=1 produces numpy.AxisError?
                # totalE = np.sum(convert_ADC_to_e(signal_clst_event, charge_cal), axis=1)
                totalE = np.sum(convert_ADC_to_e(signal_clst_event, charge_cal), axis=1)

                # eError is a list containing electron signal noise
                # totalNoise = np.sqrt(np.sum(convert_ADC_to_e(noise_clst_event, charge_cal),
                #                             axis=1))
                totalNoise = np.sqrt(np.sum(convert_ADC_to_e(noise_clst_event, charge_cal), axis=1))

                # incrementor += 1

                preresults = {"signal": totalE, "noise": totalNoise}

                self.results_dict[data]["Clustersize"].append(preresults)
            #print(time() - start)


            # With all the data from every clustersize add all together and fit the langau to it
            finalE = np.zeros(0)
            finalNoise = np.zeros(0)
            for cluster in self.results_dict[data]["Clustersize"]:
                indi = np.nonzero(cluster["signal"] > 0)[0]  # Clean up and extra energy cut
                nogarbage = cluster["signal"][indi]
                indi = np.nonzero(nogarbage < self.Ecut)[0]  # ultra_high_energy_cut
                cluster["signal"] = cluster["signal"][indi]
                finalE = np.append(finalE, cluster["signal"])
                finalNoise = np.append(finalNoise, cluster["noise"])

            # Fit the langau to it
            if self.plotfit:
                coeff, pcov, hist, error_bins = self.fit_langau(finalE, finalNoise, bins=self.bins)
                self.results_dict[data]["langau_coeff"] = coeff
                self.results_dict[data]["langau_data"] = [np.arange(1., 100000., 1000.),
                                                          pylandau.langau(np.arange(1., 100000., 1000.),
                                                                          *coeff)]  # aka x and y data
                self.results_dict[data]["data_error"] = error_bins
            self.results_dict[data]["signal"] = finalE
            self.results_dict[data]["noise"] = finalNoise

            # Consider now only the seedcut hits for the langau,
            if self.main.kwargs["configs"].get("langau", {}).get("seed_cut_langau", False):
                seed_cut_channels = self.data[data]["base"]["Channel_hit"]
                signals = self.data[data]["base"]["Signal"]
                finalE = []
                seedcutADC = []
                for i, signal in enumerate(tqdm(signals, desc="(langau SC) Processing events")):
                    if signal[seed_cut_channels[i]].any():
                        seedcutADC.append(signal[seed_cut_channels[i]])

                self.log.info("Converting ADC to electrons for SC Langau...")
                converted = convert_ADC_to_e(seedcutADC, charge_cal)
                for conv in converted:
                    finalE.append(sum(conv))
                finalE = np.array(finalE, dtype=np.float32)

                # get rid of 0 events
                indizes = np.nonzero(finalE > 0)[0]
                nogarbage = finalE[indizes]
                indizes = np.nonzero(nogarbage < self.Ecut)[0]  # ultra_high_energy_cut
                if self.plotfit:
                    coeff, pcov, hist, error_bins = self.fit_langau(nogarbage[indizes], bins=self.bins)
                    self.results_dict[data]["langau_coeff_SC"] = coeff
                    self.results_dict[data]["langau_data_SC"] = [np.arange(1., 100000., 1000.),
                                                                 pylandau.langau(np.arange(1., 100000., 1000.), *coeff)]
                    # aka x and y data
                    # coeff, pcov, hist, error_bins = self.fit_langau(nogarbage, bins=500)
                self.results_dict[data]["signal_SC"] = nogarbage[indizes]

>>>>>>> Dominic_dev

        return self.results_dict.copy()

    def cluster_analysis(self, valid_events_Signal,
                         valid_events_clusters, valid_events_clustersize):
        """Perform cluster analysis"""
        for size in tqdm(self.cluster_size_list, desc="(langau) Processing clustersize"):
            # get the events with the different clustersizes
            ClusInd = [[], []]
            for i, event in enumerate(valid_events_clustersize):
                # cls_ind = np.nonzero(valid_events_clustersize == size)[0]
                for j, clus in enumerate(event):
                    if clus == size:
                        ClusInd[0].extend([i])
                        ClusInd[1].extend([j])

            signal_clst_event = []
            noise_clst_event = []
            for i, ind in enumerate(tqdm(ClusInd[0], desc="(langau) Processing event")):
                y = ClusInd[1][i]
                # Signal calculations
                signal_clst_event.append(np.take(valid_events_Signal[ind], valid_events_clusters[ind][y]))
                # Noise Calculations
                noise_clst_event.append(
                    np.take(self.main.noise, valid_events_clusters[ind][y]))  # Get the Noise of an event

            totalE = np.sum(
                self.main.calibration.convert_ADC_to_e(signal_clst_event),
                axis=1)

            # eError is a list containing electron signal noise
            totalNoise = np.sqrt(np.sum(
                self.main.calibration.convert_ADC_to_e(noise_clst_event),
                axis=1))

            # incrementor += 1

            preresults = {"signal": totalE, "noise": totalNoise}

            self.results_dict["Clustersize"].append(preresults)

    def fit_langau(self, x, errors=np.array([]), bins=500):
        """Fits the langau to data"""
        hist, edges = np.histogram(x, bins=bins)
        if errors.any():
            binerror = self.calc_hist_errors(x, errors, edges)
        else:
            binerror = np.array([])

        # Cut off noise part
        lancut = np.max(hist) * 0.33  # Find maximum of hist and get the cut
        # TODO: Bug when using optimized vs non optimized !!!
        try:
            ind_xmin = np.argwhere(hist > lancut)[0][0]
            # Finds the first element which is higher as threshold optimized
        except:
            ind_xmin = np.argwhere(hist > lancut)[0]
            # Finds the first element which is higher as threshold non optimized

        mpv, eta, sigma, A = 27000, 1500, 5000, np.max(hist)

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
                # Warning: astype(float) is importanmt somehow, otherwise funny error happens one
                # some machines where it tells you double_t and float are not possible
                coeff, pcov = curve_fit(pylandau.langau, edges[ind_xmin:-1].astype(float),
                                        hist[ind_xmin:].astype(float), absolute_sigma=True, p0=(mpv, eta, sigma, A),
                                        bounds=(1, 500000))
            if abs(coeff[0] - oldmpv) > diff:
                mpv, eta, sigma, A = coeff
                oldmpv = mpv
            else:
                converged = True
            if it > 50:
                converged = True
                warnings.warn("Langau has not converged after 50 attempts!")

        return coeff, pcov, hist, binerror

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
                np.nonzero(data["base"]["Numclus"] == clus)[0])  # Indizes of events with the desired clusternumbers
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

<<<<<<< HEAD
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
=======
    #@jit(nopython=False, nogil=True, cache=True)
    def process_cluster_size(self, args):
        # get the events with the different clustersizes
        size, valid_events_clustersize, valid_events_Signal, valid_events_clusters, noise, charge_cal = args
        ClusInd = [[], []]
        for i, event in enumerate(valid_events_clustersize):
            # cls_ind = np.nonzero(valid_events_clustersize == size)[0]
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
        totalE = np.sum(convert_ADC_to_e(signal_clst_event, charge_cal), axis=1)

        # eError is a list containing electron signal noise
        # totalNoise = np.sqrt(np.sum(convert_ADC_to_e(noise_clst_event, charge_cal),
        #                             axis=1))
        totalNoise = np.sqrt(np.sum(convert_ADC_to_e(noise_clst_event, charge_cal), axis=1))

        # incrementor += 1

        preresults = {"signal": totalE, "noise": totalNoise}

        #self.results_dict[data]["Clustersize"].append(preresults)
        return preresults


    def plot(self):
        """Plots the data calculated so the energy data and the langau"""

        for file, data in self.results_dict.items():
            fig = plt.figure("Langau from file: {!s}".format(file))

            # Plot delay
            plot = fig.add_subplot(111)
            hist, edges = np.histogram(data["signal"], bins=self.bins)
            plot.hist(data["signal"], bins=self.bins, density=False, alpha=0.4, color="b", label="All clusters")
            if self.plotfit:
                plot.plot(data["langau_data"][0], data["langau_data"][1], "r--",
                            color="g",
                            label="Langau: \n mpv: {mpv!s} \n eta: {eta!s} \n sigma: {sigma!s} \n A: {A!s} \n".format(
                            mpv=data["langau_coeff"][0],
                            eta=data["langau_coeff"][1],
                            sigma=data["langau_coeff"][2],
                            A=data["langau_coeff"][3]))
                plot.errorbar(data["langau_data"][0], data["langau_data"][1],
                              np.sqrt(pylandau.langau(data["langau_data"][0], *data["langau_coeff"])),
                              fmt=".", color="r", label="Error of Fit")

            plot.set_xlabel('electrons [#]')
            plot.set_ylabel('Count [#]')
            plot.set_title('All clusters Langau from file: {!s}'.format(file))

            # Plot the different clustersizes as well into the langau plot
            colour = ['green', 'red', 'orange', 'cyan', 'black', 'pink', 'magenta']
            for i, cls in enumerate(data["Clustersize"]):
                if i < 7:
                    plot.hist(cls["signal"], bins=self.bins, density=False, alpha=0.3, color=colour[i],
                              label="Clustersize: {!s}".format(i + 1))
                else:
                    warnings.warn(
                        "To many histograms for this plot. "
                        "Colorsheme only supports seven different histograms. Extend if need be!")
                    continue

            plot.legend()
            fig.tight_layout()

            if self.main.kwargs["configs"].get("langau", {}).get("seed_cut_langau", False):
                fig = plt.figure("Seed cut langau from file: {!s}".format(file))

                # Plot Seed cut langau
                plot = fig.add_subplot(111)
                # indizes = np.nonzero(data["signal_SC"] > 0)[0]
                plot.hist(data["signal_SC"], bins=self.bins, density=False, alpha=0.4, color="b", label="Seed clusters")
                if self.plotfit:
                    plot.plot(data["langau_data_SC"][0], data["langau_data_SC"][1], "r--", color="g",
                            label="Langau: \n mpv: {mpv!s} \n eta: {eta!s} \n sigma: {sigma!s} \n A: {A!s} \n".format(
                            mpv=data["langau_coeff_SC"][0], eta=data["langau_coeff_SC"][1],
                            sigma=data["langau_coeff_SC"][2],
                            A=data["langau_coeff_SC"][3]))
                    plot.errorbar(data["langau_data_SC"][0], data["langau_data_SC"][1],
                                  np.sqrt(pylandau.langau(data["langau_data_SC"][0], *data["langau_coeff_SC"])),
                                  fmt=".", color="r", label="Error of Fit")
                plot.set_xlabel('electrons [#]')
                plot.set_ylabel('Count [#]')
                plot.set_title('Seed cut Langau from file: {!s}'.format(file))
                plot.legend()
>>>>>>> Dominic_dev
