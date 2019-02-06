"""This file contains the class for the Landau-Gauss calculation"""

# pylint: disable=C0103,E1101,R0913

import logging
import warnings
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm
import iminuit
import pylandau
# from nb_analysisFunction import *
from utilities import convert_ADC_to_e


class Langau:
    """This class calculates the langau distribution and returns the best
    values for landau and Gauss fit to the data"""

    def __init__(self, main_analysis):
        """Gets the main analysis class and imports all things needed for
        its calculations"""

        self.log = logging.getLogger(__class__.__name__)
        self.log.setLevel(logging.DEBUG)
        if self.log.hasHandlers() is False:
            format_string = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
            formatter = logging.Formatter(format_string)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.log.addHandler(console_handler)

        self.main = main_analysis
        self.data = self.main.outputdata.copy()
        self.results_dict = {}  # Containing all data processed
        self.pedestal = self.main.pedestal
        self.pool = self.main.Pool
        self.poolsize = self.main.process_pool

    def run(self):
        """Calculates the langau for the specified data"""

        # Which clusters need to be considered
        clustersize_list = self.main.kwargs["configs"].get(
            "langau", {}).get("clustersize", [-1])
        if not isinstance(clustersize_list, list):
            clustersize_list = list(clustersize_list)

        if clustersize_list[0] == -1:
            # If nothing is specified
            clustersize_list = list(range(\
                    1, self.main.kwargs["configs"]["max_cluster_size"] + 1))

        # Go over all datafiles
        for data in tqdm(self.data, desc="(langau) Processing file:"):
            self.results_dict[data] = {}
            # Here events with only one cluster are choosen
            indizes = self.get_num_clusters(self.data[data], 1)[0]
            valid_events_clustersize = np.take(
                self.data[data]["base"]["Clustersize"],
                indizes)  # Get the clustersizes of valid events
            valid_events_clusters = np.take(
                self.data[data]["base"]["Clusters"], indizes)
            valid_events_Signal = np.take(
                self.data[data]["base"]["Signal"],
                indizes)  # Get the clustersizes of valid events
            # Get events which show only cluster in its data
            charge_cal, noise = self.main.calibration.charge_cal, self.main.noise
            self.results_dict[data]["Clustersize"] = []

            # General langau, where all clustersizes are considered
            if self.main.usejit and self.poolsize > 1:
                paramslist = []
                for size in clustersize_list:
                    cls_ind = np.nonzero(valid_events_clustersize == size)[0]
                    paramslist.append(
                        (cls_ind,
                         valid_events_Signal,
                         valid_events_clusters,
                         self.main.calibration.charge_cal,
                         self.main.noise))

                # Here multiple cpu calculate the energy of the events per
                # clustersize
                # COMMENT: langau_cluster is not defined!!!
                results = self.pool.starmap(langau_cluster,
                                            paramslist,
                                            chunksize=1)

                self.results_dict[data]["Clustersize"] = results

            else:
                for size in tqdm(clustersize_list,
                                 desc="(langau) Processing clustersize"):
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
                        signal_clst_event.append(
                            np.take(
                                valid_events_Signal[ind],
                                valid_events_clusters[ind][0]))
                        # Noise Calculations
                        # Get the Noise of an event
                        noise_clst_event.append(
                            np.take(noise, valid_events_clusters[ind][0]))

                    totalE = np.sum(
                        convert_ADC_to_e(signal_clst_event, charge_cal), axis=1)
                    # eError is a list containing electron signal noise
                    totalNoise = np.sqrt(np.sum(\
                        convert_ADC_to_e(noise_clst_event, charge_cal), axis=1))

                    #incrementor += 1

                    preresults = {}
                    preresults["signal"] = totalE
                    preresults["noise"] = totalNoise

                    self.results_dict[data]["Clustersize"].append(preresults)

            # With all the data from every clustersize add all together and fit
            # the langau to it
            finalE = np.zeros(0)
            finalNoise = np.zeros(0)
            for cluster in self.results_dict[data]["Clustersize"]:
                finalE = np.append(finalE, cluster["signal"])
                finalNoise = np.append(finalNoise, cluster["noise"])

            # Fit the langau to it
            # COMMENT: pcov, hist unused? consider
            coeff, pcov, hist, error_bins = self.fit_langau(finalE, finalNoise)
            self.results_dict[data]["signal"] = finalE
            self.results_dict[data]["noise"] = finalNoise
            self.results_dict[data]["langau_coeff"] = coeff
            self.results_dict[data]["langau_data"] = [
                np.arange(1., 100000., 1000.),
                # aka x and y data
                pylandau.langau(np.arange(1., 100000., 1000.), *coeff)]
            self.results_dict[data]["data_error"] = error_bins

            # Consider now only the seedcut hits for the langau,
            if self.main.kwargs["configs"].get("langau", {})\
                                          .get("seed_cut_langau", False):
                seed_cut_channels = self.data[data]["base"]["Channel_hit"]
                signals = self.data[data]["base"]["Signal"]
                finalE = []
                seedcutADC = []
                for i, signal in enumerate(
                        tqdm(signals, desc="(langau SC) Processing events")):
                    if signal[seed_cut_channels[i]].any():
                        seedcutADC.append(signal[seed_cut_channels[i]])

                self.log.info("Converting ADC to electrons...")
                converted = convert_ADC_to_e(seedcutADC, charge_cal)
                for conv in converted:
                    finalE.append(sum(conv))
                finalE = np.array(finalE, dtype=np.float32)

                # get rid of 0 events
                indizes = np.nonzero(finalE > 0)[0]
                nogarbage = finalE[indizes]
                 # ultra_high_energy_cut
                indizes = np.nonzero(nogarbage < 120000)[0]
                coeff, pcov, hist, error_bins = self.fit_langau(
                    nogarbage[indizes], bins=500)
                self.results_dict[data]["signal_SC"] = nogarbage[indizes]
                self.results_dict[data]["langau_coeff_SC"] = coeff
                self.results_dict[data]["langau_data_SC"] = [\
                        np.arange(1., 100000., 1000.),
                        # aka x and y data
                        pylandau.langau(np.arange(1., 100000., 1000.), *coeff)]

        return self.results_dict.copy()

    def fit_landau_migrad(self, x, y, p0, limit_mpv, limit_eta,
                          limit_sigma, limit_A):
        # TODO make it possible with error calculation



        # Prefit to get correct errors
        yerr = np.sqrt(y)  # Assume error from measured data
        yerr[y < 1] = 1
        # COMMENT: iminuit not in requirements
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
        # COMMENT: where does langau come from? are you creating a LANGAU object inside a LANGAU object?
        yerr = np.sqrt(langau(x, mpv=m.values['mpv'],
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
            self.log.warning("MPV error determination with Minos failed! "
                             "You can still use Hesse errors.")

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
        # TODO: Bug when using optimized vs non optimized !!!
        try:
            # Finds the first element which is higher as threshold optimized
            ind_xmin = np.argwhere(hist > lancut)[0][0]
        except BaseException:
            # Finds the first element which is higher as threshold non
            # optimized
            ind_xmin = np.argwhere(hist > lancut)[0]

        mpv, eta, sigma, A = 18000, 1500, 5000, np.max(hist)

        # Fit with constrains
        converged = False
        iter_index = 0
        oldmpv = 0
        diff = 100
        while not converged:
            iter_index += 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # create a text trap and redirect stdout
                # Warning: astype(float) is importanmt somehow, otherwise funny
                # error happens one some machines where it tells you double_t
                # and float are not possible
                coeff, pcov = curve_fit(pylandau.langau,
                                        edges[ind_xmin:-1].astype(float),
                                        hist[ind_xmin:].astype(float),
                                        absolute_sigma=True,
                                        p0=(mpv, eta, sigma, A),
                                        bounds=(1, 500000))
            if abs(coeff[0] - oldmpv) > diff:
                mpv, eta, sigma, A = coeff
                oldmpv = mpv
            else:
                converged = True
            if iter_index > 50:
                converged = True
                warnings.warn("Langau has not converged after 50 attempts!")

        return coeff, pcov, hist, binerror

    def get_num_clusters(self, data, num_cluster=1):
        """
        Get all clusters which seem important- Here custers with numclus will
        be returned.

        :param data: data file which should be searched
        :param num_cluster: number of cluster which should be considered 1 is
                            default and minimum. 0 makes no sense
        :return: list of data indizes after cluster consideration
                 (so basically eventnumbers which are good)
        """
        return np.nonzero(
            data["base"]["Numclus"] == num_cluster)  # Indizes of events with the desired clusternumbers

    def calc_hist_errors(self, x, errors, bins):
        """Calculates the errors for the bins in a histogram if error of
        simple point is known"""
        errorBins = np.zeros(len(bins) - 1)
        binsize = bins[1] - bins[0]

        iter_index = 0
        for ind in bins:
            if ind != bins[-1]:
                ind_where_bin = np.where((x >= ind) & (x < (binsize + ind)))[0]
                #mu, std = norm.fit(self.CMnoise)
                if ind_where_bin.any():
                    errorBins[iter_index] = np.mean(np.take(errors, ind_where_bin))
                iter_index += 1

        return errorBins

    def plot(self):
        """Plots the data calculated so the energy data and the langau"""

        for file, data in self.results_dict.items():
            fig = plt.figure("Langau from file: {!s}".format(file))

            # Plot delay
            plot = fig.add_subplot(111)
            hist, edges = np.histogram(data["signal"], bins=500)
            plot.hist(
                data["signal"],
                bins=500,
                density=False,
                alpha=0.4,
                color="b",
                label="All clusters")
            plot.errorbar(edges[:-1],
                          hist,
                          xerr=data["data_error"],
                          fmt='o',
                          markersize=1,
                          color="red")
            plot.plot(
                data["langau_data"][0],
                data["langau_data"][1],
                "r--",
                color="g",
                label="Langau: \n mpv: {mpv!s} \n eta: {eta!s} \n sigma: {sigma!s} \n A: {A!s} \n".format(
                    mpv=data["langau_coeff"][0],
                    eta=data["langau_coeff"][1],
                    sigma=data["langau_coeff"][2],
                    A=data["langau_coeff"][3]))
            plot.set_xlabel('electrons [#]')
            plot.set_ylabel('Count [#]')
            plot.set_title('All clusters Langau from file: {!s}'.format(file))
            #plot.legend(["Langau: \n mpv: {mpv!s} \n eta: {eta!s} \n sigma: {sigma!s} \n A: {A!s} \n".format(mpv=data["langau_coeff"][0],eta=data["langau_coeff"][1],sigma=data["langau_coeff"][2],A=data["langau_coeff"][3])])

            # Plot the different clustersizes as well into the langau plot
            colour = [
                'green',
                'red',
                'orange',
                'cyan',
                'black',
                'pink',
                'magenta']
            for i, cls in enumerate(data["Clustersize"]):
                if i < 7:
                    plot.hist(
                        cls["signal"],
                        bins=500,
                        density=False,
                        alpha=0.3,
                        color=colour[i],
                        label="Clustersize: {!s}".format(
                            i + 1))
                else:
                    warnings.warn(
                        "To many histograms for this plot. Colorsheme only supports seven different histograms. Extend if need be!")
                    continue

            plot.set_xlim(0, 100000)
            plot.legend()
            fig.tight_layout()
            # plt.draw()

            if self.main.kwargs["configs"].get(
                    "langau",
                    {}).get(
                    "seed_cut_langau",
                    False):
                fig = plt.figure(
                    "Seed cut langau from file: {!s}".format(file))

                # Plot Seed cut langau
                plot = fig.add_subplot(111)
                #indizes = np.nonzero(data["signal_SC"] > 0)[0]
                plot.hist(
                    data["signal_SC"],
                    bins=500,
                    density=False,
                    alpha=0.4,
                    color="b",
                    label="Seed clusters")
                plot.plot(
                    data["langau_data_SC"][0],
                    data["langau_data_SC"][1],
                    "r--",
                    color="g",
                    label="Langau: \n mpv: {mpv!s} \n eta: {eta!s} \n sigma: {sigma!s} \n A: {A!s} \n".format(
                        mpv=data["langau_coeff_SC"][0],
                        eta=data["langau_coeff_SC"][1],
                        sigma=data["langau_coeff_SC"][2],
                        A=data["langau_coeff_SC"][3]))
                plot.set_xlabel('electrons [#]')
                plot.set_ylabel('Count [#]')
                plot.set_title('Seed cut Langau from file: {!s}'.format(file))
                plot.set_xlim(0, 100000)
                plot.legend()


def minimizeMe(x, y, mpv, eta, sigma, A, yerr):
    chi2 = np.sum(np.square(\
                    y - pylandau.langau(x, mpv, eta, sigma, A)\
                    .astype(float)) / np.square(yerr.astype(float)))
    # devide by NDF
    return chi2 / (x.shape[0] - 5)
