"""This file contains the class for charge sharing analysis"""
#pylint: disable=C0103

# Import statements
import numpy as np
import logging
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt
# from nb_analysisFunction import *
from analysis_classes.utilities import convert_ADC_to_e


class chargesharing:
    """ A class calculating the charge sharing between two strip clusters
    and plotting it into a histogram and a eta plot"""

    def __init__(self, main_analysis):
        """Initialize some important parameters"""
        self.main = main_analysis
        self.clustersize = 2  # Other thing would not make sense for interstrip analysis
        self.data = self.main.outputdata.copy()
        self.results_dict = {}  # Containing all data processed
        self.log = logging.getLogger()

    def run(self):
        """Runs the analysis"""
        for data in tqdm(self.data, desc="(chargesharing) Processing file:"):
            self.results_dict[data] = {}
            # Get clustersizes of 2 and only events which show only one cluster in its data (just to be sure
            indizes_clusters = np.nonzero(
                self.data[data]["base"]["Numclus"] == 1)  # Indizes of events with the desired clusternumbers
            clusters_raw = np.take(self.data[data]["base"]["Clustersize"], indizes_clusters)
            clusters_flattend = np.concatenate(clusters_raw).ravel()  # so that they are easy accessible
            indizes_clustersize = np.nonzero(
                clusters_flattend == 2)  # Indizes of events with the desired clusternumbers
            indizes = np.take(indizes_clusters, indizes_clustersize)[0]

            # Data containing the al and ar values as list entries data[0] --> al
            raw = np.take(self.data[data]["base"]["Signal"], indizes)
            raw = np.reshape(np.concatenate(raw), (len(raw), self.main.numchan))
            hits = np.concatenate(np.take(self.data[data]["base"]["Clusters"], indizes))
            al = np.zeros(len(indizes))  # Amplitude left and right
            ar = np.zeros(len(indizes))
            il = np.min(hits, axis=1)  # Indizes of left and right
            ir = np.max(hits, axis=1)
            # final_data = np.zeros((len(indizes), 2))

            for i, event, ali, ari, l, r in zip(range(len(al)), raw, al, ar, il, ir):
                al[i] = event[l]  # So always the left strip is choosen
                ar[i] = event[r]  # Same with the right strip

            # Convert ADC to actual energy
            al = convert_ADC_to_e(al, self.main.calibration.charge_cal)
            ar = convert_ADC_to_e(ar, self.main.calibration.charge_cal)

            final_data = np.array([al, ar])
            eta = ar / (al + ar)
            theta = np.arctan(ar / al)

            # Calculate the gauss distributions

            # Cut the eta in two halves and fit gaussian to it
            bins = 200
            etahist, edges = np.histogram(eta, bins=bins)
            length = len(etahist)
            mul, stdl = norm.fit(etahist[:int(length / 2)])
            mur, stdr = norm.fit(etahist[int(length / 2):])

            self.results_dict[data]["data"] = final_data
            self.results_dict[data]["eta"] = eta
            self.results_dict[data]["theta"] = theta
            self.results_dict[data]["fits"] = ((mul, stdl), (mur, stdr), edges, bins)

        return self.results_dict.copy()

    def plot(self):
        """Plots all results"""

        for file, data in self.results_dict.items():
            fig = plt.figure("Charge sharing from file: {!s}".format(file))

            # Plot delay
            plot = fig.add_subplot(221)
            counts, xedges, yedges, im = plot.hist2d(data["data"][0, :], data["data"][1, :], bins=400,
                                                     range=[[0, 50000], [0, 50000]])
            plot.set_xlabel('A_left (electrons)')
            plot.set_ylabel('A_right (electrons)')
            fig.colorbar(im)
            plot.set_title('Charge distribution interstrip')

            plot = fig.add_subplot(222)
            # left = stats.norm.pdf(data["fits"][2][:100], loc=data["fits"][0][0], scale=data["fits"][0][1])
            # right = stats.norm.pdf(data["fits"][2], loc=data["fits"][1][0], scale=data["fits"][1][1])
            # plot.plot(data["fits"][2][:100], left,"r--", color="r")
            # plot.plot(data["fits"][2], right,"r--", color="r")
            plot.set_xlabel('eta')
            plot.set_ylabel('entries')
            plot.set_title('Eta distribution')

            plot = fig.add_subplot(223)
            plot.set_xlabel('theta/Pi')
            plot.set_ylabel('entries')
            plot.set_title('Theta distribution')

            fig.suptitle('Charge sharing analysis from file {!s}'.format(file))
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            # plt.draw()