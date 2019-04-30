"""This file contains the class for charge sharing analysis"""
#pylint: disable=C0103,E1111
import logging
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt
from .utilities import set_attributes


class ChargeSharing:
    """ A class calculating the charge sharing between two strip clusters.

    This is particularity usefull to see if the charge is correctly shared between strips.
    For more infos how to interpret these please look for good papers =).-

    How does it work:
        - First you find all clusters of size 2.
        - Order them: Left to right channel
        - Convert ADC to electrons
        - Calculate the eta and theta distribution: eta = ar / (al + ar) and theta = np.arctan(ar / al)
          Eta is in my opinion not as good as the theta since eta is a projection on a plane and theta a
          projection in polar coordinates. The distribution looks in most cases better and is way more easy to interpret.

    """

    def __init__(self, main_analysis, configs, logger = None):
        """Initialize some important parameters"""

        # Makes the entries of the dict to member object of the class
        set_attributes(self, configs)

        self.log = logger or logging.getLogger(__class__.__name__)
        self.main = main_analysis
        self.clustersize = 2  # Other thing would not make sense for interstrip analysis
        self.data = self.main.outputdata.copy()
        self.results_dict = {}  # Containing all data processed

    def run(self):
        """Runs the analysis"""
        self.results_dict = {}
        # Get clustersizes of 2 and only events which show only one cluster in its data (just to be sure

        # Indizes of events with the desired clusternumbers
        indizes_clusters = np.nonzero(self.data["base"]["Numclus"] == 1)
        # Take the desired clusters
        clusters_raw = np.take(self.data["base"]["Clustersize"], indizes_clusters)
        clusters_flattend = np.concatenate(clusters_raw).ravel()  # so that they are easy accessible
        # Now take only the events with clustersize 2
        indizes_clustersize = np.nonzero(clusters_flattend == 2)
        # Here we need to do a little detour, since we have derived indizes from Clustersize we do not have the same
        # Index order as we need, therefore we need to get indizes from indizes_clusters to get the indizes
        # from the real data
        indizes = np.take(indizes_clusters, indizes_clustersize)[0]

        # Data containing the al and ar values as list entries data[0] --> al
        # Take the data from the Signal data
        raw = np.take(self.data["base"]["Signal"], indizes)
        raw = np.reshape(np.concatenate(raw), (len(raw), self.main.numChan))
        # Find the channels with the hits
        hits = np.concatenate(np.take(self.data["base"]["Clusters"], indizes))
        al = np.zeros(len(indizes))  # Amplitude left and right
        ar = np.zeros(len(indizes))
        strl = np.zeros(len(indizes)) # Strips hits for left side
        strr = np.zeros(len(indizes)) # Strips hits for left side
        # Order the hits assecnding (they are channel numbers)
        il = np.min(hits, axis=1)  # Indizes of left and right
        ir = np.max(hits, axis=1)
        # final_data = np.zeros((len(indizes), 2))

        for i, event, ali, ari, l, r in zip(range(len(al)), raw, al, ar, il, ir):
            al[i] = event[l]  # So always the left strip is choosen
            ar[i] = event[r]  # Same with the right strip
            strl[i] = l
            strr[i] = r

        # Convert ADC to actual energy
        al = self.main.calibration.convert_ADC_to_e(al, strl)
        ar = self.main.calibration.convert_ADC_to_e(ar, strr)

        # Calculate eta and theta
        final_data = np.array([al, ar])
        eta = ar / (al + ar)
        theta = np.arctan(ar / al)

        # Calculate the gauss distributions
        # Cut the eta in two halves and fit gaussian to it
        # Todo: not yet working correctly
        bins = 200
        etahist, edges = np.histogram(eta, bins=bins)
        thetahist, thedges = np.histogram(theta, bins=bins)
        length = len(etahist)
        mul, stdl = norm.fit(etahist[:int(length / 2)])
        mur, stdr = norm.fit(etahist[int(length / 2):])

        self.results_dict["data"] = final_data
        self.results_dict["eta"] = eta
        self.results_dict["theta"] = theta
        self.results_dict["fits"] = {}
        self.results_dict["fits"]["eta"] = (etahist, edges, bins)
        self.results_dict["fits"]["theta"] = (thetahist, thedges, bins)

        return self.results_dict.copy()
