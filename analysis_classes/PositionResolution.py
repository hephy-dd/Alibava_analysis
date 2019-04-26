"""In this module the eta algorithm for position resolution will be used to determine
the position performance of the sensor. The Charge Sharing analysis must be performed prior
to this Analysis"""


import logging
import numpy as np
from scipy.stats import norm
from .utilities import set_attributes
import scipy.integrate as integrate


class PositionResolution:
    """All functions concerning the position resolution performance testing

    How does it work:
        Describe how it works...

     # Position Resolution Analysis specific params
        - Some cool params


    Written by Dominic Bloech

    """

    def __init__(self, main_analysis, configs, logger=None):
        """
        Init for the Position Resolution analysis class

        :param main_analysis: The main analysis with all its parameters
        :param configs: The dictionary with the langau specific parameters
        :param logger: A specific logger if you want
        """

        # Makes the entries of the dict to member object of the class
        set_attributes(self, configs)

        self.log = logger or logging.getLogger(__class__.__name__)
        self.main = main_analysis
        self.data = self.main.outputdata.copy()

        self.eta = self.data["ChargeSharing"]["eta"]
        self.theta = self.data["ChargeSharing"]["theta"]
        self.Neta = self.data["ChargeSharing"]["fits"]["eta"][0]
        self.Ntheta = self.data["ChargeSharing"]["fits"]["theta"][0]
        self.etaedges = self.data["ChargeSharing"]["fits"]["eta"][1]
        self.thetaedges = self.data["ChargeSharing"]["fits"]["theta"][1]

        self.results = {}



    def run(self):
        """Does all the work"""

        # Eta positions
        eta_pos = self.eta_algorithm(self.eta, self.Neta, self.etaedges)
        theta_pos = self.eta_algorithm(self.theta, self.Ntheta, self.thetaedges)

        self.results = {"eta": eta_pos, "theta": theta_pos}

        return self.results




    def eta_algorithm(self, etas, N, etaedges):

        # Calculate the diffs dN/deta
        dNdx = np.diff(N) / np.diff(etaedges)
        # Will be integrated from 0-1, because first starts at 0 and last ends at 1
        integratedeta = integrate.trapz(dNdx, etaedges)

        # Generate output array
        positions = np.zeros(len(etas), dtype=np.float)
        for i, eta in enumerate(etas):
            endedge = np.nonzero(etaedges <= eta)
            positions[i] = self.pitch*integrate.trapz(dNdx, etaedges[endedge])/integratedeta

        return positions