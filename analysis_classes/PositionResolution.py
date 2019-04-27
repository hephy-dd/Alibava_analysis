"""In this module the eta algorithm for position resolution will be used to determine
the position performance of the sensor. The Charge Sharing analysis must be performed prior
to this Analysis"""


import logging
import numpy as np
from .utilities import set_attributes
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter


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
        eta_pos, Neta = self.eta_algorithm(self.eta, self.Neta, self.etaedges)
        theta_pos, Ntheta = self.eta_algorithm(self.theta, self.Ntheta, self.thetaedges)

        self.results = {"eta": eta_pos, "theta": theta_pos, "N_theta": Ntheta, "N_eta": Neta}

        return self.results

    def eta_algorithm(self, etas, N, etaedges):

        if self.SavGol:
            params = self.SavGol_params
            self.log.debug("Applying SavGol filter to input data...")
            # Todo: SavGol filter gives FutureError when using this. WEarning that array slicing will be done differently in the future. I dont see why my code is the reason for that
            for i in range(self.SavGol_iter):
                N = savgol_filter(N, params[0], params[1])

        #cs = CubicSpline(etaedges[:-1], N)
        # Calculate the diffs dN/deta
        #csderiv = cs.derivative(1)
        #csfullIntegrate = float(csderiv.integrate(etaedges[0],etaedges[-1]))

        # Todo: use trapezoid method to integrate???
        diffs = np.diff(etaedges)
        csfullIntegrate = np.sum(N*diffs)

        # Generate output array
        positions = np.zeros(len(etas), dtype=np.float)
        for i, eta in enumerate(etas):
            endedge = np.nonzero(etaedges <= eta)[0]
            #positions[i] = self.pitch*csderiv.integrate(0, etas[endedge[-1]])/csfullIntegrate
            integ = np.sum(N[endedge-1]*diffs[endedge-1])
            positions[i] = self.pitch*integ/csfullIntegrate

        return positions, N