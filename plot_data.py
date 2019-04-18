"""PlotData Class"""
# pylint: disable=R0201,C0103,E0401
import numpy as np
import logging
from scipy.stats import norm
import matplotlib.pyplot as plt
# import pylandau
from analysis_classes.utilities import handle_sub_plots, gaussian
# from analysis_classes.utilities import read_file

class PlotData:
    """Plots for ALiBaVa Analysis"""
    def __init__(self, logger=None):
        self.log = logger or logging.getLogger(__class__.__name__)
        # canvas for plotting the data [width, height (inches)]
        self.ped_fig = None
        self.cal_fig = None
        self.main_fig = None

        # self.cfg = read_file("plot_cfg.yml")

        self.ped_plots = [self.plot_noise_ch,
                          self.plot_pedestal,
                          self.plot_cm,
                          self.plot_noise_hist]
        self.cal_plots = [self.plot_signal_conversion_fit,
                          self.plot_signal_conversion_fit_detail,
                          self.plot_gain_hist]
        self.single_event_plots = [self.plot_single_event_SN,
                                   self.plot_single_event_ch]

        self.main_plots = [self.plot_cluster_hist,
                           self.plot_clustersizes,
                           self.plot_hitmap,
                           self.plot_langau_per_clustersize,
                           self.plot_seed_signal_e]

        self.groups = {"pedestal": self.ped_plots,
                       "calibration": self.cal_plots,
                       "single_event": self.single_event_plots,
                       "main": self.main_plots,
                       }
        self.groups["Brace yourself! plots are comming"] = np.concatenate([x for x in self.groups.items()])

    def plot_data(self, obj, group=None, fig_name=None):
        """Plots the data calculated by the framework. Suppress drawing and
        showing the canvas by setting "show" to False.
        Returns matplotlib.pyplot.figure object.
        """

        if group == "all" or group == None:
            for grp in self.groups:
                if fig_name is None:
                    fig_name = grp
                fig = plt.figure(fig_name, figsize=[10, 8])
                for func in self.groups[grp]:
                    func(obj, fig)
                fig.tight_layout()

        else:
            if group in self.groups:
                if fig_name is None:
                    fig_name = group
                fig = plt.figure(fig_name, figsize=[10, 8])
                for func in self.groups[group]:
                    func(obj, fig)
                fig.tight_layout()
            else:
                self.log.error("Plotting could not be done."
                               " Plot group {} was not recognised. Skipping!".format(group))
        # if group == "pedestal":
        #     if fig_name is None:
        #         fig_name = "Pedestal analysis"
        #     fig = plt.figure(fig_name, figsize=[10, 8])
        #     for func in self.ped_plots:
        #         func(obj, fig)
        #     fig.tight_layout()
        # if group == "calibration":
        #     if fig_name is None:
        #         fig_name = "Calibration analysis"
        #     fig = plt.figure(fig_name, figsize=[10, 8])
        #     for func in self.cal_plots:
        #         func(obj, fig)
        #     fig.tight_layout()
        # if group == "main":
        #     # self.main_fig = plt.figure("Main analysis", figsize=[10, 8])
        #     for func in self.main_plots:
        #         plot = func(obj)
        #         plt.gcf().tight_layout()

    def show_plots(self):
        """Draw and show plots"""
        plt.draw()
        plt.show()

    ### Pedestal Plots ###
    def plot_noise_ch(self, obj, fig=None):
        """plot noise per channel"""
        noise_plot = handle_sub_plots(fig, 221)
        noise_plot.bar(np.arange(obj.numchan), obj.noise, 1.,
                       alpha=0.4, color="b", label="Noise level per strip")
        # plot line idicating masked and unmasked channels
        valid_strips = np.zeros(obj.numchan)
        valid_strips[obj.noisy_strips] = 1
        noise_plot.plot(np.arange(obj.numchan), valid_strips, color="r",
                        label="Masked strips")

        # Plot the threshold for deciding a good channel
        xval = [0, obj.numchan]
        yval = [obj.median_noise + obj.noise_cut,
                obj.median_noise + obj.noise_cut]
        noise_plot.plot(xval, yval, "r--", color="g",
                        label="Threshold for noisy strips")

        noise_plot.set_xlabel('Channel [#]')
        noise_plot.set_ylabel('Noise [ADC]')
        noise_plot.set_title('Noise levels per Channel')
        noise_plot.legend(loc='upper right')
        return noise_plot

    def plot_pedestal(self, obj, fig=None):
        """Plot pedestal and noise per channel"""
        pede_plot = handle_sub_plots(fig, 222)
        pede_plot.bar(np.arange(obj.numchan), obj.pedestal, 1., yerr=obj.noise,
                      error_kw=dict(elinewidth=0.2, ecolor='r', ealpha=0.1),
                      alpha=0.4, color="b", label="Pedestal")
        pede_plot.set_xlabel('Channel [#]')
        pede_plot.set_ylabel('Pedestal [ADC]')
        pede_plot.set_title('Pedestal levels per Channel with noise')
        pede_plot.set_ylim(bottom=min(obj.pedestal) - 50.)
        pede_plot.legend(loc='upper right')
        return pede_plot

    def plot_cm(self, obj, fig=None):
        """Plot the common mode distribution"""
        plot = handle_sub_plots(fig, 223)
        _, bins, _ = plot.hist(obj.CMnoise, bins=50, density=True,
                               alpha=0.4, color="b", label="Common mode")
        # Calculate the mean and std
        mu, std = norm.fit(obj.CMnoise)
        # Calculate the distribution for plotting in a histogram
        p = norm.pdf(bins, loc=mu, scale=std)
        plot.plot(bins, p, "r--", color="g", label="Fit")
        plot.set_xlabel('Common mode [ADC]')
        plot.set_ylabel('[%]')
        plot.set_title(r'$\mathrm{Common\ mode\:}\ \mu=' + str(round(mu, 2)) \
                       + r',\ \sigma=' + str(round(std, 2)) + r'$')
        plot.legend(loc='upper right')
        return plot

    def plot_noise_hist(self, obj, fig=None):
        """Plot total noise distribution. Find an appropriate Gaussian while
        excluding the "ungaussian" parts of the distribution"""
        plot = handle_sub_plots(fig, 224)
        n, bins, _ = plot.hist(obj.total_noise, bins=500, density=False,
                               alpha=0.4, color="b", label="Noise")
        plot.set_yscale("log", nonposy='clip')
        plot.set_ylim(1.)

        # Cut off "ungaussian" noise
        cut = np.max(n) * 0.2  # Find maximum of hist and get the cut
        # Finds the first element which is higher as optimized threshold
        ind = np.concatenate(np.argwhere(n > cut))

        # Calculate the mean and std
        mu, std = norm.fit(bins[ind])
        # Calculate the distribution for plotting in a histogram
        plotrange = np.arange(-35, 35)
        p = gaussian(plotrange, mu, std, np.max(n))
        plot.plot(plotrange, p, "r--", color="g", label="Fit")
        plot.set_xlabel('Noise')
        plot.set_ylabel('count')
        plot.set_title("Noise Histogram")
        return plot



    ### Calibration Plots ###
    def plot_signal_conversion_fit(self, obj, fig):
        """Plots test pulses as a function of ADC singals. Shows conversion
        fit that is used to convert ADC signals to e signals."""
        # plot = fig.add_subplot(221)
        plot = handle_sub_plots(fig, 221)
        plot.set_xlabel('Mean Signal [ADC]')
        plot.set_ylabel('Test Pulse Charge [e]')
        plot.set_title('Signal Fit - ADC Signal vs. e Signal')
        plot.plot(obj.mean_sig_all_ch, obj.pulses,
                  label="Mean signal over all channels")
        plot.plot(obj.mean_sig_all_ch,
                  obj.convert_ADC_to_e(obj.mean_sig_all_ch),
                  linestyle="--", color="r", label="Conversion fit")
        plot.legend()

    def plot_signal_conversion_fit_detail(self, obj, fig, upper_lim_x=250,
                                          upper_lim_y=30000):
        """Zooms into the the important region of the signal conversion plot
        to see if the fit is sufficient there"""
        plot = handle_sub_plots(fig, 223)
        # plot = fig.add_subplot(223)
        plot.set_xlabel('Mean Signal [ADC]')
        plot.set_ylabel('Test Pulse Charge [e]')
        plot.set_title('Signal Fit Detail - ADC Signal vs. e Signal')
        plot.plot(obj.mean_sig_all_ch,
                  obj.pulses,
                  label="Mean signal over all channels")
        plot.plot(obj.mean_sig_all_ch,
                  obj.convert_ADC_to_e(obj.mean_sig_all_ch),
                  linestyle="--", color="r", label="Conversion fit")
        plot.set_xlim(right=upper_lim_x)
        plot.set_ylim(top=upper_lim_y)
        plot.legend()
        return plot

    def plot_gain_hist(self, obj, fig, cut=1.5):
        """Plot histogram of gain according to mean signal of all channels"""
        gain_hist = handle_sub_plots(fig, 222)
        gain_hist.set_ylabel('Count [#]')
        gain_hist.set_xlabel('Gain [e-]')
        gain_hist.set_title('Gain Histogram - Gain of Test Pulses')
        gain_lst, mean, ex_ratio = obj.gain_calc(cut)
        gain_hist.hist(gain_lst, range=[mean*0.75, cut*mean],
                       alpha=0.4, bins=200, color="b",
                       label="Gain of unmasked channels")
        textstr = '\n'.join(("mean = %.f" %(mean),
                             "cut = %.1f" %(cut),
                             "exclusion ratio = %.2f" %(ex_ratio),
                             "entries = %.f" %(len(gain_lst))))
        gain_hist.text(0.6, 0.85, textstr, transform=gain_hist.transAxes,
                       fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round',
                                 facecolor='white',
                                 alpha=0.5))
        gain_hist.legend()
        return gain_hist



    ### Main Plots ###
    def plot_cluster_hist(self, obj, fig=None):
        """Plots cluster size distribution of all event clusters"""
        # Plot Clustering results
        numclusters_plot = handle_sub_plots(fig, 331)

        # Plot Number of clusters
        bins, counts = np.unique(obj.outputdata["base"]["Numclus"],
                                 return_counts=True)
        numclusters_plot.bar(bins, counts, alpha=0.4, color="b")
        numclusters_plot.set_xlabel('Number of clusters [#]')
        numclusters_plot.set_ylabel('Occurance [#]')
        numclusters_plot.set_title('Number of clusters')
        # numclusters_plot.set_yscale("log", nonposy='clip')
        return numclusters_plot

    def plot_clustersizes(self, obj, fig=None):
        """Plot clustersizes"""
        clusters_plot = handle_sub_plots(fig, 332)

        bins, counts = np.unique(np.concatenate(obj.outputdata["base"]["Clustersize"]),
                                 return_counts=True)
        clusters_plot.bar(bins, counts, alpha=0.4, color="b")
        clusters_plot.set_xlabel('Clustersize [#]')
        clusters_plot.set_ylabel('Occurance [#]')
        clusters_plot.set_title('Clustersizes')
        # clusters_plot.set_yscale("log", nonposy='clip')

        # fig.tight_layout()
        # fig.subplots_adjust(top=0.88)
        return clusters_plot

    def plot_hitmap(self, obj, fig=None):
        """Plots the hitmap of the measurement."""
        hitmap_plot = handle_sub_plots(fig, 333)
        hitmap_plot.set_title("Event Hitmap")
        hitmap_plot.bar(np.arange(obj.numchan),
                        obj.outputdata["base"]["Hitmap"][len(obj.outputdata["base"]["Hitmap"]) - 1],
                        1., alpha=0.4, color="b")
        hitmap_plot.set_xlabel('channel [#]')
        hitmap_plot.set_ylabel('Hits [#]')
        if fig is None:
            hitmap_plot.set_title('Hitmap')
        return hitmap_plot

    def plot_single_event_ch(self, obj, fig=None, eventnum=1000):
        """ Plots a single event and its data"""
        # fig = plt.figure("Event number {!s}, from file: {!s}".format(eventnum, file))
        channel_plot = handle_sub_plots(fig, 334)
        channel_plot.bar(np.arange(obj.numchan),
                         obj.outputdata["base"]["Signal"][eventnum], 1.,
                         alpha=0.4, color="b")
        channel_plot.set_xlabel('channel [#]')
        channel_plot.set_ylabel('Signal [ADC]')
        channel_plot.set_title('Signal of event #%d' %eventnum)
        return channel_plot

    def plot_single_event_SN(self, obj, fig=None, eventnum=1000):
        """Plot signal/Noise"""
        SN_plot = handle_sub_plots(fig, 335)
        SN_plot.bar(np.arange(obj.numchan),
                    obj.outputdata["base"]["SN"][eventnum], 1.,
                    alpha=0.4, color="b")
        SN_plot.set_xlabel('channel [#]')
        SN_plot.set_ylabel('Signal/Noise [ADC]')
        SN_plot.set_title('Signal/Noise of event #%d' %eventnum)
        # fig.subplots_adjust(top=0.88)
        return SN_plot

    def plot_langau_per_clustersize(self, obj, fig=None, fit_langau=True):
        """Plots the data calculated so the energy data and the langau"""
        data = obj.outputdata["Langau"]
        # fig = plt.figure("Langau from file: {!s}".format(file))
        # Plot delay
        plot = handle_sub_plots(fig, 335)
        plot.set_title("Signals of different cluster sizes")
        # hist, edges = np.histogram(data["signal"], bins=obj.bins)
        plot.hist(data["signal"], bins=data["bins"], density=False,
                  alpha=0.4, color="b", label="All clusters")
        #plot.errorbar(edges[:-1], hist, xerr=data["data_error"], fmt='o', markersize=1, color="red")
        if fit_langau:
            plot.plot(data["langau_data"][0], data["langau_data"][1], "r--",
                      color="g")
                      # label="Langau: \n mpv: {mpv!s} \n eta: {eta!s} \n sigma: {sigma!s} \n A: {A!s} \n".format(
                      #     mpv=data["langau_coeff"][0],
                      #     eta=data["langau_coeff"][1],
                      #     sigma=data["langau_coeff"][2],
                      #     A=data["langau_coeff"][3]))
            # plot.errorbar(data["langau_data"][0], data["langau_data"][1],
            #               np.sqrt(pylandau.langau(data["langau_data"][0], *data["langau_coeff"])),
            #               fmt=".", color="r", label="Error of Fit")

        plot.set_xlabel('Cluster Signal [e]')
        plot.set_ylabel('Events [#]')
        # plot.set_title('All clusters Langau from file: {!s}'.format(file))

        # Plot the different clustersizes as well into the langau plot
        colour = ['green', 'red', 'orange', 'cyan', 'black', 'pink', 'magenta']
        for i, cls in enumerate(data["Clustersize"]):
            if i < 7:
                plot.hist(cls["signal"], bins=data["bins"], density=False,
                          alpha=0.3, color=colour[i],
                          label="Clustersize: {!s}".format(i + 1))
            else:
                # self.log.warning(
                #     "To many histograms for this plot. "
                #     "Colorscheme only supports seven different histograms. Extend if need be!")
                continue

        plot.legend()
        return plot

    def plot_seed_signal_e(self, obj, fig=None, seed_cut=True, fit_langau=True):
        """Plots seed signal and langau distribution"""
        if seed_cut:
            # fig = plt.figure("Seed cut langau from file: {!s}".format(file))
            # Plot Seed cut langau
            plot = handle_sub_plots(fig, 336)
            # indizes = np.nonzero(data["signal_SC"] > 0)[0]
            data = obj.outputdata["Langau"]
            plot.hist(data["signal_SC"],
                      bins=data["bins"], density=False,
                      alpha=0.4, color="b", label="Signals")
            if fit_langau:
                plot.plot(data["langau_data_SC"][0], data["langau_data_SC"][1],
                          "r--", color="g",
                          label="Fit")
                textstr = '\n'.join((
                    "mpv = %.f" %(data["langau_coeff_SC"][0]),
                    "eta = %.2f" %(data["langau_coeff_SC"][1]),
                    "sigma = %.2f" %(data["langau_coeff_SC"][2]),
                    "A = %.2f" %(data["langau_coeff_SC"][3])))
                                     # "entries = %.f" %(len(gain_lst))))
                plot.text(0.8, 0.8, textstr, transform=plot.transAxes,
                          fontsize=10,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round',
                                    facecolor='white',
                                    alpha=0.5))
                # plot.errorbar(data["langau_data_SC"][0], data["langau_data_SC"][1],
                #               np.sqrt(pylandau.langau(data["langau_data_SC"][0], *data["langau_coeff_SC"])),
                #               fmt=".", color="r", label="Error of Fit")
            plot.set_xlabel('Seed Signal [e]')
            plot.set_ylabel('Events [#]')
            if fig is None:
                plot.set_title('Seed Signal in e')
            plot.legend()
            return plot
