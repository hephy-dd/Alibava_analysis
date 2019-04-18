"""PlotData Class"""
# pylint: disable=R0201,C0103,E0401
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import logging
# import pylandau
from analysis_classes.utilities import handle_sub_plots, gaussian
from analysis_classes.utilities import create_dictionary

class PlotData:
    """Plots for ALiBaVa Analysis"""
    def __init__(self):
        # canvas for plotting the data [width, height (inches)]

        self.log = logging.getLogger(__name__)
        self.cfg = create_dictionary("plot_cfg.yml")
        #self.groups["Brace yourself! plots are comming"] = np.concatenate([x for x in self.groups.items()])

    def plot_data(self, mcfg, obj, group=None, fig_name=None):
        """Plots the data calculated by the framework. Suppress drawing and
        showing the canvas by setting "show" to False.
        Returns matplotlib.pyplot.figure object.
        """
        figures = []
        if group=="all" or group=="from_file":
            for grp in self.cfg["Render"]:
                fig_name = grp
                fig = plt.figure(fig_name, figsize=[10, 8])
                for funcname, cfg in zip(self.cfg["Render"][grp]["Plots"], self.cfg["Render"][grp]["arrangement"]):
                    try:
                        plot_func = getattr(self, str(funcname))
                        plot_func(cfg, obj, fig)
                    except Exception as err:
                        self.log.error("Plotting function {} raised an error. Error: {}".format(funcname, err))
                fig.subplots_adjust(hspace = 0.3) # Adjusts the padding so nothing is overlapping

    def show_plots(self):
        """Draw and show plots"""
        plt.draw()
        plt.show()

    ### Pedestal Plots ###
    def plot_noise_ch(self, cfg, obj, fig=None):
        """plot noise per channel"""
        data = obj["NoiseAnalysis"]

        noise_plot = handle_sub_plots(fig, cfg)
        noise_plot.bar(np.arange(data.numchan), data.noise, 1.,
                       alpha=0.4, color="b", label="Noise level per strip")
        # plot line idicating masked and unmasked channels
        valid_strips = np.zeros(data.numchan)
        valid_strips[data.noisy_strips] = 1
        noise_plot.plot(np.arange(data.numchan), valid_strips, color="r",
                        label="Masked strips")

        # Plot the threshold for deciding a good channel
        xval = [0, data.numchan]
        yval = [data.median_noise + data.noise_cut,
                data.median_noise + data.noise_cut]
        noise_plot.plot(xval, yval, "r--", color="g",
                        label="Threshold for noisy strips")

        noise_plot.set_xlabel('Channel [#]')
        noise_plot.set_ylabel('Noise [ADC]')
        noise_plot.set_title('Noise levels per Channel')
        noise_plot.legend(loc='upper right')
        return noise_plot

    def plot_pedestal(self, cfg, obj, fig=None):
        """Plot pedestal and noise per channel"""
        data = obj["NoiseAnalysis"]
        pede_plot = handle_sub_plots(fig, cfg)
        pede_plot.bar(np.arange(data.numchan), data.pedestal, 1., yerr=data.noise,
                      error_kw=dict(elinewidth=0.2, ecolor='r', ealpha=0.1),
                      alpha=0.4, color="b", label="Pedestal")
        pede_plot.set_xlabel('Channel [#]')
        pede_plot.set_ylabel('Pedestal [ADC]')
        pede_plot.set_title('Pedestal levels per Channel with noise')
        pede_plot.set_ylim(bottom=min(data.pedestal) - 50.)
        pede_plot.legend(loc='upper right')
        return pede_plot

    def plot_cm(self, cfg, obj, fig=None):
        """Plot the common mode distribution"""
        data = obj["NoiseAnalysis"]
        plot = handle_sub_plots(fig, cfg)
        _, bins, _ = plot.hist(data.CMnoise, bins=50, density=True,
                               alpha=0.4, color="b", label="Common mode")
        # Calculate the mean and std
        mu, std = norm.fit(data.CMnoise)
        # Calculate the distribution for plotting in a histogram
        p = norm.pdf(bins, loc=mu, scale=std)
        plot.plot(bins, p, "r--", color="g", label="Fit")
        plot.set_xlabel('Common mode [ADC]')
        plot.set_ylabel('[%]')
        plot.set_title(r'$\mathrm{Common\ mode\:}\ \mu=' + str(round(mu, 2)) \
                       + r',\ \sigma=' + str(round(std, 2)) + r'$')
        plot.legend(loc='upper right')
        return plot

    def plot_noise_hist(self,cfg, obj, fig=None):
        """Plot total noise distribution. Find an appropriate Gaussian while
        excluding the "ungaussian" parts of the distribution"""
        data = obj["NoiseAnalysis"]
        plot = handle_sub_plots(fig, cfg)
        n, bins, _ = plot.hist(data.total_noise, bins=500, density=False,
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
    def plot_signal_conversion_fit(self, cfg, obj, fig):
        """Plots test pulses as a function of ADC singals. Shows conversion
        fit that is used to convert ADC signals to e signals."""
        # plot = fig.add_subplot(221)
        data = obj["Calibration"]
        plot = handle_sub_plots(fig, cfg)
        plot.set_xlabel('Mean Signal [ADC]')
        plot.set_ylabel('Test Pulse Charge [e]')
        plot.set_title('Signal Fit - ADC Signal vs. e Signal')
        plot.plot(data.mean_sig_all_ch, data.pulses,
                  label="Mean signal over all channels")
        plot.plot(data.mean_sig_all_ch,
                  data.convert_ADC_to_e(data.mean_sig_all_ch),
                  linestyle="--", color="r", label="Conversion fit")
        plot.legend()

    def plot_signal_conversion_fit_detail(self, cfg, obj, fig):
        """Zooms into the the important region of the signal conversion plot
        to see if the fit is sufficient there"""
        data = obj["Calibration"]
        upper_lim_x = self.cfg["Upper_limits_conversion"]["ADC_Signal"]
        upper_lim_y = self.cfg["Upper_limits_conversion"]["e_Signal"]
        plot = handle_sub_plots(fig, cfg)
        # plot = fig.add_subplot(223)
        plot.set_xlabel('Mean Signal [ADC]')
        plot.set_ylabel('Test Pulse Charge [e]')
        plot.set_title('Signal Fit Detail - ADC Signal vs. e Signal')
        plot.plot(data.mean_sig_all_ch,
                  data.pulses,
                  label="Mean signal over all channels")
        plot.plot(data.mean_sig_all_ch,
                  data.convert_ADC_to_e(data.mean_sig_all_ch),
                  linestyle="--", color="r", label="Conversion fit")
        plot.set_xlim(right=upper_lim_x)
        plot.set_ylim(top=upper_lim_y)
        plot.legend()
        return plot

    def plot_gain_hist(self, cfg, obj, fig):
        """Plot histogram of gain according to mean signal of all channels"""
        data = obj["Calibration"]
        cut = self.cfg["Gain_cut"]
        gain_hist = handle_sub_plots(fig, cfg)
        gain_hist.set_ylabel('Count [#]')
        gain_hist.set_xlabel('Gain [e-]')
        gain_hist.set_title('Gain Histogram - Gain of Test Pulses')
        gain_lst, mean, ex_ratio = data.gain_calc(cut)
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
    def plot_cluster_hist(self, cfg, obj, fig=None):
        """Plots cluster size distribution of all event clusters"""
        # Plot Clustering results
        data = obj["MainAnalysis"]
        numclusters_plot = handle_sub_plots(fig, cfg)

        # Plot Number of clusters
        bins, counts = np.unique(data.outputdata["base"]["Numclus"],
                                 return_counts=True)
        numclusters_plot.bar(bins, counts, alpha=0.4, color="b")
        numclusters_plot.set_xlabel('Number of clusters [#]')
        numclusters_plot.set_ylabel('Occurance [#]')
        numclusters_plot.set_title('Number of clusters')
        # numclusters_plot.set_yscale("log", nonposy='clip')
        return numclusters_plot

    def plot_clustersizes(self, cfg, obj, fig=None):
        """Plot clustersizes"""
        data = obj["MainAnalysis"]
        clusters_plot = handle_sub_plots(fig, cfg)

        bins, counts = np.unique(np.concatenate(data.outputdata["base"]["Clustersize"]),
                                 return_counts=True)
        clusters_plot.bar(bins, counts, alpha=0.4, color="b")
        clusters_plot.set_xlabel('Clustersize [#]')
        clusters_plot.set_ylabel('Occurance [#]')
        clusters_plot.set_title('Clustersizes')
        # clusters_plot.set_yscale("log", nonposy='clip')

        # fig.tight_layout()
        # fig.subplots_adjust(top=0.88)
        return clusters_plot

    def plot_hitmap(self, cfg, obj, fig=None):
        """Plots the hitmap of the measurement."""
        data = obj["MainAnalysis"]
        hitmap_plot = handle_sub_plots(fig, cfg)
        hitmap_plot.set_title("Event Hitmap")
        hitmap_plot.bar(np.arange(data.numchan),
                        data.outputdata["base"]["Hitmap"][len(data.outputdata["base"]["Hitmap"]) - 1],
                        1., alpha=0.4, color="b")
        hitmap_plot.set_xlabel('channel [#]')
        hitmap_plot.set_ylabel('Hits [#]')
        if fig is None:
            hitmap_plot.set_title('Hitmap')
        return hitmap_plot

    def plot_single_event_ch(self, cfg, obj, fig=None):
        """ Plots a single event and its data"""
        # fig = plt.figure("Event number {!s}, from file: {!s}".format(eventnum, file))
        data = obj["MainAnalysis"]
        eventnum = self.cfg["Plot_single_event"]
        channel_plot = handle_sub_plots(fig, cfg)
        channel_plot.bar(np.arange(data.numchan),
                         data.outputdata["base"]["Signal"][eventnum], 1.,
                         alpha=0.4, color="b")
        channel_plot.set_xlabel('channel [#]')
        channel_plot.set_ylabel('Signal [ADC]')
        channel_plot.set_title('Signal of event #%d' %eventnum)
        return channel_plot

    def plot_single_event_SN(self, cfg, obj, fig=None):
        """Plot signal/Noise"""
        data = obj["MainAnalysis"]
        eventnum = self.cfg["Plot_single_event"]
        SN_plot = handle_sub_plots(fig, cfg)
        SN_plot.bar(np.arange(data.numchan),
                    data.outputdata["base"]["SN"][eventnum], 1.,
                    alpha=0.4, color="b")
        SN_plot.set_xlabel('channel [#]')
        SN_plot.set_ylabel('Signal/Noise [ADC]')
        SN_plot.set_title('Signal/Noise of event #%d' %eventnum)
        # fig.subplots_adjust(top=0.88)
        return SN_plot

    def plot_langau_per_clustersize(self, cfg, obj, fig=None):
        """Plots the data calculated so the energy data and the langau"""
        data = obj["MainAnalysis"]
        fit_langau = self.cfg["Fit_langau"]
        data = data.outputdata["Langau"]
        # fig = plt.figure("Langau from file: {!s}".format(file))
        # Plot delay
        plot = handle_sub_plots(fig, cfg)
        plot.set_title("Signals of different cluster sizes")
        # hist, edges = np.histogram(data["signal"], bins=data.bins)
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

    def plot_seed_signal_e(self, cfg, obj, fig=None):
        """Plots seed signal and langau distribution"""
        data = obj["MainAnalysis"]
        seed_cut = self.cfg["Plot_seed_cut"]
        fit_langau = self.cfg["Fit_langau"]
        if seed_cut:
            # fig = plt.figure("Seed cut langau from file: {!s}".format(file))
            # Plot Seed cut langau
            plot = handle_sub_plots(fig, cfg)
            # indizes = np.nonzero(data["signal_SC"] > 0)[0]
            data = data.outputdata["Langau"]
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

    def plot_timing_profile(self, cfg, obj, fig=None):
        # Plot timing profile
        data = obj["MainAnalysis"]
        timing_plot = handle_sub_plots(fig, cfg)
        timing_plot.set_xlabel('timing [ns]')
        timing_plot.set_ylabel('average signal [ADC]')
        timing_plot.set_title('Average timing signal of seed hits')
        signal = data.prodata[:, 0]
        channels_hit = data.prodata[:, 5]
        sum_singal = np.zeros(len(signal))
        for i, sig, chan in zip(np.arange(len(signal)), signal, channels_hit):
            sum_singal[i] = np.sum(sig[chan])
        timing_data = np.zeros(150)
        # var_timing_data = np.zeros(150)
        for timing in range(1, 151):  # Timing of ALiBaVa
            timing_in = np.nonzero(np.logical_and(data.timing >= timing - 1, data.timing < timing))
            timing_data[timing - 1] = np.median(sum_singal[timing_in[0]])
            # var_timing_data[timing-1] = np.std(sum_singal[timing_in[0]])

        timing_plot.bar(np.arange(0, 150), timing_data, alpha=0.4, color="b")  # , yerr=var_timing_data)

    def plot_histogram_of_timing(self, cfg, obj, fig=None):
        # Plot histogram of timing
        data = obj["MainAnalysis"]
        timing_hist_plot = handle_sub_plots(fig, cfg)
        timing_hist_plot.set_xlabel('timing [ns]')
        timing_hist_plot.set_ylabel('count [#]')
        timing_hist_plot.set_title('Histogram of timings')

        timing_hist_plot.hist(data.timing, 150, alpha=0.4, color="b")
