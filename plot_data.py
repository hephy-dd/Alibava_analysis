"""PlotData Class"""
# pylint: disable=R0201,C0103,E0401,R0913
import pdb
import numpy as np
from scipy.stats import norm, rv_continuous
import matplotlib.pyplot as plt
import logging
import matplotlib.mlab as mlab
from scipy import optimize

# import pylandau
from analysis_classes.utilities import handle_sub_plots, gaussian, save_all_plots
from analysis_classes.utilities import create_dictionary


class PlotData:
    """Plots for ALiBaVa Analysis"""

    def __init__(self, config_path, logger=None):
        self.log = logger or logging.getLogger(__class__.__name__)
        # canvas for plotting the data [width, height (inches)]

        self.log = logging.getLogger(__name__)
        self.cfg = create_dictionary(config_path)
        # self.groups["Brace yourself! plots are comming"] = np.concatenate([x for x in self.groups.items()])

    def start_plotting(self, mcfg, obj, group=None, fig_name=None):
        """Plots the data calculated by the framework. Suppress drawing and
        showing the canvas by setting "show" to False.
        Returns matplotlib.pyplot.figure object.
        """

        if group == "all" or group == "from_file":
            for grp in self.cfg["Render"]:
                fig_name = grp
                self.log.info("Plotting group: {}".format(grp))
                fig = plt.figure(fig_name, figsize=[10, 8])
                for funcname, cfg in zip(
                    self.cfg["Render"][grp]["Plots"],
                    self.cfg["Render"][grp]["arrangement"],
                ):
                    try:
                        plot_func = getattr(self, str(funcname))
                        plot_func(cfg, obj, fig)
                    except Exception as err:
                        self.log.error(
                            "Plotting function {} raised an error. Error: {}".format(
                                funcname, err
                            )
                        )
                fig.subplots_adjust(
                    hspace=0.4
                )  # Adjusts the padding so nothing is overlapping
                fig.subplots_adjust(
                    wspace=0.3
                )  # Adjusts the padding so nothing is overlapping

    def show_plots(self):
        """Draw and show plots"""
        plt.draw()
        plt.show()

    ### Pedestal Noise Plots ###
    def plot_MaskedChannelNoise_ch(self, cfg, obj, fig=None):
        """plot noise per channel with commom mode correction and the masked strips."""
        data = obj["NoiseAnalysis"]

        noise_plot = handle_sub_plots(fig, cfg)
        noise_plot.bar(
            np.arange(data.numchan),
            data.noise,
            1.0,
            alpha=0.4,
            color="b",
            label="Noise level per strip (CMC)",
        )
        # plot line idicating masked and unmasked channels
        # valid_strips = np.zeros(data.numchan)
        # valid_strips[data.noisy_strips] = 1
        # noise_plot.plot(np.arange(data.numchan), valid_strips, color="r",
        #                label="Masked strips")

        # Plot the threshold for deciding a good channel
        # xval = [0, data.numchan]
        # yval = [data.median_noise + data.noise_cut,
        #        data.median_noise + data.noise_cut]
        # noise_plot.plot(xval, yval, "r--", color="g",
        #                label="Threshold for noisy strips")

        noise_plot.set_xlabel("Channel [#]")
        noise_plot.set_ylabel("Noise [ADC]")
        noise_plot.set_title("Noise levels per Channel (CMC)")
        noise_plot.legend(loc="upper right")
        # noise_plot.set_ylim(0,10)
        return noise_plot

    def plot_rawnoise_ch(self, cfg, obj, fig=None):
        """plot noise per channel with commom mode correction"""
        data = obj["NoiseAnalysis"]

        noise_plot = handle_sub_plots(fig, cfg)
        noise_plot.bar(
            np.arange(data.numchan),
            data.noise_raw,
            1.0,
            alpha=0.4,
            color="b",
            label="Noise level per strip (CMC)",
        )
        # plot line idicating masked and unmasked channels
        valid_strips = np.zeros(data.numchan)
        valid_strips[data.noisy_strips] = 1
        noise_plot.plot(
            np.arange(data.numchan), valid_strips, color="r", label="Masked strips"
        )

        # Plot the threshold for deciding a good channel
        xval = [0, data.numchan]
        yval = [data.median_noise + data.noise_cut, data.median_noise + data.noise_cut]
        noise_plot.plot(
            xval, yval, "r--", color="g", label="Threshold for noisy strips"
        )

        noise_plot.set_xlabel("Channel [#]")
        noise_plot.set_ylabel("Noise [ADC]")
        noise_plot.set_title("Raw Noise levels per Channel (CMC)")
        noise_plot.legend(loc="upper right")
        # noise_plot.set_ylim(0,10)
        return noise_plot

    def plot_pedestal(self, cfg, obj, fig=None):
        """Plot pedestal and noise per channel"""
        data = obj["NoiseAnalysis"]
        noise_plot = handle_sub_plots(fig, cfg)
        noise_plot.bar(
            np.arange(data.numchan),
            data.pedestal,
            1.0,
            yerr=data.noise,
            error_kw=dict(elinewidth=0.2, ecolor="r", ealpha=0.1),
            alpha=0.4,
            color="b",
            label="Pedestal",
        )
        noise_plot.set_xlabel("Channel [#]")
        noise_plot.set_ylabel("Pedestal [ADC]")
        noise_plot.set_title("Pedestal levels per Channel with noise (only non-masked)")
        noise_plot.set_ylim(bottom=min(data.pedestal) - 50.0)
        noise_plot.legend(loc="upper right")
        # noise_plot.set_ylim(0, 10)
        return noise_plot

    def plot_noiseNonCMCorr_ch(self, cfg, obj, fig=None):
        """plot noise per channel"""
        data = obj["NoiseAnalysis"]

        noise_plot = handle_sub_plots(fig, cfg)
        noise_plot.bar(
            np.arange(data.numchan),
            data.noiseNCM,
            1.0,
            alpha=0.4,
            color="b",
            label="Noise level per strip",
        )
        # plot line idicating masked and unmasked channels
        # valid_strips = np.zeros(data.numchan)
        # valid_strips[data.noisy_strips] = 1
        # noise_plot.plot(np.arange(data.numchan), valid_strips, color="r",
        #                label="Masked strips")

        # Plot the threshold for deciding a good channel
        noise_plot.set_xlabel("Channel [#]")
        noise_plot.set_ylabel("Noise with common-mode [ADC]")
        noise_plot.set_title("Noise level per Channel non-common-mode corrected")
        noise_plot.legend(loc="upper right")
        # noise_plot.set_ylim(0,10)
        return noise_plot

    def plot_rawnoiseNonCMCorr_ch(self, cfg, obj, fig=None):
        """plot noise per channel"""
        data = obj["NoiseAnalysis"]

        noise_plot = handle_sub_plots(fig, cfg)
        noise_plot.bar(
            np.arange(data.numchan),
            data.noiseNCM_raw,
            1.0,
            alpha=0.4,
            color="b",
            label="Noice level per strip",
        )
        # plot line idicating masked and unmasked channels
        valid_strips = np.zeros(data.numchan)
        valid_strips[data.noisy_strips] = 1
        noise_plot.plot(
            np.arange(data.numchan), valid_strips, color="r", label="Masked strips"
        )

        # Plot the threshold for deciding a good channel
        noise_plot.set_xlabel("Channel [#]")
        noise_plot.set_ylabel("Common mode [ADC]")
        noise_plot.set_title("Noise level per Channel")
        noise_plot.legend(loc="upper right")
        return noise_plot

    def plot_cm(self, cfg, obj, fig=None):
        """Plot the common mode distribution"""
        data = obj["NoiseAnalysis"]
        plot = handle_sub_plots(fig, cfg)
        _, bins, _ = plot.hist(
            data.CMnoise,
            bins=50,
            density=True,
            alpha=0.4,
            color="b",
            label="Common mode",
        )
        # Calculate the mean and std
        mu, std = norm.fit(data.CMnoise)
        # Calculate the distribution for plotting in a histogram
        p = norm.pdf(bins, loc=mu, scale=std)
        plot.plot(bins, p, "r--", color="g", label="Fit")
        plot.set_xlabel("Common mode [ADC]")
        plot.set_ylabel("[%]")
        plot.set_title(
            r"$\mathrm{Common\ mode\:}\ \mu="
            + str(round(mu, 2))
            + r",\ \sigma="
            + str(round(std, 2))
            + r"$"
        )
        plot.legend(loc="upper right")
        return plot

    def plot_noise_hist(self, cfg, obj, fig=None):
        """Plot total noise distribution. Find an appropriate Gaussian while
        excluding the "ungaussian" parts of the distribution"""
        data = obj["NoiseAnalysis"]
        plot = handle_sub_plots(fig, cfg)
        count, bins, _ = plot.hist(
            data.total_noise,
            bins=300,
            density=False,
            alpha=0.4,
            color="b",
            label="Noise",
        )

        bin_centers = bins[:-1] + np.diff(bins) / 2  # define bin centers from bin edges

        plot.set_yscale("log", nonposy="clip")
        plot.set_ylim(1.0)

        # Cut off "ungaussian" noise for fit
        cut = np.max(count) * 0.2  # Find maximum of hist and get the cut
        # Finds indices of bins with content above threshold
        ind = np.concatenate(np.argwhere(count > cut))
        # estimators of the data
        n = sum(count[ind])
        mean_est = sum(bin_centers[ind] * count[ind]) / n
        sigma_est = np.sqrt(sum(count[ind] * (bin_centers[ind] - mean_est) ** 2) / n)
        bin_width = (bins[ind][-1] - bins[ind][0]) / len(bin_centers[ind])
        norm_est = sum(count[ind] * bin_width) / (
            sigma_est * np.sqrt(2 * np.pi)
        )  # Area / sigma*sqrt(2pi)

        # Fit the mean, std and norm
        popt, pcov = optimize.curve_fit(
            gaussian, bin_centers[ind], count[ind], p0=[mean_est, sigma_est, norm_est]
        )

        # Calculate the distribution for plotting in a histogram
        plotrange = np.arange(-300, 300)
        p = gaussian(plotrange, popt[0], popt[1], popt[2])
        plot.plot(plotrange, p, "r--", color="g", label="Fit")
        plot.set_xlabel("Noise")
        plot.set_ylabel("count")
        plot.set_title("Noise Histogram")
        return plot

    ### Calibration Plots ###
    def plot_signal_conversion_fit(self, cfg, obj, fig):
        """Plots test pulses as a function of ADC singals. Shows conversion
        fit that is used to convert ADC signals to e signals."""
        data = obj["Calibration"]
        plot = handle_sub_plots(fig, cfg)
        plot.set_xlabel("Mean Signal [ADC]")
        plot.set_ylabel("Test Pulse Charge [e]")
        plot.set_title("Signal Fit - ADC Signal vs. e Signal")
        plot.plot(
            data.mean_sig_all_ch, data.pulses, label="Mean signal over all channels"
        )
        # plot.plot(data.mean_sig_all_ch,
        #          data.convert_ADC_to_e(data.mean_sig_all_ch, use_mean=True),
        #          linestyle="--", color="r", label="Conversion fit")
        plot.errorbar(
            data.mean_sig_all_ch,
            data.convert_ADC_to_e(data.mean_sig_all_ch, use_mean=True),
            yerr=data.convert_ADC_to_e(data.mean_std_all_ch, use_mean=True),
            markersize=1,
            color="red",
            label="Conversion fit",
        )
        plot.legend()

    def plot_signal_conversion_fit_single(self, cfg, obj, fig):
        """Plot all individual fits here"""

    def plot_signal_conversion_fit_detail(self, cfg, obj, fig):
        """Zooms into the the important region of the signal conversion plot
        to see if the fit is sufficient there"""
        data = obj["Calibration"]
        upper_lim_x = self.cfg["Upper_limits_conversion"]["ADC_Signal"]
        upper_lim_y = self.cfg["Upper_limits_conversion"]["e_Signal"]
        plot = handle_sub_plots(fig, cfg)
        # plot = fig.add_subplot(223)
        plot.set_xlabel("Mean Signal [ADC]")
        plot.set_ylabel("Test Pulse Charge [e]")
        plot.set_title("Signal Fit Detail - ADC Signal vs. e Signal")
        plot.plot(
            data.mean_sig_all_ch, data.pulses, label="Mean signal over all channels"
        )
        # plot.plot(data.mean_sig_all_ch,
        #          data.convert_ADC_to_e(data.mean_sig_all_ch, use_mean=True),
        #          linestyle="--", color="r", label="Conversion fit")

        plot.errorbar(
            data.mean_sig_all_ch,
            data.convert_ADC_to_e(data.mean_sig_all_ch, use_mean=True),
            yerr=data.convert_ADC_to_e(data.mean_std_all_ch, use_mean=True),
            markersize=1,
            color="red",
            label="Conversion fit",
        )
        plot.set_xlim(right=upper_lim_x)
        plot.set_ylim(top=upper_lim_y)
        plot.legend()
        return plot

    def plot_gain_hist(self, cfg, obj, fig):
        """Plot histogram of gain according to mean signal of all channels"""
        data = obj["Calibration"]
        cut = self.cfg["Gain_cut"]
        gain_hist = handle_sub_plots(fig, cfg)
        gain_hist.set_ylabel("Count [#]")
        gain_hist.set_xlabel("Gain [e-]")
        gain_hist.set_title("Gain Histogram - Gain of Test Pulses")
        gain_lst, mean, ex_ratio = data.gain_calc(cut)
        gain_hist.hist(
            gain_lst,
            range=[mean * 0.75, cut * mean],
            alpha=0.4,
            bins=200,
            color="b",
            label="Gain of unmasked channels",
        )
        textstr = "\n".join(
            (
                "mean = %.f" % (mean),
                "cut = %.1f" % (cut),
                "exclusion ratio = %.2f" % (ex_ratio),
                "entries = %.f" % (len(gain_lst)),
            )
        )
        gain_hist.text(
            0.6,
            0.85,
            textstr,
            transform=gain_hist.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )
        gain_hist.legend()
        return gain_hist

    ### Main Plots ###
    def plot_cluster_hist(self, cfg, obj, fig=None):
        """Plots cluster size distribution of all event clusters"""
        # Plot Clustering results
        data = obj["MainAnalysis"]["base"]
        numclusters_plot = handle_sub_plots(fig, cfg)

        # Plot Number of clusters
        bins, counts = np.unique(data["Numclus"], return_counts=True)
        numclusters_plot.bar(bins, counts, alpha=0.4, color="b")
        numclusters_plot.set_xlabel("Number of clusters [#]")
        numclusters_plot.set_ylabel("Occurance [#]")
        numclusters_plot.set_title("Number of clusters")
        # numclusters_plot.set_yscale("log", nonposy='clip')
        return numclusters_plot

    def plot_clustersizes(self, cfg, obj, fig=None):
        """Plot clustersizes"""
        data = obj["MainAnalysis"]["base"]
        clusters_plot = handle_sub_plots(fig, cfg)

        bins, counts = np.unique(
            np.concatenate(data["Clustersize"]), return_counts=True
        )
        clusters_plot.bar(bins, counts, alpha=0.4, color="b")
        clusters_plot.set_xlabel("Clustersize [#]")
        clusters_plot.set_ylabel("Occurance [#]")
        clusters_plot.set_title("Clustersizes")

        textstr = ""
        for i, bin in enumerate(counts):
            textstr += "Size {}: {}\n".format(i, bin)

        clusters_plot.text(
            0.8,
            0.85,
            textstr.strip(),
            transform=clusters_plot.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

        # clusters_plot.set_yscale("log", nonposy='clip')
        # fig.tight_layout()
        # fig.subplots_adjust(top=0.88)
        return clusters_plot

    def plot_hitmap_per_clustersize(self, cfg, obj, fig=None):
        """Plots the hitmap per clustersize"""
        data = obj["MainAnalysis"]["base"]
        hit_plot = handle_sub_plots(fig, cfg)
        hit_plot.set_title("Hitmap per clustersize")
        hit_plot.set_xlabel("channel [#]")
        hit_plot.set_ylabel("Hits [#]")
        # Plot the different clustersizes
        colour = ["green", "red", "orange", "cyan", "black", "pink", "magenta"]

        # Get only events with one cluster inside
        ind_only_one_cluster = np.nonzero(data["Numclus"] == 1)
        clusters_raw = np.take(data["Clustersize"], ind_only_one_cluster)
        clusters_flattend = np.concatenate(clusters_raw[0]).ravel()
        max_cluster = self.cfg["hitmap_max_clustersize"]

        for clus in range(1, max_cluster + 1):
            # Get clusters
            indizes_clustersize = np.nonzero(clusters_flattend == clus)
            indizes = np.take(ind_only_one_cluster, indizes_clustersize)[0]
            hitted = np.take(data["Clusters"], indizes)
            hitted_flatten = np.concatenate(hitted).ravel()

            hit_plot.hist(
                hitted_flatten,
                range=(0, 256),
                bins=256,
                alpha=0.3,
                color=colour[clus - 1],
                label="Clustersize: {!s}".format(clus),
            )

        hit_plot.legend()
        return hit_plot

    def plot_hitmap(self, cfg, obj, fig=None):
        """Plots the hitmap of the measurement."""
        # Todo: plot hitmap per clustersize

        data = obj["MainAnalysis"]["base"]
        hitmap_plot = handle_sub_plots(fig, cfg)
        hitmap_plot.set_title("Event Hitmap")
        hitmap_plot.bar(
            np.arange(len(data["Hitmap"][0])),
            data["Hitmap"][len(data["Hitmap"]) - 1],
            1.0,
            alpha=0.4,
            color="b",
        )

        def gaussian(x, a, mu, sig):
            return a * np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))

        # some channels signal/noise is to high, therefore they are dumped and set to zero.
        # For the gauss fit these dumped values interfere with the mu and std so that no actual fit can be made.
        # This algorithm takes care of that.
        gaussian_data = []
        i = 0
        for value in data["Hitmap"][len(data["Hitmap"]) - 1]:
            if float(value) > 0:
                gaussian_data.append(value)
            else:
                if i > 0:
                    gaussian_data.append(gaussian_data[i - 1])
                else:
                    gaussian_data.append(0.0)
            i += 1
        # print(gaussian_data, len(gaussian_data))
        params, params_covariance = optimize.curve_fit(
            gaussian, np.arange(len(gaussian_data)), gaussian_data
        )
        # print(params)
        plt.plot(
            np.arange(len(data["Hitmap"][0])),
            gaussian(np.arange(len(data["Hitmap"][0])), *params),
            "--",
            color="g",
            label="Fit",
        )
        hitmap_plot.set_xlabel("channel [#]")
        hitmap_plot.set_ylabel("Hits [#]")
        plt.legend(loc="best")

        if fig is None:
            hitmap_plot.set_title("Hitmap")
        return hitmap_plot

    def plot_single_event_ch(self, cfg, obj, fig=None):
        """Plots a single event and its data"""
        # fig = plt.figure("Event number {!s}, from file: {!s}".format(eventnum, file))
        data = obj["MainAnalysis"]["base"]
        eventnum = self.cfg["Plot_single_event"]
        channel_plot = handle_sub_plots(fig, cfg)
        channel_plot.bar(
            np.arange(len(data["Signal"][0])),
            data["Signal"][eventnum],
            1.0,
            alpha=0.4,
            color="b",
        )
        channel_plot.set_xlabel("channel [#]")
        channel_plot.set_ylabel("Signal [ADC]")
        channel_plot.set_title("Signal of event #%d" % eventnum)
        return channel_plot

    def plot_single_event_SN(self, cfg, obj, fig=None):
        """Plot signal/Noise"""
        data = obj["MainAnalysis"]["base"]
        eventnum = self.cfg["Plot_single_event"]
        SN_plot = handle_sub_plots(fig, cfg)
        SN_plot.bar(
            np.arange(len(data["Signal"][0])),
            data["SN"][eventnum],
            1.0,
            alpha=0.4,
            color="b",
        )
        SN_plot.set_xlabel("channel [#]")
        SN_plot.set_ylabel("Signal/Noise [ADC]")
        SN_plot.set_title("Signal/Noise of event #%d" % eventnum)
        # fig.subplots_adjust(top=0.88)
        return SN_plot

    def plot_langau_per_clustersize(self, cfg, obj, fig=None):
        """Plots the data calculated so the energy data and the langau"""
        data = obj["MainAnalysis"]["Langau"]
        fit_langau = self.cfg["Fit_langau"]
        # fig = plt.figure("Langau from file: {!s}".format(file))
        # Plot delay
        plot = handle_sub_plots(fig, cfg)
        plot.set_title("Signals of different cluster sizes")
        # hist, edges = np.histogram(data["signal"], bins=data.bins)
        hist_plot_y, hist_plot_x, _ = plot.hist(
            data["signal"],
            bins=data["bins"],
            density=False,
            alpha=0.4,
            color="b",
            label="All clusters",
        )
        # plot.errorbar(edges[:-1], hist, xerr=data["data_error"], fmt='o', markersize=1, color="red")
        if fit_langau:
            plot.plot(data["langau_data"][0], data["langau_data"][1], "r--", color="g")
            # label="Langau: \n mpv: {mpv!s} \n eta: {eta!s} \n sigma: {sigma!s} \n A: {A!s} \n".format(
            #     mpv=data["langau_coeff"][0],
            #     eta=data["langau_coeff"][1],
            #     sigma=data["langau_coeff"][2],
            #     A=data["langau_coeff"][3]))
            # plot.errorbar(data["langau_data"][0], data["langau_data"][1],
            #               np.sqrt(pylandau.langau(data["langau_data"][0], *data["langau_coeff"])),
            #               fmt=".", color="r", label="Error of Fit")
        # if self.cfg["Langau"].get("Charge_scale"):
        #    units = "e"
        # else:
        #    units = "ADC"

        plot.set_xlabel("Cluster Signal")
        plot.set_ylabel("Events [#]")
        # plot.set_title('All clusters Langau from file: {!s}'.format(file))

        # Plot the different clustersizes as well into the langau plot
        colour = ["green", "red", "orange", "cyan", "black", "pink", "magenta"]
        for i, cls in enumerate(data["Clustersize"]):
            if i < 7:
                plot.hist(
                    cls["signal"],
                    bins=data["bins"],
                    density=False,
                    alpha=0.3,
                    color=colour[i],
                    label="Clustersize: {!s}".format(i + 1),
                )
            else:
                self.log.warning(
                    "To many histograms for this plot. "
                    "Colorscheme only supports seven different histograms. Extend if need be!"
                )
                continue

        textstr = "\n".join(
            (
                "mpv = %.f" % (data["langau_coeff"][0]),
                "eta = %.2f" % (data["langau_coeff"][1]),
                "sigma = %.2f" % (data["langau_coeff"][2]),
                "A = %.2f" % (data["langau_coeff"][3]),
            )
        )
        # "entries = %.f" %(len(gain_lst))))

        file_max_signal = open("signal.txt", "a")
        # file_max_signal.write(str(data["langau_coeff"][0]) + "\n")
        file_max_signal.write(
            str(hist_plot_x[np.where(hist_plot_y == hist_plot_y.max())]).strip("[]")
            + "\n"
        )
        file_max_signal.close()

        plot.text(
            0.6,
            0.7,
            textstr,
            transform=plot.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

        plot.legend(loc="lower right")
        return plot

    def plot_seed_signal_e(self, cfg, obj, fig=None):
        """Plots seed signal and langau distribution"""
        data = obj["MainAnalysis"]["Langau"]
        seed_cut = self.cfg.get("Plot_seed_cut", True)
        seed_cut_langau = self.cfg.get("Plot_seed_cut_langau", False)
        if seed_cut_langau:
            fit_langau = self.cfg.get("Fit_langau", None)
        else:
            fit_langau = None

        if seed_cut:
            # fig = plt.figure("Seed cut langau from file: {!s}".format(file))
            # Plot Seed cut langau
            plot = handle_sub_plots(fig, cfg)
            # indizes = np.nonzero(data["signal_SC"] > 0)[0]
            hist_plot_y, hist_plot_x, _ = plot.hist(
                data["signal_SC"],
                bins=data["bins"],
                density=False,
                alpha=0.4,
                color="b",
                label="Signals",
            )
            if fit_langau:
                plot.plot(
                    data["langau_data_SC"][0],
                    data["langau_data_SC"][1],
                    "r--",
                    color="g",
                    label="Fit",
                )
                # "entries = %.f" %(len(gain_lst))))

                sum = 0
                for value in data["langau_data_SC"][1]:
                    sum += value
                mean = sum / len(data["langau_data_SC"][0])
                # print(mean)
                anteil = 1
                RS_RUN = self.cfg.get("RS_RUN", False)
                print(RS_RUN)
                if RS_RUN:
                    for value in data["langau_data_SC"][1]:
                        if mean - 10 < value < mean + 10:
                            if abs(1 - value / mean) < anteil:
                                i = np.where(data["langau_data_SC"][1] == value)
                                anteil = abs(1 - value / mean)

                    plot.vlines(
                        data["langau_data_SC"][0][i],
                        0,
                        data["langau_coeff_SC"][3],
                        label="mean",
                        linestyle="dashed",
                        color="b",
                    )
                    plot.vlines(
                        data["langau_coeff_SC"][0],
                        0,
                        data["langau_coeff_SC"][3],
                        label="mpv",
                        linestyle="dashed",
                        color="r",
                    )

                    textstr = "\n".join(
                        (
                            "mpv = %.f" % (data["langau_coeff_SC"][0]),
                            "mean = %.f" % (data["langau_data_SC"][0][i]),
                            "eta = %.2f" % (data["langau_coeff_SC"][1]),
                            "sigma = %.2f" % (data["langau_coeff_SC"][2]),
                            "A = %.2f" % (data["langau_coeff_SC"][3]),
                        )
                    )
                else:
                    textstr = "\n".join(
                        (
                            "mpv = %.f" % (data["langau_coeff_SC"][0]),
                            "eta = %.2f" % (data["langau_coeff_SC"][1]),
                            "sigma = %.2f" % (data["langau_coeff_SC"][2]),
                            "A = %.2f" % (data["langau_coeff_SC"][3]),
                        )
                    )

                    def gaussian(x, a, mu, sig):
                        return a * np.exp(
                            -np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0))
                        )

                    mu, sigma = norm.fit(data["signal_SC"])
                    plot.plot(
                        data["langau_data_SC"][0],
                        gaussian(
                            data["langau_data_SC"][0], hist_plot_y.max(), mu, sigma
                        ),
                        "r--",
                        color="g",
                        label="Fit",
                    )
                # print(mu,sigma)
                # file_max_signal = open("signal.txt", "a")
                # file_max_signal.write(str(hist_plot_x[np.where(hist_plot_y == hist_plot_y.max())]).strip("[]") + "\n")
                # file_max_signal.write(str(mu) + "\t" + str(data["langau_coeff_SC"][0]) + "\t" + str(hist_plot_y.max()) + "\n")
                # file_max_signal.close()

                plot.text(
                    0.6,
                    0.7,
                    textstr,
                    transform=plot.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
                )
                # plot.errorbar(data["langau_data_SC"][0], data["langau_data_SC"][1],
                #               np.sqrt(pylandau.langau(data["langau_data_SC"][0], *data["langau_coeff_SC"])),
                #               fmt=".", color="r", label="Error of Fit")

            # if self.cfg["Langau"].get("Charge_scale"):
            #    units = "e"
            # else:
            #    units = "ADC"

            plot.set_xlabel("Seed Signal")
            plot.set_ylabel("Events [#]")
            # if fig is None:
            plot.set_title("Seed Signal in e")
            plot.legend(loc="lower right")
            return plot

    def plot_timing_profile(self, cfg, obj, fig=None):
        """Plots the average signal in 1ns steps for each timing of an event.
        Ideally this should be constant. No matter at what timing the signals are coming.
        But if you have pulse shape recognition activated the sampling starts at different timings
        for each event. If the algorithm misjudges the rising edge du to noise etc. the timing profile can
        differ. With this plot you can check this."""
        data = obj["MainAnalysis"]["base"]
        configs = self.cfg.get("Timing2Dhist", {})
        timing_plot = handle_sub_plots(fig, cfg)
        timing_plot.set_xlabel("timing [ns]")
        timing_plot.set_ylabel("average signal [ADC]")
        timing_plot.set_title("Average timing signal of seed hits")
        signal = data["Signal"]
        channels_hit = data["Channel_hit"]
        time = data["Timing"].astype(np.float32)
        sum_singal = np.zeros(len(signal))
        for i, sig, chan in zip(np.arange(len(signal)), signal, channels_hit):
            sum_singal[i] = np.sum(sig[chan])
        max_time = int(np.max(time) + 1)
        timing_data = np.zeros(max_time)
        # var_timing_data = np.zeros(150)
        for timing in range(1, max_time):  # Timing of ALiBaVa
            timing_in = np.nonzero(np.logical_and(time >= timing - 1, time < timing))
            if len(timing_in[0]):
                timing_data[timing - 1] = np.mean(sum_singal[timing_in[0]])
            # var_timing_data[timing-1] = np.std(sum_singal[timing_in[0]])

        timing_plot.bar(
            np.arange(0, max_time), timing_data, alpha=0.4, color="b"
        )  # , yerr=var_timing_data)
        if configs.get("invertY", False):
            timing_plot.invert_yaxis()

    def plot_histogram_of_timing(self, cfg, obj, fig=None):
        """This simply pots a histogram of all timings.
        If pulse shape recognition is on this should be equally distributed.
        """
        # Todo: test it without PSR
        data = obj["MainAnalysis"]["base"]
        timing_hist_plot = handle_sub_plots(fig, cfg)
        timing_hist_plot.set_xlabel("timing [ns]")
        timing_hist_plot.set_ylabel("count [#]")
        timing_hist_plot.set_title("Histogram of timings")

        timing_hist_plot.hist(
            data["Timing"].astype(np.float32), 150, alpha=0.4, color="b"
        )

    def plot_2d_timing_profile(self, cfg, obj, fig=None):
        """Plots the 2D histogram of the timing profile.
        Warning: No averaging done here!!!
        It considers only the hitted channels and sums up the ADC for clusters"""

        data = obj["MainAnalysis"]["base"]
        configs = self.cfg["Timing2Dhist"]
        plot = handle_sub_plots(fig, cfg)
        plot.set_xlabel("timing [ns]")
        plot.set_ylabel("ADC [#]")
        plot.set_title("2D Histogram of timings with signal")

        signal = data["Signal"]
        channels_hit = data["Channel_hit"]
        time = data["Timing"].astype(np.float32)
        sum_singal = np.zeros(len(signal))
        for i, sig, chan in zip(np.arange(len(signal)), signal, channels_hit):
            sum_singal[i] = np.sum(sig[chan])

        counts, xedges, yedges, im = plot.hist2d(
            time,
            sum_singal,
            bins=configs.get("bins", 30),
            range=[[0, np.max(time)], configs.get("yrange", [-250, -1])],
        )
        fig.colorbar(im)
        if configs.get("invertY", False):
            plot.invert_yaxis()

    def plot_chargesharing_2dhist(self, cfg, obj, fig=None):
        """Plots the 2dhisto of the chargesharing"""

        data = obj["MainAnalysis"]["ChargeSharing"]
        # Plot delay
        plot = fig.add_subplot(cfg)
        counts, xedges, yedges, im = plot.hist2d(
            data["data"][0, :],
            data["data"][1, :],
            bins=400,
            range=[[0, 50000], [0, 50000]],
        )
        plot.set_xlabel("A_left (electrons)")
        plot.set_ylabel("A_right (electrons)")
        fig.colorbar(im)
        plot.set_title("Eta distribution")

    def plot_eta_distribution(self, cfg, obj, fig=None):
        """Plots the theta distribution of the chargesharing analysis"""

        data = obj["MainAnalysis"]["ChargeSharing"]
        plot = fig.add_subplot(cfg)
        counts, edges, im = plot.hist(
            data["eta"], bins=200, range=(0, 1), alpha=0.4, color="b"
        )
        plot.set_xlabel("eta")
        plot.set_ylabel("entries")
        plot.set_title("Eta distribution")

        if "PositionResolution" in obj["MainAnalysis"]:
            data2 = obj["MainAnalysis"]["PositionResolution"]
            plot.plot(data["fits"]["eta"][1][:-1], data2["N_eta"], color="red")

    def plot_theta_distribution(self, cfg, obj, fig=None):
        """Plots the eta distribution of the chargesharing analysis"""

        data = obj["MainAnalysis"]["ChargeSharing"]
        plot = fig.add_subplot(cfg)
        counts, edges, im = plot.hist(
            data["theta"] / np.pi, bins=200, alpha=0.4, color="b", range=(0, 0.5)
        )
        plot.set_xlabel("theta/Pi")
        plot.set_ylabel("entries")
        plot.set_title("Theta distribution")

        if "PositionResolution" in obj["MainAnalysis"]:
            data2 = obj["MainAnalysis"]["PositionResolution"]
            plot.plot(
                data["fits"]["theta"][1][:-1] / np.pi, data2["N_theta"], color="red"
            )

    def plot_eta_algorithm_positions(self, cfg, obj, fig=None):
        """Eta algorithm positions plot"""
        data = obj["MainAnalysis"]["PositionResolution"]
        plot = fig.add_subplot(cfg)
        plot.set_xlabel("Position [um]")
        plot.set_ylabel("Hits [#]")
        plot.set_title("Hit positions with eta")
        counts, edges, im = plot.hist(data["eta"], bins=50, alpha=0.4, color="b")

    def plot_theta_algorithm_positions(self, cfg, obj, fig=None):
        """Eta algorithm positions plot"""
        data = obj["MainAnalysis"]["PositionResolution"]
        plot = fig.add_subplot(cfg)
        plot.set_xlabel("Position [um]")
        plot.set_ylabel("Hits [#]")
        plot.set_title("Hit positions with theta")
        counts, edges, im = plot.hist(data["theta"], bins=50, alpha=0.4, color="b")

    def plot_efficiency(self, cfg, obj, fig=None):
        """Plot efficiency of seed signals vs. applied threshold and
        show the maximum threshold for aim_eff"""

        if "Langau" in obj["MainAnalysis"]:
            plot = handle_sub_plots(fig, cfg)
            data = obj["MainAnalysis"]["Langau"]

            aim_eff = self.cfg["Efficiency_plot"]["aim_eff"]
            max_range = self.cfg["Efficiency_plot"]["max_range"]
            step_size = self.cfg["Efficiency_plot"]["step_size"]

            step_lst = np.arange(0, max_range + step_size, step_size)
            eff_lst = np.zeros(len(step_lst), dtype=np.float)
            tot_len = len(data["signal_SC"])
            # In principal its a survival function what we calculate here
            # Todo: Use the scipy version for the survival function rv_continuous.sf
            for i, step in enumerate(step_lst):
                eff_lst[i] = np.count_nonzero(data["signal_SC"] > step) / tot_len
            plot.plot(step_lst, eff_lst, "r--", label="Efficiency")
            index = np.where(eff_lst >= aim_eff)[0][-1]
            if step_lst[index] >= max_range:
                threshold = "> {}".format(max_range)
            else:
                threshold = str(step_lst[index])
            textstr = "\n".join(
                (
                    "Sth @ %s = %s" % (str(aim_eff), threshold),
                    "Events = %.f" % (tot_len),
                )
            )
            plot.text(
                0.5,
                0.7,
                textstr,
                transform=plot.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
            )
            plot.set_xlabel("Threshold")
            plot.set_ylabel("Efficiency [%]")
            plot.set_title("Efficiency vs. Seed Threshold")
            plot.fill_between(step_lst, 0, eff_lst, facecolor="blue", alpha=0.2)
            plot.legend()
            return plot
