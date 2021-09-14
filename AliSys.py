#!/usr/bin/env python3
"""Wrapper for full alibava analysis via console"""
import os, sys
from argparse import ArgumentParser
from plot_data import PlotData
from analysis_classes.utilities import create_dictionary
from analysis_classes import Calibration
from analysis_classes import NoiseAnalysis
from analysis_classes import MainAnalysis
from analysis_classes.utilities import save_all_plots, save_dict, read_meas_files
import matplotlib.pyplot as plt
from analysis_classes import post_analysis
import pdb

def main(args):
    """Start analysis"""
    if args.config:
        cfg = create_dictionary(args.config)
        ext = os.path.dirname(args.config)
    else:
        print("AliSys needs at least the --config parameter. Type AliSys --help to see all params")
        sys.exit(0)
    plot = PlotData(os.path.join(os.getcwd(),ext,cfg.get("plot_config_file", "plot_cfg.yml")))
    results = {}
    post_analysis_results=[]

    meas_files = read_meas_files(cfg)
    it = 0
    for ped, cal, run in meas_files:
        it+=1

        ped_data = NoiseAnalysis(ped, configs=cfg)

        results["NoiseAnalysis"] = ped_data

        cal_data = Calibration(cal, Noise_calc=ped_data, configs=cfg)

        results["Calibration"] = cal_data

        cfg.update({"calibration": cal_data,
                    "noise_analysis": ped_data})

        if run:
            run_data = MainAnalysis(run, configs=cfg)
            results["MainAnalysis"] = run_data.results
            
        #use data for further analysis..
        if cfg.get('run_post_analysis'): post_analysis_results.append(post_analysis.main(ped, cal, run, results))

        # Start plotting all results
        if it > 1:  # Closing the old files
            plt.close("all")
        plot.start_plotting(cfg, results, group="from_file")

        if cfg.get("Output_folder", "") and cfg.get("Output_name", "") and cfg.get("Save_output", False):
            if cfg["Output_name"] == "generic":
                fileName = os.path.basename(os.path.splitext(run)[0])
            else:
                fileName = cfg["Output_name"]
            save_all_plots(fileName, cfg["Output_folder"], dpi=300)
            if cfg.get("Pickle_output", False):
                save_dict(run_data.outputdata,
                          cfg["Output_folder"],
                          cfg["Output_name"],
                          cfg["Pickle_output"])

    if cfg.get('run_post_analysis'): post_analysis.final(post_analysis_results)
    if args.show_plots and it==1:
        plot.show_plots()

    plt.close("all")

if __name__ == "__main__":

    PARSER = ArgumentParser()
    PARSER.add_argument("--config",
                        help="The path to the config file for the analysis run",
                        default="")
    PARSER.add_argument("--show_plots",
                        help="Show all generated plots when analysis is done",
                        type=bool, default=True)
    main(PARSER.parse_args())
