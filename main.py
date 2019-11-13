"""Wrapper for full alibava analysis via console"""
import os
from argparse import ArgumentParser
from plot_data import PlotData
from analysis_classes.utilities import create_dictionary
from analysis_classes import Calibration
from analysis_classes import NoiseAnalysis
from analysis_classes import MainAnalysis
from analysis_classes.utilities import save_all_plots, save_dict, read_meas_files
import matplotlib.pyplot as plt
DEF = os.path.join(os.getcwd(), "Examples", "marius_config.yml")

def main(args):
    """Start analysis"""
    if args.config != "":
        cfg = create_dictionary(args.config)
        ext = os.path.dirname(args.config)
    else:
        cfg = create_dictionary(DEF)

    plot = PlotData(os.path.join(os.getcwd(),ext,cfg.get("plot_config_file", "plot_cfg.yml")))
    results = {}

    for ped, cal, run in read_meas_files(cfg):

        ped_data = NoiseAnalysis(ped, configs=cfg)

        results["NoiseAnalysis"] = ped_data

        cal_data = Calibration(cal, Noise_calc=ped_data, configs=cfg)

        results["Calibration"] = cal_data

        cfg.update({"calibration": cal_data,
                    "noise_analysis": ped_data})

        run_data = MainAnalysis(run, configs=cfg)
        results["MainAnalysis"] = run_data.results

        # Start plotting all results
        plot.plot_data(cfg, results, group="from_file")

        if cfg.get("Output_folder", "") and cfg.get("Output_name", ""):
            if cfg["Output_name"] == "generic":
                fileName = os.path.basename(os.path.splitext(run)[0])
            else:
                fileName = cfg["Output_name"]
            plt.close("all")
            save_all_plots(fileName, cfg["Output_folder"], dpi=300)
            if cfg.get("Pickle_output", False):
                save_dict(run_data.outputdata,
                          os.path.join(cfg["Output_folder"],
                                       cfg["Output_name"], ".dba"))

    if args.show_plots:
        plot.show_plots()

if __name__ == "__main__":

    PARSER = ArgumentParser()
    PARSER.add_argument("--config",
                        help="The path to the config file for the analysis run",
                        default="")
    PARSER.add_argument("--show_plots",
                        help="Show all generated plots when analysis is done",
                        type=bool, default=True)
    main(PARSER.parse_args())
