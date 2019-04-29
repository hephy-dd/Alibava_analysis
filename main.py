"""Wrapper for full alibava analysis via console"""
import os
from argparse import ArgumentParser
from plot_data import PlotData
from analysis_classes.utilities import create_dictionary
from analysis_classes import Calibration
from analysis_classes import NoiseAnalysis
from analysis_classes import MainAnalysis
from analysis_classes.utilities import save_all_plots, save_dict, read_meas_files
DEF = os.path.join(os.getcwd(), "Examples", "marius_config.yml")

def main(args):
    """Start analysis"""
    if args.config != "":
        cfg = create_dictionary(args.config)
    else:
        cfg = create_dictionary(DEF)
    plot = PlotData()
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
            save_all_plots(cfg["Output_name"], cfg["Output_folder"], dpi=300)
            if cfg.get("Pickle_output", False):
                save_dict(run_data.outputdata,
                          os.path.join(cfg["Output_folder"],
                                       cfg["Output_name"], ".dba"))
    plot.show_plots()

if __name__ == "__main__":

    PARSER = ArgumentParser()
    PARSER.add_argument("--config",
                        help="The path to the config file for the analysis run",
                        default="")
    main(PARSER.parse_args())
