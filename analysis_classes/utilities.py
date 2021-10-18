"""This file contains functions and classes which can be classified as utilitie
functions for a more general purpose. Furthermore, these functions are for
python analysis of ALIBAVA files."""
# pylint: disable=C0103,C0301,R1710,R0903,E0401
import logging
import logging.config
import os
import struct
import sys
from pydoc import locate
import numpy as np
import h5py
import yaml
from tqdm import tqdm
from six.moves import cPickle as pickle  # for performance
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.integrate as integrate
import json
from copy import deepcopy
from argparse import ArgumentParser
import pdb


def read_meas_files(cfg, match_files=True):
    """Reads cfg file, returns lists of files and compares their length"""
    ped_files = []
    cal_files = []
    run_files = []
    if "Pedestal_file" in cfg:
        ped_files = cfg["Pedestal_file"]
    if "Delay_scan" in cfg:
        cal_files = cfg["Delay_scan"]
    if "use_charge_cal" in cfg:
        if cfg["use_charge_cal"] and "Charge_scan" in cfg:
            cal_files = cfg["Charge_scan"]
            print("Using Charge scan")

    if "Measurement_file" in cfg:
        run_files = cfg["Measurement_file"]

    if not match_files:
        if isinstance(ped_files, str):
            ped_files = [ped_files]
        if isinstance(cal_files, str):
            cal_files = [cal_files]
        if isinstance(run_files, str):
            run_files = [run_files]

        return ped_files, cal_files, run_files

    # Case selection if files are lists or simply one file as paths
    if all(isinstance(i, list) for i in [ped_files, run_files, cal_files]):
        if not len(ped_files) == len(run_files) or not len(cal_files) == len(run_files):
            raise ValueError(
                "Number of pedestal, calibration and measurement files"
                " does not match..."
            )
    # Case where only a path string is passed
    elif all(isinstance(i, str) for i in [ped_files, run_files, cal_files]):
        ped_files = [ped_files]
        cal_files = [cal_files]
        run_files = [run_files]

    # Case where only one pedestal and cal file but several run files, use the same pedestal and cal for all runs
    elif (
        isinstance(run_files, list)
        and isinstance(ped_files, str)
        and isinstance(cal_files, str)
    ):
        # self.log.info("Several run files for only one pedestal and calibration file. "
        #              "Same calibration and pedestal file for all measurement files will be used")
        ped_files = [ped_files for i in run_files]
        cal_files = [cal_files for i in run_files]

    else:
        raise ValueError(
            "Pedestal, calibration and measurement files must "
            "either be passed as strings or as "
            "lists of same length"
        )

    return zip(ped_files, cal_files, run_files)


def load_parser():
    PARSER = ArgumentParser()
    PARSER.add_argument(
        "--config", help="The path to the config file for the analysis run", default=""
    )
    # PARSER.add_argument("--path", help="The path to a measurement file", default="")

    PARSER.add_argument(
        "--show_plots",
        help="Show all generated plots when analysis is done",
        type=bool,
        default=True,
    )
    return PARSER


def handle_sub_plots(fig, index=111):
    """Adds subplot to existing figure or creates a new one if fig
    non-existing"""
    if fig is None:
        fig = plt.figure()
        plot = fig.add_subplot(111)
    else:
        plot = fig.add_subplot(index)
    return plot


def init_logger(path="logger.yml", default_level=logging.INFO, env_key="LOG_CFG"):
    """Loads a logger file and initiates logger"""
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(os.path.normpath(path)):
        with open(path, "rt") as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


LOG = logging.getLogger("utilities")


def load_plugins(valid_plugins):
    """Load additional analysis functions. File names are expected to be all
    lower case while class names are capitalized.
    Args:
        - valid_plugins (str): class names"""
    all_plugins = {}
    all_analysis_files = os.listdir("./analysis_classes/")
    if valid_plugins:
        for file in all_analysis_files:
            for plugin in valid_plugins:
                if os.path.splitext(file)[0].lower() == plugin.lower():
                    all_plugins[plugin] = locate(
                        "analysis_classes." + plugin + "." + plugin
                    )
    return all_plugins


def create_dictionary(abs_filepath):
    """Creates a dictionary with all values written in the file using yaml"""
    with open(abs_filepath, "r") as yfile:
        dic = yaml.safe_load(yfile)
    return dic


def import_h5(path):
    """
    This functions imports hdf5 files generated by ALIBAVA.
    If you pass several pathes, then you get list of objects back, containing
    the data respectively
    :param pathes: pathes to the datafiles which should be imported
    :return: list
    """
    # First check if path exists and if so import hdf5 file
    try:
        if not os.path.exists(os.path.normpath(path)):
            raise Exception("The path {!s} does not exist.".format(path))
        return h5py.File(os.path.normpath(path), "r")
    except OSError as err:
        LOG.error("Encountered an OSError: %s", str(err))
        return False


def get_xy_data(data, header=0):
    """This functions takes a list of strings, containing a header and xy data,
    return values are 2D np.array of the data and the header lines"""

    np2Darray = np.zeros((len(data) - int(header), 2), dtype=np.float32)
    for i, item in enumerate(data):
        if i > header - 1:
            list_data = list(map(float, item.split()))
            np2Darray[i - header] = np.array(list_data)
    return np2Darray


# @jit(nopython=False)
def read_binary_Alibava(filepath):
    """Reads binary alibava files"""
    with open(os.path.normpath(filepath), "rb") as f:
        header = f.read(16)
        Starttime = struct.unpack("II", header[0:8])[0]  # Is a uint32
        Runtype = struct.unpack("i", header[8:12])[0]  # int32
        Headerlength = struct.unpack("I", header[12:16])
        header = f.read(Headerlength[0])
        Header = struct.unpack("{}s".format(Headerlength[0]), header)[0].decode("Utf-8")
        Pedestal = np.array(struct.unpack("d" * 256, f.read(8 * 256)), dtype=np.float32)
        Noise = np.array(struct.unpack("d" * 256, f.read(8 * 256)), dtype=np.float32)
        byteorder = sys.byteorder

        # Data Blocks
        # Read all data Blocks
        # Warning Alibava Binary calibration files have no indicatior how many events are really inside the file
        # The eventnumber corresponds to the pulse number -->
        # Readout of files have to be done until end of file is reached
        # and the eventnumber must be calculated --> Advantage: Damaged files can be read as well
        # events = Header.split("|")[1].split(";")[0]
        event_data = []
        events = 0
        eof = False
        # for event in range(int(events)):
        while not eof:
            blockheader = f.read(4)  # should be 0xcafe002
            if blockheader == b"\x02\x00\xfe\xca" or blockheader == b"\xca\xfe\x00\x02":
                events += 1
                blocksize = struct.unpack("I", f.read(4))
                event_data.append(f.read(blocksize[0]))
            else:
                LOG.info(
                    "Warning: While reading data Block {}. "
                    "Header was not the 0xcafe0002 it was {!s}".format(
                        events, str(blockheader)
                    )
                )
                if not blockheader:
                    LOG.info(
                        "Persumably end of binary file reached. "
                        "Events read: {}".format(events)
                    )
                    eof = True
        dic = {
            "header": {"noise": Noise, "pedestal": Pedestal, "Attribute:setup": None},
            "events": {
                "header": Header,
                "signal": np.zeros((int(events), 256), dtype=np.float32),
                "temperature": np.zeros(int(events), dtype=np.float32),
                "time": np.zeros(int(events), dtype=np.float32),
                "clock": np.zeros(int(events), dtype=np.float32),
            },
            "scan": {
                "start": Starttime,
                "end": None,
                "value": None,  # Values of cal files for example. eg. 32 pulses for a charge scan steps should be here
                "attribute:scan_definition": None,
            },
        }
        # Disect the header for the correct informations for values
        points = Header.split("|")[1].split(";")
        params = [x.strip("\x00") for x in points]

        # Alibava binary have (unfortunately) a non consistend header format
        # Therefore, we have to distinguish between the two formats --> len(params) = 4 --> Calibration
        # len(params) = 2 --> Eventfile

        if len(params) >= 4:  # Cal file
            dic["scan"]["value"] = np.arange(
                int(params[1]), int(params[2]), int(params[3])
            )  # aka xdata
        elif len(params) == 2:  # Events file
            dic["scan"]["value"] = np.arange(0, int(params[0]), step=1)  # aka xdata

        shift1 = int.from_bytes(b"0xFFFF0000", byteorder=byteorder)
        shift2 = int.from_bytes(b"0xFFFF", byteorder=byteorder)
        # decode data from data Blocks
        for i, event in enumerate(event_data):
            dic["events"]["clock"][i] = struct.unpack("III", event[0:12])[-1]
            coded_time = struct.unpack("I", event[12:16])[0]
            # coded_time = event[12:16]
            ipart = (coded_time & shift1) >> 16
            fpart = (np.sign(ipart)) * (coded_time & shift2)
            time = 100 * ipart + fpart
            dic["events"]["time"][i] = time
            dic["events"]["temperature"][i] = (
                0.12 * struct.unpack("H", event[16:18])[0] - 39.8
            )

            # There seems to be garbage data which needs to be cut out
            padding = 18 + 32
            part1 = list(struct.unpack("h" * 128, event[padding : padding + 2 * 128]))
            padding += 2 * 130 + 28
            part2 = list(struct.unpack("h" * 128, event[padding : padding + 2 * 128]))
            part1.extend(part2)
            dic["events"]["signal"][i] = np.array(part1)
            # dict["events"]["signal"][i] =struct.unpack("H"*256, event[18:18+2*256])
            # extra = struct.unpack("d", event[18+2*256:18+2*256+4])[0]

    return dic


def read_file(filepath, binary=False):
    """Just reads a file and returns the content line by line"""
    if os.path.exists(os.path.normpath(filepath)):
        if not binary:
            with open(os.path.normpath(filepath), "r") as f:
                read_data = f.readlines()
            return read_data
    else:
        LOG.info("No valid path passed: {!s}".format(filepath))
        return None


def clustering(estimator):
    """Does the clustering up to the max cluster number, you just need the
    estimator and its config parameters"""
    return estimator


def count_sub_length(ndarray):
    """This function count the length of sub elements (depth 1) in the ndarray
    and returns an array with the lengthes with the same dimension as ndarray"""
    results = np.zeros(len(ndarray))
    # COMMENT: there is a neat built-in function for this called enumerate
    for i in range(len(ndarray)):
        if len(ndarray[i]) == 1:
            results[i] = len(ndarray[i][0])
    return results


def save_all_plots(name, folder, figs=None, dpi=200):
    """
    This function saves all generated plots to a specific folder with the defined name in one pdf
    :param name: Name of output
    :param folder: Output folder
    :param figs: Figures which you want to save to one pdf (leaf empty for all plots) (list)
    :param dpi: image dpi
    :return: None
    """
    try:
        pp = PdfPages(os.path.normpath(folder) + "\\" + name + ".pdf")
    except PermissionError:
        raise PermissionError(
            "While overwriting the file {!s} a permission error occured, "
            "please close file if opened!".format(name + ".pdf")
        )
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
        # print(figs)
        # axes = figs[0].get_axes()
        # for ax in axes:
        #     plt.axes(ax)
    for fig in tqdm(figs, desc="Saving plots"):
        # fig = plt.figure()
        fig.set_figheight(9)
        fig.set_figwidth(16)
        fig.savefig(pp, format="pdf")
    pp.close()


class NoStdStreams(object):
    """Surpresses all output of a function when called with with"""

    def __init__(self, stdout=None, stderr=None):
        self.devnull = open(os.devnull, "w")
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()


def gaussian(x, mu, sig, a):
    """Simple but fast implementation of as gaussian distribution"""
    return a * np.exp(-np.power(x - mu, 2.0) / (2.0 * np.power(sig, 2.0)))


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def save_configs(configs, name, path):
    """This function saves the configs of the current run"""
    try:
        yaml.safe_dump(configs, os.path.normpath(path + "\\" + name))
    except OSError as err:
        LOG.error("Failed to save configs.", exc_info=True)


class Bdata:
    """Creates an object which can handle numpy arrays. By passing lables you
    can get the columns of the multidimensional array. Its like a pandas array
    but with way less overhead.
    If you store a Bdata object you can get columns by accessing it via Bdata['label']
    Not passing an argument results in"""

    def __init__(self, data=np.array([]), labels=None):

        # Has nothing to do here, this is a Data type not a typical class
        # self.log = logging.getLogger(__class__.__name__)
        # self.log.setLevel(logging.DEBUG)
        # if self.log.hasHandlers() is False:
        #    format_string = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        #    formatter = logging.Formatter(format_string)
        #    console_handler = logging.StreamHandler()
        #    console_handler.setFormatter(formatter)
        #    self.log.addHandler(console_handler)

        self.data = data
        self.labels = labels
        self.log = LOG

        if len(self.data) != len(self.labels):
            self.log.warning("Data missmatch!")

    def __getitem__(self, arg=None):
        # COMMENT: else returns 'None' is correct?
        if arg:
            return self.get(arg)

    def __repr__(self):
        return repr(self.data)

    def keys(self):
        """Returns the keys list"""
        return self.labels

    def get(self, label):
        """DOC of function"""
        return self.data[:, self.labels.index(label)]


def save_dict(di_, filepath_, name_, type_):
    """
    Dict to save as type
    :param di_: dict
    :param filename_: filepath
    :param type_: type of output
    :return: None
    """
    if type_.lower() == "json":
        # JSON serialize
        LOG.info("Saving JSON file...")
        save_dict_as_json(
            deepcopy(di_), os.path.join(os.path.normpath(filepath_)), name_
        )

    if type_.lower() == "pickle":
        with open(
            os.path.join(os.path.normpath(filepath_), "{}.pickle".format(name_)), "wb"
        ) as f:
            LOG.info("Saving pickle to file...")
            pickle.dump(di_, f)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Bdata):
            data = {}
            for key in obj.labels:
                data[key] = obj[key].tolist()
            return data
        return json.JSONEncoder.default(self, obj)


def save_dict_as_json(data, dirr, base_name):

    # Create a json dump
    json_dump = json.dumps(data, cls=NumpyEncoder)
    # Write the data to file, the whole dic
    with open(os.path.join(dirr, "{}.json".format(base_name)), "w") as outfile:
        json.dump(json_dump, outfile)


def load_dict(filename_):
    """DOC of function"""
    with open(os.path.normpath(filename_), "rb") as f:
        ret_di = pickle.load(f)
    return ret_di


# Here the logger will be initialized!
init_logger(path="logger.yml")


def set_attributes(obj, dict):
    """Set all attributes for the configs in the passed object"""
    for name, value in dict.items():
        setattr(obj, name, value)


def integ(f, *args):
    """Generall purpose integration function"""
    return integrate.quad(lambda x: float(f(x, *args)), 80, 180)


if __name__ == "__main__":
    pass
