# Alibava_analysis

This software was developed by Dominic Blöch during his Phd Thesis at the HEPHY Vienna.
It is intended to analyse Alibava Systems hdf5 files. It feautures a basic clustering analysis as well as Langau fitting to the energy deposition of events

## Getting Started

In order to run this program you need a Python Anaconda distribution. For more information on versions see Chapter "What you need".

### Setting Up The Environement

With python up and running you can run the the "environement_setup.py" file by.

```
conda env create -f Alibava_requirement.yml
```

this will (when Anaconda is installed) automatically install all required modules for the program to run. If you don't have Anaconda installed and don't want to use it, you can look in the "Alibava_requirement.yml" file to see what dependencies the program needs.

## Running The Program

Now it should be possible to run the program by:

```
python main_analysis.py --config <path_to_config YAML file>
```


## How to Use

In the future here will be a Link to the docs or something else

## What you need

### Python

You need python 3.6 64 bit distribution. (32 bit works as well but unstable)
I recommended to use this program with a [Anaconda](https://www.anaconda.com/download/) python distribution, it will work with a normal version too, but I have not tested it.

The program is known to be running on Windows, Linux (Centos7) and Mac.


## Built With

* [Numba](http://numba.pydata.org/) - For Low-Level optimizations
* [Numpy](http://www.numpy.org/) - For numerical operations
* [SciPy](https://www.scipy.org/) - For numerical operations
* [Matplotlib](https://matplotlib.org/) - For the plots
* [PyLandau](https://github.com/SiLab-Bonn/pylandau) - For Langau fitting
#COMMENT: pylandau requiers Microsoft Visual C++ 14.0 from https://visualstudio.microsoft.com/visual-cpp-build-tools/
# COMMENT: After the installation you need to update setuptools as well 'pip install --upgrade setuptools'


## Authors

* **Dominic Blöch** - *Initial work* - [Chilldose](https://github.com/Chilldose)

## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE) file for details

## Acknowledgments

* Thanks to my friends for their inspiring conversations and help
