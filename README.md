# FRONTIERS Demonstration Codes

[Note: this project is in active development.]

This is a repository of [SUNDIALS](https://github.com/LLNL/sundials)-based applications to assess and demonstrate the parallel performance of new capabilities that have been added to SUNDIALS as part of the [FRONTIERS SciDAC project](https://www.scidac.gov/projects/2023/fusion-energy-sciences/project_2023_006.html).


## Installation

The following steps describe how to build the demonstration code in a Linux or MacOS environment.


### Gettting the Code

To obtain the code, clone this repository with Git:

```bash
  git clone https://github.com/sundials-codes/frontiers-demonstrations.git
```


### Requirements

To compile the codes in this repository you will need:

* [CMake](https://cmake.org) 3.20 or newer (both for SUNDIALS and for this repository)

* C compiler (C99 standard) and C++ compiler (C++11 standard)

* an MPI library e.g., [OpenMPI](https://www.open-mpi.org/), [MPICH](https://www.mpich.org/), etc.


The codes in this repository depend on one external library:

* [SUNDIALS](https://github.com/LLNL/sundials)

If this is not already available on your system, it may be cloned from GitHub as a submodule.  After cloning this repository using the command above, you can retrieve this submodule via:

```bash
  cd frontiers-demonstrations/deps
  git submodule init
  git submodule update
```

We note that a particular benefit of retrieving dependencies using submodules is that they point to specific revisions of dependent libraries that are known to work correctly with the codes in this repository.

Additionally, the python postprocessing scripts in this repository require a number of additional python packages, including [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), and [Pandas](https://pandas.pydata.org/).


### Building the Dependencies

We recommend that users follow the posted instructions for installing SUNDIALS.

#### SUNDIALS

[The SUNDIALS build instructions are linked here](https://sundials.readthedocs.io/en/latest/sundials/Install_link.html#building-and-installing-with-cmake).  Note that of the many SUNDIALS build options, this repository requires only a minimal SUNDIALS build with MPI.  The following steps can be used to build SUNDIALS using this minimal configuration:

```bash
mkdir deps/sundials/build
cd deps/sundials/build
cmake -DCMAKE_INSTALL_PREFIX=../../sundials-install -DENABLE_MPI=ON -DSUNDIALS_INDEX_SIZE=32 ..
make -j install
```

Instructions for building SUNDIALS with additional options [may be found here](https://sundials.readthedocs.io/en/latest/sundials/Install_link.html).

#### Python packages

Since each of NumPy, Matplotlib, and Pandas are widely used, it is likely that these are already installed on your system.  However, if those are missing or need to be updated, then we recommend that these be installed in a Python virtual environment, as follows:

```bash
python3 -m venv .venv
source .venv/bin/activate
cd deps
pip install -r python_requirements.txt
```

You may "deactivate" this Python environment from your current shell with the command

```bash
deactivate
```

and in the future you can "reactivate" the python environment in your shell by running from the top-level directory of this repository

```bash
source .venv/bin/activate
```


### Configuration Options

Once the necessary dependencies have been installed, the following CMake variables can be used to configure the build for this repository:

* `CMAKE_INSTALL_PREFIX` - the path where executables and input files should be installed e.g., `my/install/path`. The executables will be installed in the `bin` directory and input files in the `tests` directory under the given path.

* `CMAKE_C_COMPILER` - the C compiler to use e.g., `mpicc`. If not set, CMake will attempt to automatically detect the C compiler.

* `CMAKE_C_FLAGS` - the C compiler flags to use e.g., `-g -O2`.

* `CMAKE_C_STANDARD` - the C standard to use, defaults to `99`.

* `CMAKE_CXX_COMPILER` - the C++ compiler to use e.g., `mpicxx`. If not set,
  CMake will attempt to automatically detect the C++ compiler.

* `CMAKE_CXX_FLAGS` - the C++ flags to use e.g., `-g -O2`.

* `CMAKE_CXX_STANDARD` - the C++ standard to use, defaults to `11`.

* `SUNDIALS_ROOT` - the root directory of the SUNDIALS installation, defaults to the value of the `SUNDIALS_ROOT` environment variable. If not set, CMake will attempt to automatically locate a SUNDIALS install on the system.

* `CMAKE_CUDA_COMPILER` - the CUDA compiler to use e.g., `nvcc`. If not set,
  CMake will attempt to automatically detect the CUDA compiler.

* `CMAKE_CUDA_FLAGS` - the CUDA compiler flags to use.

* `CMAKE_CUDA_ARCHITECTURES` - the CUDA architecture to target e.g., `70`.


### Building

Like most CMake-based projects, in-source builds are not permitted, so the code should be configured and built from a separate build directory, e.g.,

```bash
  mkdir frontiers-demonstrations/build
  cd frontiers-demonstrations/build
  cmake -DSUNDIALS_ROOT="[sundials-path] .."
  make -j install
```

where `[sundials-path]` is the path to the top-level folder containing the SUNDIALS installation.

If SUNDIALS was installed using the submodule-based instructions above, then the following commands should be sufficient to install the executables into a new `frontiers-demonstrations/bin` directory:

```bash
  mkdir frontiers-demonstrations/build
  cd frontiers-demonstrations/build
  cmake -DSUNDIALS_ROOT=../deps/sundials-install ..
  make -j install
```
