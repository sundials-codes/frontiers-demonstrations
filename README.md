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

The results in the paper on embedded IMEX SSP methods were generated with the `embedded-ssp-imex` branch of this repository. To reproduce them, switch to that branch after cloning:

```bash
  cd frontiers-demonstrations
  git checkout embedded-ssp-imex
```


### Requirements

To compile the codes in this repository you will need:

* [CMake](https://cmake.org) 3.20 or newer (both for SUNDIALS and for this repository)

* C compiler (C99 standard) and C++ compiler (C++17 standard)

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

Additionally, the python postprocessing scripts in this repository require a number of additional python packages, including [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), [Pandas](https://pandas.pydata.org/), and [openpyxl](https://openpyxl.readthedocs.io/) (used by Pandas to write the Excel files).


### Building the Dependencies

We recommend that users follow the posted instructions for installing SUNDIALS.

#### SUNDIALS

[The SUNDIALS build instructions are linked here](https://sundials.readthedocs.io/en/latest/sundials/Install_link.html#building-and-installing-with-cmake).  Note that of the many SUNDIALS build options, this repository requires only a minimal SUNDIALS build with MPI.  The following steps can be used to build SUNDIALS using this minimal configuration:

```bash
mkdir deps/sundials/build
cd deps/sundials/build
cmake -DCMAKE_INSTALL_PREFIX=../../sundials-install -DSUNDIALS_ENABLE_MPI=ON -DSUNDIALS_INDEX_SIZE=32 ..
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

* `CMAKE_C_COMPILER` - the C compiler to use e.g., `mpicc`. If not set, CMake will attempt to automatically detect the C compiler.

* `CMAKE_C_FLAGS` - the C compiler flags to use e.g., `-g -O2`.

* `CMAKE_C_STANDARD` - the C standard to use, defaults to `99`.

* `CMAKE_CXX_COMPILER` - the C++ compiler to use e.g., `mpicxx`. If not set,
  CMake will attempt to automatically detect the C++ compiler.

* `CMAKE_CXX_FLAGS` - the C++ flags to use e.g., `-g -O2`.

* `CMAKE_CXX_STANDARD` - the C++ standard to use, defaults to `17`.

* `SUNDIALS_ROOT` - the root directory of the SUNDIALS installation, defaults to the value of the `SUNDIALS_ROOT` environment variable. If not set, CMake will attempt to automatically locate a SUNDIALS install on the system.

* `CMAKE_CUDA_COMPILER` - the CUDA compiler to use e.g., `nvcc`. If not set,
  CMake will attempt to automatically detect the CUDA compiler.

* `CMAKE_CUDA_FLAGS` - the CUDA compiler flags to use.

* `CMAKE_CUDA_ARCHITECTURES` - the CUDA architecture to target e.g., `70`.

* `USE_IMEX_SSP_TABLES` - compile the examples with the IMEX SSP Butcher tables in `imex_ssp_tables/imex_ssp_tables.hpp` (see [IMEX SSP Butcher Tables](#imex-ssp-butcher-tables) below), defaults to `ON`.


### Building

Like most CMake-based projects, in-source builds are not permitted, so the code should be configured and built from a separate build directory, e.g.,

```bash
  mkdir frontiers-demonstrations/build
  cd frontiers-demonstrations/build
  cmake -DSUNDIALS_ROOT="[sundials-path]" ..
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

The executables, together with the Python test and plotting scripts and `run-all-examples.sh`, are always installed into the `bin` directory at the top level of this repository. After changing a source file, running `make -j install` again from the build directory recompiles and reinstalls.


### IMEX SSP Butcher Tables

The examples use the following embedded implicit-explicit strong-stability-preserving (IMEX SSP) methods:

| Method  | `--dirk_table`                 | `--erk_table`                |
|---------|--------------------------------|------------------------------|
| SSP212  | `ARKODE_SSP_SDIRK_2_1_2`       | `ARKODE_SSP_ERK_2_1_2`       |
| SSP312  | `ARKODE_SSP_DIRK_3_1_2`        | `ARKODE_SSP_ERK_3_1_2`       |
| SSPL312 | `ARKODE_SSP_LSPUM_SDIRK_3_1_2` | `ARKODE_SSP_LSPUM_ERK_3_1_2` |
| SSP423  | `ARKODE_SSP_ESDIRK_4_2_3`      | `ARKODE_SSP_ERK_4_2_3`       |
| SSP923  | `ARKODE_SSP_ESDIRK_9_2_3`      | `ARKODE_SSP_ERK_9_2_3`       |

The coefficients of these methods are stored in `imex_ssp_tables/imex_ssp_tables.hpp`; for SSPL312 these are the updated embedding coefficients. When `USE_IMEX_SSP_TABLES` is `ON` (the default), every example is compiled with this header. When an example selects one of the methods above by name (through `ARKStepSetTableName`), the tables are built from the header and passed to SUNDIALS with `ARKStepSetTables`. Any other table name, e.g., the `ARKODE_ARK548L2SA_*` tables used for the reference solutions, is passed to SUNDIALS unchanged.

Because the coefficients come from this header, the five methods do not need to be part of the SUNDIALS installation that is used. Other parts of SUNDIALS (e.g., step-size control) can change between SUNDIALS versions, so results may differ slightly between versions even though the method coefficients are the same. Results that are compared with each other should therefore be generated with the same SUNDIALS installation.


## Running the Examples

All tests are run from the `bin` directory. If the Python packages were installed in a virtual environment, activate it first.

To run all examples, use the bash script `run-all-examples.sh` (run it with `./` or `bash`, not with `python3`):

```bash
  cd frontiers-demonstrations/bin
  ./run-all-examples.sh
```

This runs, in order: the linear advection-reaction example, the population density example, and the hyperbolic relaxation example. For each example the reference solutions are generated first, followed by the test runs.

The examples can also be run individually. In each case the reference solution script must be run before the test script:

```bash
  cd frontiers-demonstrations/bin

  # linear advection-reaction
  python3 runtests_linear_adv_rec_refSol.py
  python3 runtests_linear_adv_rec.py

  # population density
  python3 runtests_population_refSol.py
  python3 runtests_population.py uncheckedSMOOTH
  python3 runtests_population.py ssp

  # hyperbolic relaxation
  python3 runtests_hyperbolic_relaxation_refSol.py
  python3 runtests_hyperbolic_relaxation.py
```

The population test script has two modes, which must be run in this order:

* `uncheckedSMOOTH` - runs each method and produces the final-time plots used to choose the smoothness tolerances `TAU`.

* `ssp` - reads the output of the `uncheckedSMOOTH` run and classifies each run using the `TAU` values set in `runtests_population.py`.

The mode is given as a command-line argument; without an argument, the value of `MODE` set in `runtests_population.py` is used. `run-all-examples.sh` runs both modes one after the other, using the `TAU` values currently set in `runtests_population.py`.

The output files (Excel files and plots) are written to the directory from which the scripts are run.


### Precomputed Statistics

The `statistics` directory contains the Excel files with the statistics produced by the test scripts for each example, so that the results can be examined, and the plots regenerated, without rerunning the examples and their reference solutions:

* `statistics/linear-advection-reaction` - `linear_adv_rec_stats_fixed.xlsx` and `linear_adv_rec_stats_adaptive.xlsx`, produced by `runtests_linear_adv_rec.py`.

* `statistics/population` - `population_runs_with_bisection.xlsx`, `bisection_initial_brackets.xlsx`, and `bisection_final_brackets.xlsx`, produced by `runtests_population.py uncheckedSMOOTH`; and `population_ssp_runs_with_bisection_fixed.xlsx`, `population_ssp_runs_with_bisection_adaptive.xlsx`, and `bisection_ssp_final_brackets.xlsx`, produced by `runtests_population.py ssp`.

* `statistics/hyperbolic-relaxation` - `hyperbolic_relaxation_stats_fixed.xlsx` and `hyperbolic_relaxation_stats_adaptive.xlsx`, produced by `runtests_hyperbolic_relaxation.py`.

These files have the same names as the files written by the test scripts.

To regenerate the plots of an example from these files:

1. Copy the Excel files of the example from `statistics` into the directory from which the test script is run, e.g., `bin`.

2. In the test script, comment out the lines that run the tests and write the Excel files, so that only the plotting part is executed:

   * `runtests_linear_adv_rec.py` - from the line `# run tests and collect results as pandas data frames (one for fixed, one for adaptive)` to the line `RunStatsAdaptiveDf.to_excel(fnameAdaptive + '.xlsx', index=False)`.

   * `runtests_hyperbolic_relaxation.py` - from the line `# run tests and collect results as pandas data frames (one for fixed, one for adaptive)` to the line `RunStatsAdaptiveDf.to_excel(fnameAdaptive + '.xlsx', index=False)`.

   * `runtests_population.py` - in the `ssp` section, from the line `runs = pd.read_excel("population_runs_with_bisection.xlsx").to_dict('records')` to the line `pd.DataFrame.from_records(final_rows).to_excel("bisection_ssp_final_brackets.xlsx", index=False)`.

3. Run the test script, e.g., `python3 runtests_linear_adv_rec.py`, `python3 runtests_hyperbolic_relaxation.py`, or `python3 runtests_population.py ssp`. The reference solution scripts do not need to be run.

The final-time solution plots produced by `runtests_population.py uncheckedSMOOTH` cannot be regenerated from the Excel files, since they require the solutions from running the example.
