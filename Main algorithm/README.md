# Efficient 3D Forward Modeling of Time-Domain Electromagnetics Based on Octree Grid Local Refinement

This project provides Python scripts for three-dimensional time-domain electromagnetic (TEM) forward modeling based on octree grid local refinement. The code is built on SimPEG and discretize, extends the magnetic-flux-density forward modeling workflow, and uses a shifted-inverse Arnoldi/Krylov reduced-order method to compute off-time TEM responses.

## Features

- Build 3D octree meshes with local refinement near the surface, transmitter, anomalous bodies, and layered interfaces.
- Support multiple 3D model cases, including a single block anomaly, a layered medium with a stepped anomaly, and complex anomalies controlled by external DAT files.
- Compute the magnetic field z-component `Bz` for off-time TEM responses.
- Compute `dBz/dt` through interpolation and central differencing.
- Support grid-size filtering and computational-domain filtering experiments.

## Directory Structure

```text
Main algorithm/
├── simulation.py
├── model1-cewang.py
├── model1-cexian.py
├── model2-cewang.py
├── model2-cexian.py
├── model3-cewang.py
├── Grid Size Filtering.py
├── Compute domain filtering.py
├── data/
│   ├── Calculation Domain Filtering/
│   ├── Grid Size Filtering/
│   ├── model1/
│   ├── model2/
│   └── model3/
└── README.md
```

Notes:

- `simulation.py`: Core simulation module. It defines `SimulationTEM` and implements initial-field calculation, receiver projection, the SAI-SD solver, and adaptive subspace-dimension selection.
- `model1-cewang.py`: 3D forward modeling script for Model 1 with a surface receiver grid.
- `model1-cexian.py`: 3D forward modeling script for Model 1 with a receiver survey line.
- `model2-cewang.py`: 3D forward modeling script for Model 2 with a surface receiver grid.
- `model2-cexian.py`: 3D forward modeling script for Model 2 with a receiver survey line.
- `model3-cewang.py`: 3D forward modeling script for Model 3 with a surface receiver grid. It depends on external DAT files to define complex anomalies and mesh-refinement regions.
- `Grid Size Filtering.py`: Forward-modeling experiment for comparing different minimum grid sizes.
- `Compute domain filtering.py`: Forward-modeling experiment for comparing different computational-domain sizes.
- `data/`: Input-data and output-result directory. Only folders are listed here; files inside `data/` are not listed.

## Environment Requirements

Python 3.7 is recommended. The versions below are the versions used by the author. Other versions may work, but interface changes in SimPEG and discretize may affect compatibility.

| Package | Version used by the author |
| --- | --- |
| Python | 3.7 |
| matplotlib | 3.5.3 |
| numpy | 1.21.6 |
| scipy | 1.7.3 |
| SimPEG | 0.17.0 |
| discretize | 0.8.0 |
| pypardiso | 0.4.3 |

The scripts use `scipy` for matrix exponential and pseudo-inverse calculations, such as `scipy.linalg.expm` and `scipy.linalg.pinv`. The scipy version used by the author is 1.7.3.

The code comments note that SimPEG 0.18.0 and later may cause compatibility issues related to `MfMui`. For reproducibility, SimPEG 0.17.0 is recommended.

## Installation

Create an isolated Conda environment:

```bash
conda create -n tem-octree python=3.7
conda activate tem-octree
```

Install the dependency versions used by the author:

```bash
pip install numpy==1.21.6 scipy==1.7.3 matplotlib==3.5.3 SimPEG==0.17.0 discretize==0.8.0 pypardiso==0.4.3
```

If `pypardiso` or its underlying MKL dependencies fail to install, use Conda to handle the numerical-library dependencies first, then install the remaining Python packages.

## Quick Start

Enter the `Main algorithm` directory and run the required case script. For example:

```bash
python model1-cexian.py
```

Run a 3D forward-modeling case with a surface receiver grid:

```bash
python model1-cewang.py
```

Run the grid-size filtering experiment:

```bash
python "Grid Size Filtering.py"
```

Run the computational-domain filtering experiment:

```bash
python "Compute domain filtering.py"
```

In Windows PowerShell, you can also use:

```powershell
python .\model1-cexian.py
python ".\Grid Size Filtering.py"
```

## Model Cases

### Model 1

Model 1 is a single highly conductive block anomaly embedded in a homogeneous half-space.

- Background conductivity: `1e-2 S/m`
- Air conductivity: `1e-8 S/m`
- Anomaly conductivity: `1 S/m`
- Approximate anomaly range: `x = -40 m to 40 m`, `y = -40 m to 40 m`, `z = -130 m to -50 m`
- Transmitter: square loop of about `200 m x 200 m`, with current `10 A`
- Time samples: `1e-5 s to 1e-2 s`, with 101 logarithmically spaced points

Related scripts:

- `model1-cewang.py`: `11 x 11` surface receiver grid.
- `model1-cexian.py`: 101-point receiver line along the x direction.

### Model 2

Model 2 combines a layered anomaly and a stepped highly conductive anomaly.

- Background conductivity: approximately `1/3 * 1e-2 S/m`
- Layered-anomaly conductivity: `1e-2 S/m`
- Stepped-anomaly conductivity: `1 S/m`
- Layered-anomaly depth: approximately `-80 m to -160 m`
- Transmitter and time-sampling settings are similar to Model 1.

Related scripts:

- `model2-cewang.py`: `11 x 11` surface receiver grid.
- `model2-cexian.py`: 101-point receiver line along the y direction.

### Model 3

Model 3 uses external DAT files to define complex anomalies and local mesh-refinement regions.

- Minimum grid size: `1.25 m`
- Background conductivity: `1e-2 S/m`
- Low-conductivity anomaly: `1e-3 S/m`
- High-conductivity anomaly: `1 S/m`
- Main output directory: `data/model3/`

Note: Some DAT file paths in `model3-cewang.py` are written as local absolute paths. Before running the script, change them to valid paths on your machine. For open-source reproducibility, it is recommended to use relative paths under `data/model3/`.

## Core Algorithm

`SimulationTEM` in `simulation.py` inherits from SimPEG's `Simulation3DMagneticFluxDensity`. The main workflow is:

1. Compute the initial static magnetic field from the transmitter and conductivity model.
2. Assemble the discrete curl operator and material-parameter matrices.
3. Build a Krylov subspace using a shifted-inverse Arnoldi process.
4. Compute time stepping in the reduced subspace through matrix exponentials.
5. Project magnetic flux density to receiver locations to obtain `Bz`.
6. Apply numerical differentiation to `Bz` to obtain `dBz/dt`.

Main methods:

- `init_field`: Computes and caches the initial magnetic field.
- `rec_proj`: Builds the receiver projection matrix for the z-component magnetic field.
- `solve_mag_SAI_SD_off(t_full, tol, m, gamma)`: Solves the off-time response using a specified subspace dimension.
- `opt_SD_m(t_full, tolerance, gamma)`: Searches for a suitable subspace dimension based on the residual threshold.

## Outputs

The scripts usually generate the following files:

- `time.dat` or `*_t.dat`: Off-time sampling points.
- `*bz.dat` or `*_bz.dat`: Magnetic field z-component at receiver locations.
- `*dbz-dt.dat` or `*_dbz.dat`: Time derivative of the magnetic field z-component.
- `model_bf.dat`: Background model.
- `model_af.dat`: Conductivity model after adding anomalies.
- `*.png`: Optional mesh or model-section figures.

Most output paths are defined inside the scripts. They are usually under `data/model1/`, `data/model2/`, `data/model3/`, `data/Grid Size Filtering/`, or `data/Calculation Domain Filtering/`.

## Pre-Run Checklist

Before running the scripts, check that:

- The corresponding output folders exist under `data/`; otherwise, `np.savetxt` may fail because the directory does not exist.
- The current working directory is `Main algorithm`, so the scripts can import `simulation.py` and resolve relative paths correctly.
- `SimPEG==0.17.0` and `discretize==0.8.0` are installed.
- The external DAT file paths in `model3-cewang.py` have been changed to paths accessible on the current machine.
- If plotting is enabled, note that the scripts reference `C:\Windows\Fonts\simsun.ttc` by default on Windows. On non-Windows systems, modify the font path or disable plotting.

## Troubleshooting

### ModuleNotFoundError: No module named 'SimPEG'

The current Python environment does not have SimPEG installed. Activate the correct environment and install the version used by the author:

```bash
pip install SimPEG==0.17.0
```

### SimPEG attribute or matrix errors

If the error is related to attributes such as `MfMui`, the SimPEG version may be incompatible. Use the versions used by the author:

```bash
pip install SimPEG==0.17.0 discretize==0.8.0
```

### Error when saving results

If the error says that a path cannot be found, create the output directory specified in the script first. For example:

```bash
mkdir -p data/model1
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force .\data\model1
```

### Model 3 cannot find DAT files

`model3-cewang.py` depends on external DAT files and contains local absolute paths. Change these paths to relative paths inside the current project or to your actual data paths.

## Citation and Statement

This code is intended for research on 3D time-domain electromagnetic forward modeling. It demonstrates the use of octree-grid local refinement and reduced-order solving for transient electromagnetic response calculation.

The package versions listed above are the versions used by the author. To improve reproducibility, it is recommended to run the code with the same versions.
