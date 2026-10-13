# analysis-multivariate-SSA-DMD

This repository contains the Python code used for the Master's thesis "Detecting Daily Oscillations in Eddy Covariance Time Series". The script implements and applies Multivariate Singular Spectrum Analysis (SSA) and Dynamic Mode Decomposition (DMD) to analyze environmental time series data from eddy covariance flux towers.

## Overview

The primary script, `analysis_Multivariate.py`, is a comprehensive tool for time series analysis. It is designed to process data from FLUXNET sites, perform spectral decomposition, and generate a wide array of visualizations to interpret the results. The analysis can be configured to focus on specific time periods (years, seasons, months), compare different environmental variables, and investigate specific ecological phenomena like masting years.

## Features

*   **Time Series Decomposition:** Implements both Dynamic Mode Decomposition (DMD) and Singular Spectrum Analysis (SSA) for multivariate time series.
*   **Data Preprocessing:** Includes utilities for cleaning, low-pass filtering (FFT and Butterworth), and temporal aggregation (e.g., daily or seasonal means).
*   **Configurable Analysis:** A large set of global flags allows for easy configuration of the analysis pipeline—select the method (DMD/SSA), time range, variables, and specific analyses to run.
*   **Extensive Visualization:** Generates a rich set of plots using Matplotlib and Seaborn, including:
    *   Original vs. reconstructed time series.
    *   DMD eigenvalues in the complex plane.
    *   Temporal dynamics and FFT spectra of identified modes.
    *   Channel participation heatmaps.
    *   Cross-correlation analysis.
*   **Specialized Ecological Analysis:** Contains functions to correlate flux tower data with masting events from the MASTREE+ dataset.

## Dependencies

The script is written in Python 3 and requires the following libraries:

*   `numpy`
*   `scipy`
*   `pandas`
*   `seaborn`
*   `matplotlib`

You can install them using pip:
```bash
pip install numpy scipy pandas seaborn matplotlib
```

## Setup and Usage

### 1. Data Preparation

The script expects a specific directory structure and data files.

1.  **Data Directory:** Create a data directory. By default, the script looks for `C:\Users\gemmy\Documents\Data`. You must edit the `DATA_DIR` variable in `analysis_Multivariate.py` to point to your data folder.

    ```python
    # analysis_Multivariate.py
    DATA_DIR = Path(r'/path/to/your/data')
    ```

2.  **Input Files:** Place your time series data inside the `DATA_DIR`. The script is designed to work with `.csv` files from the FLUXNET database, where each file is named after a site (e.g., `DE-Hai.csv`). It also uses metadata and masting data files:
    *   `_FLUXNET.csv`: Metadata for flux tower sites (latitude, longitude, etc.).
    *   `_MASTREEplus.csv`: Masting data for masting-related analysis.

### 2. Configuration

All analysis parameters are controlled by global variables at the top of the `analysis_Multivariate.py` script. Before running, modify these flags to define the desired analysis.

**Key Configuration Variables:**

*   `do_dmd` / `do_ssa`: Set one to `True` to choose the analysis method.
*   `select_year` / `YEAR`: Set to `True` and specify the `YEAR` to analyze a single year.
*   `select_month` / `MONTH`: Set to `True` and specify the `MONTH` to analyze a single month.
*   `do_season`: Set to `True` to perform analysis on climatological seasons (Spring, Summer, etc.).
*   `site_var_dict`: A dictionary to define which sites and which variables to process. The available variables are mapped in the `var_map` dictionary.
*   `DATA_DIR` / `SAVE_DIR`: **Must be updated** to your local paths for data input and plot/result output.

### 3. Running the Analysis

Once the data is in place and the configuration is set, run the script from your terminal:

```bash
python analysis_Multivariate.py
```

The script will process the configured sites and variables, and all generated plots and CSV outputs will be saved to the directory specified by the `SAVE_DIR` variable.

## Methodology

The analysis pipeline proceeds through the following high-level steps:

1.  **Data Loading & Preprocessing:** Loads `.csv` files for specified FLUXNET sites. The data is cleaned, missing values are handled, and an optional FFT-based low-pass filter is applied to remove high-frequency noise.
2.  **Temporal Selection:** The dataset is sliced to match the configured time window (e.g., a specific year, a specific season across all years). The script can also compute climatologies, such as a typical seasonal cycle averaged over many years.
3.  **Hankel Embedding:** The 1D multivariate time series is transformed into a higher-dimensional trajectory matrix (a Block Hankel matrix). This step is crucial for applying spectral methods like SSA and DMD to time series data.
4.  **Decomposition (DMD/SSA):** The core analysis is performed on the trajectory matrix to decompose the signal into a set of modes, each associated with a specific frequency and growth/decay rate.
    *   **SSA** uses Singular Value Decomposition (SVD) to find orthonormal basis vectors (modes) that capture the variance in the data.
    *   **DMD** finds a best-fit linear operator that advances the system in time, yielding modes that are not necessarily orthogonal but have pure temporal frequencies.
5.  **Reconstruction & Visualization:** The original signal can be reconstructed using a subset of the identified modes. The script then generates a suite of visualizations to analyze the modes, their frequencies, their temporal evolution, and the overall accuracy of the reconstruction.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
