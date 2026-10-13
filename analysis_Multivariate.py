# Load packages

import warnings
import sys
import csv
from pathlib import Path

from scipy import signal
from scipy.signal import detrend
from scipy.signal import correlate
from scipy.optimize import curve_fit

import numpy as np
from numpy.fft import rfft, rfftfreq, irfft

import pandas as pd
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

from math import radians, sin, cos, sqrt, atan2

import calendar
import matplotlib.dates as mdates


plt.rcParams.update({
    'axes.titlesize': 20,       # Title font size
    'axes.labelsize': 26,       # X and Y label font size
    'xtick.labelsize': 22,      # X tick label font size
    'ytick.labelsize': 22,      # Y tick label font size
    'legend.fontsize': 22       # Legend font size
})

do_lowpass = True
do_season = True
do_spring = True

cross_corr = False
time_series = False
select_year = True
YEAR = 2017
select_month = True
MONTH = 5
MONTH_NAME = calendar.month_name[MONTH]
do_median = False

do_dmd = True
do_ssa = not do_dmd
csv_harmonics = False

csv_slopes = False
show_slopes = False

do_daily = False
show_errors = False
do_split_dmd = False
do_control_dmd = False

select_normal = False
select_mast = False
do_masting = False
csv_masting = False
yearly_medians = False
cumulative_masting = False

# Target directories
DATA_DIR = Path(r'C:\Users\gemmy\Documents\Data')
SAVE_DIR = Path(r'C:\Users\gemmy\Documents\Multivariate_DMD-SSA')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

carbon_units = ' [' + r'$\mu\mathrm{mol}\ \mathrm{CO_2}\ \mathrm{m}^{-2}\ \mathrm{s}^{-1}$' + ']'
radiation_units = ' [' + r'$\mathrm{W}\ \mathrm{m}^{-2}$' + ']'
temperature_units = ' [' + '°C' + ']'
swc_units = ' [' + '%' + ']'

''' ------------------ Functions declaration ------------------ '''

def clean(var_name):
    # Split and remove empty tokens (handles cases like '6__GPP')
    parts = [p for p in var_name.split('_') if p]

    # Drop leading numeric prefix
    if parts and parts[0].isdigit():
        parts = parts[1:]

    # Reconstruct tokens, but keep SW_IN intact
    cleaned_parts = []
    i = 0
    while i < len(parts):
        # Detect SW_IN pattern
        if i + 1 < len(parts) and parts[i] == 'SW' and parts[i+1] == 'IN':
            cleaned_parts.append('SW_IN')
            i += 2
        else:
            cleaned_parts.append(parts[i])
            i += 1

    return ', '.join(cleaned_parts)


def fft_lowpass(X0, meas_per_day, cutoff):
    '''
    FFT low-pass filter with detrending and mirroring to reduce edge artifacts.
    Works for both 1D arrays (single variable) and 2D arrays (M variables x N time steps).
    '''
    # If input is 1D, promote to 2D for uniform handling
    if X0.ndim == 1:
        X0 = X0[np.newaxis, :]

    M, N = X0.shape
    X0_new = np.zeros_like(X0, dtype=float)

    for m in range(M):
        x0 = X0[m, :]

        # Remove mean + best‑fit straight line through the data
        trend_removed = detrend(x0, type='linear')

        # Mirror the data at both ends to reduce edge discontinuities
        x0_mirrored = np.concatenate([trend_removed[N-1::-1], trend_removed, trend_removed[::-1]])

        # FFT on mirrored data
        freqs = rfftfreq(len(x0_mirrored), d=1/meas_per_day)
        mask = freqs <= cutoff

        x0_fft = rfft(x0_mirrored)
        x0_fft[~mask] = 0

        # Inverse FFT
        x0_new_mirrored = irfft(x0_fft, n=len(x0_mirrored))

        # Extract the middle segment (original)
        x0_new = x0_new_mirrored[N:2*N]

        # Restore mean
        x0_new += np.mean(x0)

        X0_new[m, :] = x0_new

    return X0_new


def butter_lowpass(X0, cutoff, nyq, order=4):
    '''
    Apply a low-pass Butterworth filter to a signal.
    '''
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff)
    X1 = signal.filtfilt(b, a, X0)
    return X1


def block_average(X0, block_size):
    '''
    Average consecutive blocks of length block_size.
    '''
    # Handle 1D input by promoting to 2D
    if X0.ndim == 1:
        X0 = X0[np.newaxis, :]

    M, N = X0.shape
    n_blocks = N // block_size
    N_trimmed = n_blocks * block_size

    # Trim to multiple of block_size
    X0_trimmed = X0[:, :N_trimmed]

    # Reshape: (M, n_blocks, block_size) and average over last axis
    X1 = X0_trimmed.reshape(M, n_blocks, block_size).mean(axis=2)

    return X1


def plot_time_series(time, X0, qc_vals, site_name, row_name, single_year=False, single_week=False, SAVE_DIR=SAVE_DIR, save=True):
    '''
    Plot time series with Quality Control.
    '''

    # Find unique QC values
    unique_qc = np.unique(qc_vals)

    # Define a color palette (extendable)
    base_colors = ['green', 'blue', 'orange', 'magenta']

    qc_colors = base_colors[:len(unique_qc)]

    new_cmap = ListedColormap(qc_colors)

    # Scatter plot
    plt.figure(figsize=(12,6))
    if single_week:
        plt.plot(time, X0, color='gray')
        sc = plt.scatter(time, X0, c=qc_vals, cmap=new_cmap, s=25, alpha=1, edgecolors='none')

    else:
        sc = plt.scatter(time, X0, c=qc_vals, cmap=new_cmap, s=4, alpha=1, edgecolors='none')

    #plt.title(f'{clean(var_name)} in {site_name}')
    
    if row_name == 'GPP':
        units = carbon_units
    elif row_name == 'SW_IN':
        units = radiation_units
    elif row_name == 'TA' or row_name == 'TS':
        units = temperature_units
    elif row_name == 'SWC':
        units = swc_units
    plt.ylabel(row_name + units)
    plt.tick_params(axis='both', which='major')

    if single_year:
        plt.xlabel(f'Year {YEAR}')
        mask = (time.year >= YEAR) & (time.year < YEAR+1)
        plt.xlim(time[mask][0], time[mask][-1])
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        if site_name == 'DE-Hai':
            if row_name == 'GPP':
                plt.ylim(-2, 50)
            if row_name == 'TA':
                plt.ylim(-18, 35)
            if row_name == 'TS':
                plt.ylim(-4, 25)
        else:
            y_slice = X0[mask]
            ymin = y_slice.min()
            ymax = y_slice.max()
            pad  = 0.05 * (ymax - ymin)
            plt.ylim(ymin - pad, ymax + pad)
        

    if single_week:
        plt.xlabel(f'{MONTH_NAME} {YEAR}')
        mask = ((time.year == YEAR) & (time.month == MONTH) & (time.day <= 5))
        start = time[mask][0]
        end = time[mask][-1]
        plt.xlim(start, end)
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d'))
        y_slice = X0[mask]
        ymin = y_slice.min()
        ymax = y_slice.max()
        pad  = 0.05 * (ymax - ymin)
        plt.ylim(ymin - pad, ymax + pad)


    # Colorbar with QC ticks
    
    cbar = plt.colorbar(sc)
    cbar.set_label('Quality Flag', fontsize=20)
    cbar.set_ticks(unique_qc)
    cbar.set_ticklabels([str(val) for val in unique_qc])
    

    plt.tight_layout()
    if save:
        if single_year:
            plt.savefig(SAVE_DIR / f'{site_name}_{row_name}_{YEAR}_series.png', dpi=300, bbox_inches='tight')
        elif single_week:
            plt.savefig(SAVE_DIR / f'{site_name}_{row_name}_{MONTH_NAME.lower()}{YEAR}_series.png', dpi=300, bbox_inches='tight')
    else: plt.show()
    plt.close()


def hankel(X0, h):
    '''
    Interleaved multivariate Hankel embedding.
    '''
    X0 = np.atleast_2d(X0)
    M, N = X0.shape
    w = N - h + 1
    if w <= 0:
        raise ValueError(f'h={h} must be between 1 and length of series (N={N})')

    # Build Hankel blocks for each variable
    blocks = []
    for m in range(M):
        Hm = np.vstack([X0[m, i:i+h] for i in range(w)])  # shape (w, h)
        blocks.append(Hm)

    # Interleave rows: [row0_var0, row0_var1, ..., row1_var0, row1_var1, ...]
    Xh = np.vstack([row for i in range(w) for row in (blocks[m][i] for m in range(M))])
    return Xh


def normalize_rows(Xh):
    '''
    Center and normalize each row of a 2D array.
    '''
    # Compute row means and stds
    row_means = Xh.mean(axis=1, keepdims=True)
    row_stds  = Xh.std(axis=1, keepdims=True)

    # Avoid division by zero: replace 0 std with 1
    row_stds[row_stds == 0] = 1.0

    # Normalize
    Xh_new = (Xh - row_means) / row_stds
    return Xh_new, row_means, row_stds


def dmd(Xh, meas_per_day, svd_rank=None, exact=True, only_harmonics=True):
    '''
    Perform Exact DMD on data matrix Xh of shape (M(N-h+1), h):
    M(N-h+1) = number of lags (spatial dimension in Hankel space),
    h = number of time snapshots in Hankel space.
    '''

    # Split into X and Xp (time-shifted data matrices)
    X = Xh[:, :-1] # (M(N-h+1), h-1)
    Xp = Xh[:, 1:] # (M(N-h+1), h-1)

    # SVD of X
    U, s, VT = np.linalg.svd(X, full_matrices=False)
    # U: (M(N-h+1), r)
    # s: (r,)
    # VT: (r, h-1)

    # Determine truncation rank
    
    if svd_rank is None or svd_rank > len(s):
        r = len(s)
    else:
        r = svd_rank

    U_r = U[:, :r] # (M(N-h+1), r)

    s_r = s[:r] # (r,)

    VT_r = VT[:r, :] # (r, h-1)
    V_r = VT_r.conj().T # (h-1, r)

    # Build reduced A_tilde
    S_inv = np.diag(1.0 / s_r)
    A_tilde = U_r.conj().T @ Xp @ V_r @ S_inv # (r, r)

    # Eigen-decomposition of A_tilde
    eigs, phi_tilde = np.linalg.eig(A_tilde)
    # eigs: (r,)
    # phi_tilde: (M(N-h+1),)

    # Modes
    if exact:
        modes = Xp @ V_r @ S_inv @ phi_tilde # (M(N-h+1), r)
    else:
        modes = U_r @ phi_tilde # (M(N-h+1), r)

    # Initial amplitudes (solves P*b=X0)
    b = np.linalg.lstsq(modes, Xh[:, 0], rcond=None)[0] # (r,)

    # Dynamics
    K = np.arange(Xh.shape[1]) # 0 ... h-1
    dynamics = np.array([b * (eigs**k) for k in K]).T # (r, h)

    # Compute DMD frequencies (signed)
    def dmd_frequency(lam):
        return np.angle(lam) / (2*np.pi)

    freqs = np.array([dmd_frequency(eig) for eig in eigs])

    # Build sorting keys
    # 1) sort by absolute frequency
    # 2) for equal |freq|, put positive freq first
    sort_keys = np.lexsort((-freqs, np.abs(freqs)))

    # Apply permutation
    eigs = eigs[sort_keys]
    modes = modes[:, sort_keys]
    b = b[sort_keys]
    dynamics = dynamics[sort_keys, :]

    # Continuous-time eigenvalues per day
    omega = np.log(eigs) * meas_per_day

    # Cycles per day: 1 repeats every day, 2 twice a day, 0.5 every 2 days, 1/365 every year
    freqs = np.imag(omega) / (2 * np.pi)

    if only_harmonics:
        har_indices = []
        for i in range(r):
            dyn = dynamics[i, :].real
            is_har, is_fund = check_harmonicity(dyn, meas_per_day)

            if is_har:
                har_indices.append(i)

        modes_har = modes[:, har_indices]
        dynamics_har = dynamics[har_indices, :]
        Xh_rec = modes_har @ dynamics_har # (M(N-h+1), h))

    else:
        Xh_rec = modes @ dynamics # (M(N-h+1), h))

    return eigs, freqs, modes, dynamics, b, Xh_rec, r


def split_dmd(Xh, Yh, svd_rank=None, exact=True):

    # Split into X and Xp (time-shifted data matrices)
    X = Xh # (M(N-h+1), h-1)
    Xp = Yh # (M(N-h+1), h-1)

    # SVD of X
    U, s, VT = np.linalg.svd(X, full_matrices=False)
    # U: (M(N-h+1), r)
    # s: (r,)
    # VT: (r, h-1)

    # Determine truncation rank
    
    if svd_rank is None or svd_rank > len(s):
        r = len(s)
    else:
        r = svd_rank

    U_r = U[:, :r] # (M(N-h+1), r)

    s_r = s[:r] # (r,)

    VT_r = VT[:r, :] # (r, h-1)
    V_r = VT_r.conj().T # (h-1, r)

    # Build reduced A_tilde
    S_inv = np.diag(1.0 / s_r)
    A_tilde = U_r.conj().T @ Xp @ V_r @ S_inv # (r, r)

    # Eigen-decomposition of A_tilde
    eigs, phi_tilde = np.linalg.eig(A_tilde)
    # eigs: (r,)
    # phi_tilde: (M(N-h+1),)

    # Modes
    if exact:
        modes = Xp @ V_r @ S_inv @ phi_tilde # (M(N-h+1), r)
    else:
        modes = U_r @ phi_tilde # (M(N-h+1), r)

    # Initial amplitudes
    b = np.linalg.lstsq(modes, Xh[:, 0], rcond=None)[0] # (r,)

    # Dynamics
    K = np.arange(Xh.shape[1]) # 0 ... h-1
    dynamics = np.array([b * (eigs**k) for k in K]).T # (r, h)

    # DMD Reconstruction
    Xh_rec = modes @ dynamics # (M(N-h+1), h)

    return eigs, modes, dynamics, Xh_rec, r


def control_dmd(Xh, Yh, svd_rank_r=None, svd_rank_w=None):
    '''
    Perform DMD with control.
    '''

    # Split in time-shifted matrices
    Xp = Xh[:, 1:]
    X = Xh[:, :-1]
    Y = Yh[:, :-1]

    # Build matrix Omega
    Omega = np.vstack((X, Y))

    # Perform SVD on Xp and on Omega
    U, s, VT = np.linalg.svd(X, full_matrices=False)
    M, l, NT = np.linalg.svd(Omega, full_matrices=False)

    # Determine truncation rank r for Xp and w for Omega
    if svd_rank_r is None or svd_rank_r > len(s):
        r = len(s)
    else:
        r = svd_rank_r
    U_r = U[:, :r]
    s_r = s[:r]
    S_inv = np.diag(1.0 / s_r)
    VT_r = VT[:r, :]
    V_r = VT_r.conj().T

    if svd_rank_w is None or svd_rank_w > len(l):
        w = len(l)
    else:
        w = svd_rank_w
    M_w = M[:, :w]
    l_w = l[:w]
    L_inv = np.diag(1.0 / l_w)
    NT_w = NT[:w, :]
    N_w = NT_w.conj().T

    # Compute the approximation of A and B (computationally prohibitive)
    MaT = M_w[:X.shape[0], :].conj().T
    MbT = M_w[X.shape[0]:, :].conj().T

    # Compute A_tilde and B_tilde
    A_tilde = U_r.conj().T @ Xp @ N_w @ L_inv @ MaT @ U_r
    B_tilde = U_r.conj().T @ Xp @ N_w @ L_inv @ MbT

    # Compute the eigenvalues and eigenvectors of A
    lam, phi_tilde = np.linalg.eig(A_tilde) # matrix phi_tilde is (r,r)
    phi = U_r @ A_tilde @ phi_tilde # matrix phi is (N-h+1,r)

    # Reconstruction:

    # Reduced X
    h = X.shape[1]
    X_r = np.zeros((r, h), dtype=complex)
    X_r[:, 0] = U_r.conj().T @ X[:, 0]

    # Iterative map
    for k in range(h-1):
        X_r[:, k+1] = A_tilde @ X_r[:, k] + B_tilde @ Y[:, k]

    # Lift back to full state
    X_rec = U_r @ X_r

    return lam, phi, X_rec, r, w


def ssa(Xh, meas_per_day, svd_rank=None, only_harmonics=True):
    '''
    Perform Singular Spectrum Analysis (SSA) on a Hankel-embedded matrix Xh.
    '''

    # SVD of the Hankel matrix
    U, s, VT = np.linalg.svd(Xh, full_matrices=False)

    # Determine truncation rank
    if svd_rank is None or svd_rank > len(s):
        r = len(s)
    else:
        r = svd_rank

    U_r = U[:, :r]   # (N-h+1, r)
    s_r = s[:r]      # (r,)
    VT_r = VT[:r, :] # (r, h)

    if only_harmonics:
        har_indices = []
        for i in range(r):
            dyn = VT_r[i, :].real
            is_har, is_fund = check_harmonicity(dyn, meas_per_day)

            if is_har:
                har_indices.append(i)

        modes_har = U_r[:, har_indices]
        dynamics_har = VT_r[har_indices, :]
        eigenvalues_har = s_r[har_indices]
        Xh_rec = modes_har @ np.diag(eigenvalues_har) @ dynamics_har # (M(N-h+1), h))

    else:
        Xh_rec = U_r @ np.diag(s_r) @ VT_r # (M(N-h+1), h))

    return s_r, U_r, VT_r, Xh_rec, r


def un_hankel(Xh, M):
    '''
    Inverse of interleaved multivariate Hankel embedding.
    '''
    Xh = np.asarray(Xh)
    total_rows, h = Xh.shape
    if total_rows % M != 0:
        raise ValueError('Number of rows in Xh must be divisible by M')

    w = total_rows // M
    N = w + h - 1
    X_rec = np.zeros((M, N), dtype=Xh.dtype)
    counts = np.zeros((M, N), dtype=int)

    for i in range(w):
        for m in range(M):
            row = Xh[i * M + m] # row i of variable m
            for j in range(h):
                X_rec[m, i + j] += row[j]
                counts[m, i + j] += 1

    # Average overlapping entries
    X_rec /= counts
    return X_rec


def plot_reconstruction(time, X0, X0_rec, site_name, var_name, SAVE_DIR=SAVE_DIR, save=True):
    '''
    Plot original vs reconstructed.
    '''
    plt.figure(figsize=(12,6))
    plt.plot(time, X0, label='Original', lw=3, alpha=0.8)
    plt.plot(time, X0_rec, label='Reconstruction', color='black', lw=3)
    #plt.scatter(time, X0-X0_rec, color='red', s=5)
    
    rel_err = float(np.linalg.norm(X0 - X0_rec) / np.linalg.norm(X0))
    if rel_err < 0.01:
        error = '< 0.01'
    elif rel_err > 1:
        error = '> 1'
    else:
        error = f'= {rel_err:.2f}'
    #plt.title(f'err {error}')
    
    
    if clean(var_name) == 'GPP':
        plt.ylabel(clean(var_name) + carbon_units)
        #plt.ylabel(f'Residuals {clean(var_name)}')
        plt.xlabel(f'{MONTH_NAME} {YEAR}')
        mask = ((time.year == YEAR) & (time.month == MONTH) & (time.day <= 5))
        start = time[mask][0]
        end = time[mask][-1]
        plt.xlim(start, end)
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d'))
        y0 = X0[mask]
        yrec = X0_rec[mask]
        ymin = min(y0.min(), yrec.min())
        ymax = max(y0.max(), yrec.max())
        pad = 0.05 * (ymax - ymin)
        plt.ylim(ymin - pad, ymax + pad)
        """
        if site_name == 'DE-Hai':
            plt.ylim(-0.15, 0.15)
        else:
            plt.ylim(-0.3, 0.3)
        """
        
    else:
        plt.ylabel(clean(var_name))

    #plt.legend()
    if save: plt.savefig(SAVE_DIR / f'{site_name}_{var_name}_reconstruction_diff.png', dpi=300, bbox_inches='tight')
    else: plt.show()
    plt.close()


def plot_my_errors(M, r, meas_per_day, X0, Xh_, row_stds, row_means, site_name, var_name, SAVE_DIR=SAVE_DIR, save=True):
    '''
    Plot the relative errors of the reconstruction as svd_rank varies
    to find r which minimizes the error.
    '''

    rel_errors = np.zeros(r)

    for i in range(1, r+1):
        _, _, _, _, _, Xh_rec_, _ = dmd(Xh_, meas_per_day, svd_rank=i)
        Xh_rec = Xh_rec_ * row_stds + row_means
        X0_rec = un_hankel(Xh_rec, M).real
        rel_errors[i-1] = np.linalg.norm(X0 - X0_rec) / np.linalg.norm(X0)

    plt.figure(figsize=(12,6))
    plt.plot(range(1, r+1), rel_errors, color='black', lw=1.5)
    plt.title(f'Rec. relative error vs SVD rank for {clean(var_name)} in {site_name}')
    plt.xlabel('SVD rank')
    plt.ylabel('Rec. relative error')
    plt.xticks(np.arange(1, r+1, step=1))
    plt.xlim(1, r)
    plt.grid(True, linestyle='--', alpha=0.5)

    best_rank = np.argmin(rel_errors) + 1
    plt.axvline(best_rank, color='red', linestyle='--', label=f'Best rank: {best_rank}')
    plt.legend()
    plt.tight_layout()

    if save: plt.savefig(SAVE_DIR / f'{site_name}_{var_name}_errors_r.png', dpi=300, bbox_inches='tight')
    else: plt.show()
    plt.close()


def plot_errors(M, r, X0, modes, dynamics, row_stds, row_means, site_name, var_name, SAVE_DIR=SAVE_DIR, save=True):
    '''
    Plot the relative errors of the reconstruction as svd_rank varies
    to find r which minimizes the error.
    '''

    rel_errors = np.zeros(r)

    for i in range(1, r+1):
        Xh_rec_ = modes[:, :i] @ dynamics[:i, :]
        X0_rec = un_hankel(Xh_rec_ * row_stds + row_means, M)
        rel_errors[i-1] = np.linalg.norm(X0 - X0_rec) / np.linalg.norm(X0)
        print(i)

    plt.figure(figsize=(12,6))
    plt.plot(range(1, r+1), rel_errors, color='black', lw=1.5)
    plt.title(f'Rec. relative error vs SVD rank for {clean(var_name)} in {site_name}')
    plt.xlabel('SVD rank')
    plt.ylabel('Rec. relative error')
    plt.xticks(np.arange(1, r+1, step=1))
    plt.xlim(1, r)
    plt.grid(True, linestyle='--', alpha=0.5)

    best_rank = np.argmin(rel_errors) + 1
    plt.axvline(best_rank, color='red', linestyle='--', label=f'Best rank: {best_rank}')
    plt.legend()
    plt.tight_layout()

    if save: plt.savefig(SAVE_DIR / f'{site_name}_{var_name}_errors_r.png', dpi=300, bbox_inches='tight')
    else: plt.show()
    plt.close()


def plot_eigenvalues_mag(r, eigs, site_name, var_name, SAVE_DIR=SAVE_DIR, save=True):
    '''
    Plot mode number vs eigenvalue magnitude (modulus).
    '''
    magnitudes = np.abs(eigs)
    magnitudes /= magnitudes.max() if magnitudes.max() != 0 else 1

    mode_numbers = np.arange(1, r+1)

    plt.figure(figsize=(8,8))
    plt.plot(mode_numbers, magnitudes, 'o-', color='black', label='|Eigenvalue|')
    #plt.title(f'Eigenvalue Magnitudes for {clean(var_name)} in {site_name}')
    plt.xlabel('Mode Number')
    plt.ylabel('Magnitude')
    plt.xticks(np.arange(1, r+1, step=1))
    plt.xlim(1, r)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()

    ax = plt.gca()
    ax.ticklabel_format(useOffset=False, style='plain', axis='y')

    if save: plt.savefig(SAVE_DIR / f'{site_name}_{var_name}_eigenvalue_mag.png', dpi=300, bbox_inches='tight')
    else: plt.show()
    plt.close()


def plot_expl_variance(r, eigs, site_name, var_name, SAVE_DIR=SAVE_DIR, save=True):
    '''
    Plot mode number vs variance explained.
    '''
    variances = (eigs**2 / (eigs**2).sum()) * 100
    cum_sum = np.cumsum(variances)
    mode_numbers = np.arange(1, r+1)

    plt.figure(figsize=(8,8))
    plt.plot(mode_numbers, cum_sum, 's--', color='gray', label='Cumulative')
    plt.plot(mode_numbers, variances, 'o-', color='blue', label='Variance explained')
    plt.xlabel('Mode Number')
    plt.ylabel('Percentage %')
    plt.xticks(np.arange(1, r+1, step=1))
    plt.xlim(1, r)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()

    ax = plt.gca()
    ax.ticklabel_format(useOffset=False, style='plain', axis='y')

    if save: plt.savefig(SAVE_DIR / f'{site_name}_{var_name}_expl_variance.png', dpi=300, bbox_inches='tight')
    else: plt.show()
    plt.close()


def plot_amplitudes(b, site_name, var_name, SAVE_DIR=SAVE_DIR, save=True):
    '''
    Plot mode number vs amplitude.
    '''
    # Normalize amplitudes so they sum to 1
    b_abs = np.abs(b)
    b_norm = b_abs / b_abs.sum()

    plt.figure(figsize=(8,8))
    plt.plot(b_norm, marker='o', color='blue')

    # X-ticks: start at 1, step by 1
    r = len(b_norm)
    plt.xticks(ticks=np.arange(r), labels=np.arange(1, r+1))

    plt.xlabel('Mode number')
    plt.ylabel('Amplitude')

    plt.grid(alpha=0.3)

    if save: plt.savefig(SAVE_DIR / f'{site_name}_{var_name}_amplitudes.png', dpi=300, bbox_inches='tight')
    else: plt.show()
    plt.close()


def plot_eigenvalues_circle(eigs, site_name, var_name, SAVE_DIR=SAVE_DIR, show_axes=True, show_unit_circle=True, save=True):
    '''
    Plot eigenvalues in the complex plane.
    '''
    # tab10 colormap
    cmap = plt.cm.tab10

    # assign colors in pairs: (0,1) color 0, (2,3) color 1, ...
    colors = [cmap((i // 2) % 10) for i in range(len(eigs))]


    plt.figure(figsize=(8, 8))
    if show_axes:
        plt.axhline(0, color='gray', lw=0.5)
        plt.axvline(0, color='gray', lw=0.5)
    if show_unit_circle:
        theta = np.linspace(0, 2*np.pi, 200)
        plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit circle')
    
    plt.scatter(eigs.real, eigs.imag, color=colors, s=50, label=f'Eigenvalues for {var_name} in {site_name}')

    for idx, lam in enumerate(eigs, start=1):
        plt.text(lam.real + 0.01, lam.imag + 0.01, str(idx), fontsize=20, color='black')

    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')
    plt.axis('equal')
    #plt.title(f'Eigenvalues for {clean(var_name)} in {site_name} (mean={np.mean(np.abs(eigs)):.2f})')
    #plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3)
    plt.grid(True)
    if save: plt.savefig(SAVE_DIR / f'{site_name}_{var_name}_eigenvalues_circle.png', dpi=300, bbox_inches='tight')
    else: plt.show()
    plt.close()


def animate_eigenvalues_circle(eigs, site_name, var_name, SAVE_DIR=SAVE_DIR, show_axes=True, show_unit_circle=True, save=True):

    fig, ax = plt.subplots(figsize=(8, 8))

    # Static background
    if show_axes:
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5)
    if show_unit_circle:
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', label='Unit circle')

    ax.set_xlabel('Real part')
    ax.set_ylabel('Imaginary part')
    ax.axis('equal')
    ax.set_title(f'Eigenvalues for {var_name} in {site_name}')
    ax.grid(True)

    plt.tight_layout()
    scat = ax.scatter([], [], color='red', s=40)

    # For each eigenvalue, find its conjugate (within tolerance)
    used = set()
    pairs = []
    for i, lam in enumerate(eigs):
        if i in used:
            continue
        conj_idx = np.where(np.isclose(eigs, np.conj(lam), atol=1e-8))[0]
        if len(conj_idx) > 0:
            j = conj_idx[0]
            if j != i:
                pairs.append([i, j])
                used.add(i); used.add(j)
                continue
        # If no conjugate found, just add the single
        pairs.append([i])
        used.add(i)

    # Each frame will reveal one pair
    def init():
        scat.set_offsets(np.empty((0, 2)))
        return scat,

    def update(frame):
        if frame == 0: # empty background frame
            scat.set_offsets(np.empty((0, 2)))
        else: # reveal pairs up to frame-1
            idxs = [idx for pair in pairs[:frame] for idx in pair]
            points = np.column_stack([eigs.real[idxs], eigs.imag[idxs]])
            scat.set_offsets(points)
        return scat,

    # note frames = len(pairs)+1 to include the empty frame
    ani = animation.FuncAnimation(fig, update, frames=len(pairs)+1, init_func=init, blit=True, repeat=False)

    if save: ani.save(SAVE_DIR / f'{site_name}_{var_name}_eigs.gif', writer=PillowWriter(fps=2), dpi=150)
    else: plt.show()
    plt.close(fig)


def plot_channel_part(M, N_h, modes, site_name, var_name, season_name, var_list, SAVE_DIR=SAVE_DIR, save=True):
    r = modes.shape[1]

    channel_strength = np.zeros((M, r))

    for m in range(M):
        block = modes[m * N_h : (m + 1) * N_h, :] # M blocks of size (N_h, r)
        channel_strength[m, :] = np.mean(np.abs(block)**2, axis=0) # (M, r)

    col_sum = channel_strength.sum(axis=0, keepdims=True)
    col_sum[col_sum == 0] = 1.0
    channel_strength_norm = channel_strength / col_sum

    plt.figure(figsize=(8,8))
    sns.heatmap(
        channel_strength_norm,
        cmap='viridis',
        xticklabels=[f'{i+1}' for i in range(r)],
        yticklabels=[f'{var}' for var in var_list]
    )
    plt.xlabel('Mode number')
    plt.ylabel('Variable')
    if save: plt.savefig(SAVE_DIR / f'{site_name}_{var_name}_channel-{season_name}.png', dpi=300, bbox_inches='tight')
    else: plt.show()
    plt.close()


def plot_dynamics(r, dynamics, freqs, site_name, var_name, SAVE_DIR=SAVE_DIR, spacing=2.5, save=True):
    '''
    Plot dynamics rows stacked vertically with ±f pairs sharing colors.
    '''
    h = dynamics.shape[1]
    t_axis = np.arange(h)

    # Assign colors by absolute frequency so ±f share the same color
    unique_freqs = np.unique(np.round(np.abs(freqs), 8))
    color_map = {uf: plt.cm.tab10(i % 10) for i, uf in enumerate(unique_freqs)}

    for part in ['real', 'imag']:
        plt.figure(figsize=(8,8))

        for i in range(r):
            if part == 'real':
                dyn = dynamics[i, :].real
            else:
                dyn = dynamics[i, :].imag

            # Normalise for visibility
            max_abs = np.max(np.abs(dyn))
            if max_abs != 0:
                dyn = dyn / max_abs

            offset = i * spacing
            color = color_map[np.round(abs(freqs[i]), 8)]
            plt.plot(t_axis, dyn + offset, color=color)

            # Horizontal zero line for this mode
            plt.axhline(offset, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        ax = plt.gca()
    
        if do_daily:
            #ax.xaxis.set_major_locator(ticker.MultipleLocator(73))
            plt.xlabel('Time [days]')
        else:
            #ax.xaxis.set_major_locator(ticker.MultipleLocator(6))
            plt.xlabel('Time [half-hours]')

        if yearly_medians:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            plt.xlabel('Time [years]')

        plt.xlim(0, h) # dyn has h points 0...h-1
        plt.yticks([i * spacing for i in range(r)], range(1, r+1))
        plt.ylabel('Mode number')
        #plt.title(f'Dynamics-{part} for {clean(var_name)} in {site_name}')
        plt.grid(axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.tight_layout()

        if save: plt.savefig(SAVE_DIR / f'{site_name}_{var_name}_dynamics_{part}.png', dpi=300, bbox_inches='tight')
        else: plt.show()
        plt.close()


def plot_fft_dynamics(r, meas_per_day, dynamics, freqs, site_name, var_name, SAVE_DIR=SAVE_DIR, spacing=2.0, save=True):
    '''
    Plot FFT magnitude spectra of dynamics rows.
    '''

    # Frequency (only positive) bins for a FFT of length h, each time step is 1/meas_per_day of a day
    h = dynamics.shape[1]
    fft_freqs = rfftfreq(h, d=1/meas_per_day) # h/2 integers: 0 to h/2 - 1

    # Assign colors by absolute frequency so ±f share the same color
    unique_freqs = np.unique(np.round(np.abs(freqs), 8)) # h/2 real numbers: ~ 0 to h/2 - 1; h -> h-1 if h odd
    color_map = {uf: plt.cm.tab10(i % 10) for i, uf in enumerate(unique_freqs)}

    plt.figure(figsize=(8,8))
    har_indices = []
    distances = []
    for i in range(r):
        # FFT of rows of dynamics
        dyn = dynamics[i, :].real
        fft_dyn = rfft(dyn)
        mag = np.abs(fft_dyn)
        mag /= np.max(mag) if np.max(mag) != 0 else 1

        # Peak FFT frequency for this row
        fft_peak = fft_freqs[np.argmax(mag)]

        # Distance between DMD freq and FFT peak freq
        dist = abs(abs(freqs[i]) - abs(fft_peak))
        distances.append(dist)

        # Harmonicity
        is_har, is_fund = check_harmonicity(dyn, meas_per_day)
        if is_har:
            har_indices.append(i)

        # Stacked plot
        offset = i * spacing
        color = color_map[np.round(abs(freqs[i]), 8)]
        plt.plot(fft_freqs, mag + offset, color=color)
        plt.axhline(offset, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        if np.any(freqs):
            linewidth_val = 3 if is_har else 2
            plt.axvline(x=abs(freqs[i]), color=color, linestyle=':', linewidth=linewidth_val, alpha=0.6)

    mean_distance = np.mean([distances[i] for i in har_indices])
    max_distance = np.max([distances[i] for i in range(len(distances))])

    if mean_distance < 0.01:
        dist = '< 0.01'
    elif mean_distance > 1:
        dist = '> 1'
    else:
        dist = f'= {mean_distance:.2f}'

    ax = plt.gca()
    
    if do_daily:
        #ax.xaxis.set_major_locator(ticker.MultipleLocator(1/365))
        #ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x*365:.0f}'))
        plt.xlabel('FFT Frequency [cycles/day]')
    else:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.xlabel('FFT Frequency [cycles/day]')
    
    if yearly_medians:
        plt.xlabel('FFT frequency [cycles/year]')

    plt.xlim(0, r/2 + 1)
    plt.ylabel('Mode number')
    y_ticks = [i * spacing for i in range(r)]
    y_labels = [rf'$\mathbf{{{i+1}}}$' if i in har_indices else f'{i+1}' for i in range(r)]
    plt.yticks(y_ticks, y_labels)
    #plt.title(f'FFT magnitude of dynamics for {clean(var_name)} in {site_name} (dist {dist}, max dist={max_distance:.2f})')
    plt.grid(axis='x', color='gray', linestyle='--', alpha=0.1)
    plt.tight_layout()

    if save: plt.savefig(SAVE_DIR / f'{site_name}_{var_name}_fft_dynamics.png', dpi=300, bbox_inches='tight')
    else: plt.show()
    plt.close()

    return har_indices, distances


def plot_fft_data(meas_per_day, freqs, X0, X0_rec, site_name, var_name, SAVE_DIR=SAVE_DIR, save=True):
    '''
    Plot the FFT power spectrum of the original time series X0
    and the reconstructed time series X0_rec.
    '''

    fft_freqs = rfftfreq(len(X0), d=1/meas_per_day)

    # FFT of original
    fft_vals_orig = rfft(X0)
    power_orig = np.abs(fft_vals_orig)**2 #/ len(X0)**2
    power_orig /= np.max(power_orig)

    # FFT of reconstructed
    fft_vals_rec = rfft(X0_rec)
    power_rec = np.abs(fft_vals_rec)**2 #/ len(X0_rec)**2
    power_rec /= np.max(power_rec)

    # Mask out tiny values for visualization
    mask = (power_orig >= 1e-8) | (power_rec >= 1e-8)
    fft_freqs = fft_freqs[mask]
    power_orig = power_orig[mask]
    power_rec = power_rec[mask]

    # Plot both spectra and DMD freqs
    plt.figure(figsize=(8,8))
    plt.plot(fft_freqs, power_orig, color='gray', alpha=0.8, marker='x', markersize=10, label='Original FFT')
    plt.plot(fft_freqs, power_rec, color='black', lw=2, label='Reconstructed FFT')
    for f in freqs:
        plt.axvline(f, color='red', linestyle=':', linewidth=2, alpha=1)


    ax = plt.gca()

    if do_daily:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1/365))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x*365:.0f}'))
        plt.xlabel('FFT Frequency [cycles/year]')
    else:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.xlabel('FFT Frequency [cycles/day]')

    if yearly_medians:
        plt.xlabel('FFT frequency [cycles/year]')

    plt.xlim(0, 8)
    plt.yscale('log')
    plt.ylabel('GPP (FFT relative power)')
    #plt.title(f'FFT power of data or/rec for {clean(var_name)} in {site_name}')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    #plt.legend()
    plt.tight_layout()

    if save: plt.savefig(SAVE_DIR / f'{site_name}_{var_name}_fft_data.png', dpi=300, bbox_inches='tight')
    else: plt.show()
    plt.close()


def cross_corr(x, y):
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    corr = correlate(x, y, mode='full', method='fft')
    N = len(x)
    lags = np.arange(-N + 1, N)
    corr /= N
    return lags, corr


def plot_cross_analysis(x, y, label_x, label_y, units_x, units_y, site_name, season_name, SAVE_DIR=SAVE_DIR, save=True):
    # Compute cross-correlation
    lags, corr = cross_corr(x, y)

    # Find lag(s) of maximum correlation
    max_idx = np.argmax(corr)
    best_lag = lags[max_idx]
    best_corr = corr[max_idx]

    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Left: x vs y
    axs[0].plot(x, y, '.', alpha=0.6)
    #axs[0].set_title(f'{label_y} vs {label_x}')
    axs[0].set_xlabel(label_x + units_x)
    axs[0].set_ylabel(label_y + units_y)

    # Right: cross-correlation
    axs[1].plot(lags, corr, color='steelblue')
    axs[1].axvline(best_lag, color='red', linestyle='--', label=f'max lag = {best_lag}')
    #axs[1].set_title('Cross-correlation')
    axs[1].set_xlabel('Lag [half-hours]')
    axs[1].set_ylabel('Correlation')
    axs[1].legend(fontsize=14)

    plt.tight_layout(w_pad=5.0)
    if save: plt.savefig(SAVE_DIR / f'{site_name}_{label_x}_{label_y}_cross_corr-{season_name}.png', dpi=300, bbox_inches='tight')
    else: plt.show()
    plt.close()

    return best_lag, best_corr


def match_fluxnet_to_sites(fluxnet_csv, species_csv, radius_km=10, tol=1e-6):
    '''
    1) Compute all tower–site pairs within radius_km.
    2) For each SITE, keep only the closest tower(s).
    3) Aggregate by FLXTOWER -> FLXTOWER,SITE1,SITE2,...
    4) Save to species_csv + '_match.csv'.
    '''
    towers = pd.read_csv(fluxnet_csv)
    sites = pd.read_csv(species_csv)

    towers.columns = towers.columns.str.upper()
    sites.columns = sites.columns.str.upper()

    rows = []

    # Compute all tower–site distances within radius
    for _, tower in towers.iterrows():
        t_name = tower['FLXTOWER']
        t_lat = tower['LATITUDE']
        t_lon = tower['LONGITUDE']

        dists = haversine(
            t_lat, t_lon,
            sites['LATITUDE'].values,
            sites['LONGITUDE'].values
        )

        for site_idx, dist in enumerate(dists):
            if dist <= radius_km:
                rows.append({
                    'FLXTOWER': t_name,
                    'SITE': sites.iloc[site_idx]['SITE'],
                    'DIST_KM': dist
                })

    if not rows:
        print('No matches within radius.')
        return

    matches = pd.DataFrame(rows)

    # SITE-CENTRIC FILTER
    # Keep only the closest tower(s) for each site
    site_min = matches.groupby('SITE')['DIST_KM'].min().reset_index()
    site_min = site_min.rename(columns={'DIST_KM': 'MIN_DIST'})

    matches = matches.merge(site_min, on='SITE', how='left')
    matches = matches[np.abs(matches['DIST_KM'] - matches['MIN_DIST']) <= tol]

    # TOWER-CENTRIC AGGREGATION
    out_rows = []
    for t_name, grp in matches.groupby('FLXTOWER'):
        sites_list = sorted(grp['SITE'].unique())
        row = {'FLXTOWER': t_name}
        for i, s in enumerate(sites_list, start=1):
            row[f'SITE{i}'] = s
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)

    out_name = species_csv.replace('.csv', '') + '_match.csv'
    out_df.to_csv(out_name, index=False)

    print(f'Saved refined match file as: {out_name}')


def extract_mast_data(match_csv, mastree_csv, species_name):
    '''
    Takes the output of the matching function (FLXTOWER,SITE1,SITE2,...)
    and extracts only YEAR, SPECIES, VARTYPE, MAX_VALUE, VALUE
    from MASTREE+ for seed data.
    Saves: match_csv.replace('_match.csv', '') + '_MAST.csv'
    '''

    # Load data
    try:
        match = pd.read_csv(match_csv)
    except FileNotFoundError:
        return
    mast = pd.read_csv(mastree_csv, encoding='latin1', low_memory=False)

    # Normalize column names
    match.columns = match.columns.str.upper()
    mast.columns = mast.columns.str.upper()

    # Keep only seed data
    mast_seed = mast[mast['VARIABLE'].str.lower().str.contains('seed|fruit|cone|crop')]

    # Keep only the target species
    mast_seed = mast_seed[mast_seed['SPECIES'] == species_name]

    # Keep only the columns you want
    mast_seed = mast_seed[[
        'SITE',
        'YEAR',
        'SPECIES',
        'VARTYPE',
        'MAX_VALUE',
        'VALUE'
    ]]

    # Identify SITE columns (SITE1, SITE2, ...)
    site_cols = [c for c in match.columns if c.startswith('SITE')]

    # Convert wide → long
    long = match.melt(
        id_vars=['FLXTOWER'],
        value_vars=site_cols,
        var_name='SITE_COL',
        value_name='SITE'
    ).dropna(subset=['SITE'])

    # Join with MASTREE+ seed data
    merged = long.merge(
        mast_seed,
        on='SITE',
        how='left'
    )

    # Reorder columns exactly as requested
    merged = merged[[
        'FLXTOWER',
        'SITE',
        'YEAR',
        'SPECIES',
        'VARTYPE',
        'MAX_VALUE',
        'VALUE'
    ]]

    # Remove rows where no mast data was found
    merged = merged.dropna(subset=['VALUE'])

    # Save output
    out_name = match_csv.replace('_match.csv', '') + '_MAST.csv'
    merged.to_csv(out_name, index=False)

    print(f'Saved mast data file as: {out_name}')


def haversine(lat1, lon1, lat2, lon2):
    '''
    Compute haversine distance between two points (in km).
    Accepts scalars or numpy arrays.
    '''
    R = 6371.0  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    d = 2 * R * np.arcsin(np.sqrt(a))

    return d


def solar_zenith(time, lat, lon):
    '''
    Compute solar zenith angle (degrees) for a pandas.Timestamp,
    latitude and longitude in degrees.
    '''

    # Convert to radians
    lat_rad = np.radians(lat)

    # Day of year
    doy = time.timetuple().tm_yday

    # Fractional hour (UTC)
    hour = time.hour + time.minute/60 + time.second/3600

    # Solar declination (Cooper 1969)

    gamma = 2 * np.pi * (doy - 1) / 365
    decl = (0.006918 
            - 0.399912*np.cos(gamma) 
            + 0.070257*np.sin(gamma)
            - 0.006758*np.cos(2*gamma)
            + 0.000907*np.sin(2*gamma)
            - 0.002697*np.cos(3*gamma)
            + 0.00148*np.sin(3*gamma))

    # Equation of time (minutes)
    
    EoT = (229.18 * (0.000075 
                    + 0.001868*np.cos(gamma)
                    - 0.032077*np.sin(gamma)
                    - 0.014615*np.cos(2*gamma)
                    - 0.040849*np.sin(2*gamma)))

    # Solar time
    solar_time = hour + (EoT + 4*lon) / 60

    # Hour angle (radians)
    hour_angle = np.radians(15 * (solar_time - 12))

    # Solar elevation
    elevation = np.degrees(
        np.arcsin(
            np.sin(lat_rad)*np.sin(decl) +
            np.cos(lat_rad)*np.cos(decl)*np.cos(hour_angle)
        )
    )

    # Zenith angle
    zenith = 90 - elevation
    return zenith


def compute_zenith_and_tau(fluxnet_csv, site_name, df_out, sw_in):

    S0 = 1361  # solar constant W/m2

    # Load metadata
    fluxnet = pd.read_csv(fluxnet_csv)

    # Select site
    row = fluxnet.loc[fluxnet['FLXTOWER'] == site_name]
    if row.empty:
        raise ValueError(f'Site {site_name} not found in metadata CSV.')

    lat = float(row['LATITUDE'].values[0])
    lon = float(row['LONGITUDE'].values[0])

    # Compute zenith for each timestamp
    zenith = np.array([solar_zenith(t, lat, lon) for t in df_out.index])

    # Compute clearness index Kt
    cosz = np.cos(np.radians(zenith))

    mask = (cosz > 0.1) & (sw_in > 0) # mask out nighttime and low sun angles

    Kt = sw_in[mask] / (S0 * cosz[mask])

    # Estimate tau as the 95th percentile of Kt
    tau = np.nanpercentile(Kt, 95)

    return zenith, tau


def check_harmonicity(signal, meas_per_day, eps=0.20, max_harmonic=10.0):
    '''
    Check if the dominant FFT peak is harmonic and if it's the fundamental.
    '''

    fft_freqs = rfftfreq(len(signal), d=1/meas_per_day)
    spectrum = np.abs(rfft(signal))
    if spectrum.max() == 0:
        return False, False
    spectrum /= spectrum.max()

    harmonic_targets = np.arange(1.0, max_harmonic + 1.0, 1.0)
    mask = (fft_freqs >= 0) & (fft_freqs <= max_harmonic)
    f_band, S_band = fft_freqs[mask], spectrum[mask]
    if f_band.size < 5:
        return False, False

    # Find dominant peak
    k0 = np.argmax(S_band)
    mu0 = f_band[k0]

    # Fit Gaussian around the peak
    neighborhood = (f_band > mu0 - 0.5) & (f_band < mu0 + 0.5)
    f_fit, S_fit = f_band[neighborhood], S_band[neighborhood]

    def gaussian(k, mu, sigma, A):
        return A * np.exp(-0.5 * ((k - mu) / sigma)**2) / (sigma * np.sqrt(2*np.pi))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            popt, _ = curve_fit(
                gaussian, f_fit, S_fit,
                p0=[mu0, 0.2, S_band[k0]],
                bounds=([0, 0.01, 0], [max_harmonic, 1.0, 2.0])
            )
            mu_k, sigma, A = popt

        except RuntimeError:
            return False, False

    mu_abs = abs(mu_k)
    harmonic_match = np.any(np.abs(mu_abs - harmonic_targets) < eps)

    width_ok = sigma < 1 # set <--------------------------------------------------------------

    prominence_ok = (A / (sigma * np.sqrt(2*np.pi))) > 5 * np.median(S_band)

    explained = np.sum(gaussian(f_band, *popt)) / np.sum(S_band)
    fit_ok = explained > 0.60 # set <------------------------------------------------------------

    is_harmonic = bool(harmonic_match and width_ok and prominence_ok and fit_ok)
    is_fundamental = is_harmonic and (abs(mu_k - 1.0) < eps)

    return is_harmonic, is_fundamental


def seasonal_halfhour(X0, time, seasons_list, season_name, use_median=False, plot_hist=False):
    # Get season masks
    season = seasons_list[season_name]

    # Filter data
    X_season = X0[:, season]
    time_season = time[season]

    # Plot histograms if requested
    if plot_hist:
        for v in range(X_season.shape[0]):
            plt.figure(figsize=(6,4))
            plt.hist(X_season[v, :], bins=50, color='steelblue', alpha=0.7)
            plt.title(f'Histogram for {season_name} — variable {v}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

    # Build DataFrame for grouping
    df = pd.DataFrame(X_season.T, index=time_season)

    # Group by half-hour position inside the season
    grouped = df.groupby([df.index.dayofyear, df.index.hour, df.index.minute])

    # Compute climatology
    if use_median:
        one_season = grouped.median().values.T
    else:
        one_season = grouped.mean().values.T

    # Integer time axis
    n_steps = one_season.shape[1]
    time_int = np.arange(1, n_steps + 1)

    return one_season, time_int


def monthly_halfhour(X0, time, months, use_median=False, plot_hist=False):
    mask = time.month.isin(months)
    X_month = X0[:, mask]
    time_month = time[mask]

    if plot_hist:
        for v in range(X_month.shape[0]):
            plt.figure(figsize=(6,4))
            plt.hist(X_month[v, :], bins=50, color='steelblue', alpha=0.7)
            plt.title(f'Histogram for month {months} — variable {v}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

    df = pd.DataFrame(X_month.T, index=time_month)
    grouped = df.groupby([df.index.day, df.index.hour, df.index.minute])

    if use_median:
        one_month = grouped.median().values.T
    else:
        one_month = grouped.mean().values.T

    n_steps = one_month.shape[1]
    time_int = np.arange(1, n_steps + 1)

    return one_month, time_int


def plot_matrix(csv_path):
    '''
    Read matrix_dict from CSV and plot a matrix with seasons on y-axis
    and sites on x-axis.
    '''
    
    # ---------------------------------------------------------
    # 1. Read CSV and rebuild matrix_dict
    # ---------------------------------------------------------
    matrix_dict = {}
    site_order = []   # preserve order of appearance

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            site = row['site']
            season = row['season']
            num = int(row['num'])
            fund = row['fund'] == 'True'
            err = float(row['err'])

            matrix_dict[(site, season)] = {
                'num': num,
                'fund': fund,
                'err': err
            }

            if site not in site_order:
                site_order.append(site)

    # ---------------------------------------------------------
    # 2. Fixed season order
    # ---------------------------------------------------------
    season_names = ['Spring', 'Summer', 'Autumn', 'Winter']
    #season_names = ['Spring_A', 'Spring_B', 'Autumn_A', 'Autumn_B']
    site_names = site_order   # keep CSV order

    n_sites = len(site_names)
    n_seasons = len(season_names)

    # ---------------------------------------------------------
    # 3. Prepare matrices
    # ---------------------------------------------------------
    color_matrix = np.empty((n_seasons, n_sites), dtype=object)
    num_matrix = np.zeros((n_seasons, n_sites), dtype=int)

    for i, season in enumerate(season_names):
        for j, site in enumerate(site_names):

            entry = matrix_dict.get((site, season), {'num': 0, 'fund': False, 'err': 1})
            num = entry['num']
            fund = entry['fund']
            err = entry['err']

            if fund:
                if err < 0.1:
                    color = "#8afa6e"
                elif 0.1 <= err < 0.2:
                    color = "#73a7d1"
                elif 0.2 <= err < 0.3:
                    color = "#f6ec64"
                elif err >= 0.3:
                    color = "#d86fd8"
            else:
                color = 'white'     # default background

            color_matrix[i, j] = color
            num_matrix[i, j] = num

    # ---------------------------------------------------------
    # 4. Plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(n_sites*1.2, n_seasons*0.8))
    plt.subplots_adjust(top=2)

    for i in range(n_seasons):
        for j in range(n_sites):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color_matrix[i, j], edgecolor='black'))
            if num_matrix[i, j] > 0:
                ax.text(j + 0.5, i + 0.5, str(num_matrix[i, j]), ha='center', va='center', fontsize=20, color='black')

    ax.set_xticks(np.arange(n_sites) + 0.5)
    ax.set_xticklabels(site_names, rotation=45, ha='right')

    ax.set_yticks(np.arange(n_seasons) + 0.5)
    ax.set_yticklabels(season_names)

    ax.set_xlim(0, n_sites)
    ax.set_ylim(0, n_seasons)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    legend_patches = [
    mpatches.Patch(facecolor='#8afa6e', edgecolor='black', label='err < 0.1'),
    mpatches.Patch(facecolor='#7a9de1', edgecolor='black', label='0.1 ≤ err < 0.2'),
    mpatches.Patch(facecolor='#f6ec64', edgecolor='black', label='0.2 ≤ err < 0.3'),
    mpatches.Patch(facecolor='#d86fd8', edgecolor='black', label='err ≥ 0.3'),
    ]

    ax.legend(
        handles=legend_patches,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.2),
        ncol=4,
        frameon=False
    )

    if 'DMD' in csv_path:
        plt.savefig(SAVE_DIR / f'summary_DMD_{YEAR}.png', dpi=300, bbox_inches='tight')
    if 'SSA' in csv_path:
        plt.savefig(SAVE_DIR / f'summary_SSA_{YEAR}.png', dpi=300, bbox_inches='tight')





def main():
    
    ''' ------------------ Load and preprocess the data ------------------ '''

    var_map = {
    'NEE':      ('NEE_VUT_REF',     'NEE_VUT_REF_QC'),
    'RECO':     ('RECO_NT_VUT_REF', 'NEE_VUT_REF_QC'),
    'GPP':      ('GPP_DT_VUT_REF',  'NEE_VUT_REF_QC'),
    'SW_IN':    ('SW_IN_F_MDS',     'SW_IN_F_MDS_QC'),
    'TA':       ('TA_F_MDS',        'TA_F_MDS_QC'),
    'TS':       ('TS_F_MDS_1',      'TS_F_MDS_1_QC'),
    'SWC':      ('SWC_F_MDS_1',     'SWC_F_MDS_1_QC'),
    'P':        ('P',               'P_F_QC')
    }

    site_var_dict = {
        # Deciduous broadleaf
        #'BE-Lcr': ['NEE', 'RECO', 'GPP', 'SW_IN', 'TA', 'TS', 'SWC'],
        'CZ-Lnz': ['NEE', 'RECO', 'GPP', 'SW_IN', 'TA', 'TS', 'SWC'],
        'DE-Hai': ['NEE', 'RECO', 'GPP', 'SW_IN', 'TA', 'TS', 'SWC'],
        'DE-HoH': ['NEE', 'RECO', 'GPP', 'SW_IN', 'TA', 'TS', 'SWC'],
        'DK-Sor': ['NEE', 'RECO', 'GPP', 'SW_IN', 'TA', 'TS', 'SWC'],
        #'IT-BFt': ['NEE', 'RECO', 'GPP', 'SW_IN', 'TA', 'TS', 'SWC'],

        # Evergreen needleleaf
        'DE-Tha': ['NEE', 'RECO', 'GPP', 'SW_IN', 'TA', 'TS', 'SWC'],
        'FI-Hyy': ['NEE', 'RECO', 'GPP', 'SW_IN', 'TA', 'TS', 'SWC'],
        'RU-Fyo': ['NEE', 'RECO', 'GPP', 'SW_IN', 'TA', 'TS', 'SWC'],
        'IT-Lav': ['NEE', 'RECO', 'GPP', 'SW_IN', 'TA', 'TS', 'SWC'],
        'CH-Dav': ['NEE', 'RECO', 'GPP', 'SW_IN', 'TA', 'TS', 'SWC'],
        'CZ-BK1': ['NEE', 'RECO', 'GPP', 'SW_IN', 'TA', 'TS', 'SWC'],
        'CZ-RAJ': ['NEE', 'RECO', 'GPP', 'SW_IN', 'TA', 'TS', 'SWC'],
        'DE-Obe': ['NEE', 'RECO', 'GPP', 'SW_IN', 'TA', 'TS', 'SWC'],
        'DE-RuW': ['NEE', 'RECO', 'GPP', 'SW_IN', 'TA', 'TS', 'SWC'],

        # Mixed forests
        'CH-Lae': ['NEE', 'RECO', 'GPP', 'SW_IN', 'TA', 'TS', 'SWC'],
        'BE-Vie': ['NEE', 'RECO', 'GPP', 'SW_IN', 'TA', 'TS', 'SWC']
    }

    
    site_var_dict = {
        'IT-Lav': ['NEE', 'RECO', 'GPP', 'SW_IN', 'TA', 'TS', 'SWC'],
        'DE-Hai': ['NEE', 'RECO', 'GPP', 'SW_IN', 'TA', 'TS', 'SWC']
    }
    
    
    if do_dmd:
        method = 'DMD'
    elif do_ssa:
        method = 'SSA'


    if show_errors:
        x_vals = (
        0.50, 0.75,
        1.00, 1.25, 1.50, 1.75,
        2.00, 2.25, 2.50, 2.75,
        3.00, 3.25, 3.50, 3.75,
        4.00, 4.25, 4.50, 4.75,
        5.00, 5.25, 5.50, 5.75
        )

        y_vals = (
        0.0783287501330079,
        0.25550228840122313,
        0.13768466654301162,
        0.18047023645942945,
        0.20737616824612648,
        0.19317253989493866,
        0.1805803862226701,
        0.19638060210897804,
        0.21021060873269154,
        0.20954172590774126,
        0.19643423305147,
        0.20639244658836978,
        0.21256148850231277,
        0.20966884272759193,
        0.20817182639334755,
        0.2131255250831214,
        0.21757004798740814,
        0.21740145275927145,
        0.2154764808400253,
        0.2181076421198976,
        0.2207035493100962,
        0.22112083410543829
        )

        plt.figure(figsize=(10,5))
        plt.plot(x_vals, y_vals)
        plt.xlabel('Years')
        plt.ylabel('Rec. relative error')

        x_vals = (
        10, 15, 20, 25, 30, 35, 40,
        45, 50, 55, 60, 65, 70, 75,
        80, 85, 90, 95, 100, 105, 110,
        115, 120, 125, 130, 135, 140,
        145, 150, 155, 160, 165, 170,
        175, 180
        )

        y_vals = (
        2.3737959486147525e-11,
        7.676555755693489e-06,
        0.003277379319282942,
        0.003756321257659053,
        0.003955260073517537,
        0.004246521917968885,
        0.005115414728550542,
        0.2839992631257904,
        190.97645648199017,
        21666.630171724137,
        6153139.606275004,
        1151068763.0077534,
        27388283674.792526,
        90653543525.80827,
        84549051963.6822,
        83513689001.87854,
        113851154520.88737,
        151522765046.57727,
        93003716851.98901,
        15833746345.436403,
        10225344515.92107,
        2453171048.686152,
        678043516.2633682,
        113275437.15946473,
        44504150.388591684,
        4134968.158091834,
        1084319.022368284,
        226989.0085549168,
        20016.612993796185,
        1198.263255776951,
        153.35491254942417,
        11.343106276257814,
        0.527299605591124,
        0.05951796340615245,
        0.0783287501330079
        )

        plt.figure(figsize=(10,5))
        plt.plot(x_vals, y_vals)
        plt.xlabel('Days')
        plt.ylabel('Rec. relative error')
        plt.show()


    #---------------------------- READ MASTING DATA ----------------------------#
    
    if csv_masting:

        # Read raw masting data

        mast = pd.read_csv(DATA_DIR / '_MASTREEplus.csv', encoding='latin1', low_memory=False)

        mast = mast[['Country','Site','Latitude','Longitude','Year', 'Species','Value','VarType','Max_value']]

        # Fix common mojibake patterns
        replacements = {
            # Mojibake from UTF‑8 mis-decoded as Latin‑1
            'Ã¶': 'ö',
            'Ã¤': 'ä',
            'Ã¼': 'ü',
            'ÃŸ': 'ß',
            'Ã–': 'Ö',
            'Ã„': 'Ä',
            'Ãœ': 'Ü',
            'Ã©': 'é',
            'Ã¨': 'è',
            'Ã ': 'à',
            'Ã¡': 'á',
            'Ã¢': 'â',
            'Ãª': 'ê',
            'Ã³': 'ó',
            'Ãº': 'ú',
            'Ã±': 'ñ',

            # Literal <U+XXXX> sequences (common in broken Excel exports)
            '<U+0092>': '’',
            '<U+0093>': '“',
            '<U+0094>': '”',
            '<U+00A0>': ' ',
            '<U+00B0>': '°',
            '<U+00E4>': 'ä',
            '<U+00F6>': 'ö',
            '<U+00FC>': 'ü',
            '<U+00DF>': 'ß',
            '<U+00C4>': 'Ä',
            '<U+00D6>': 'Ö',
            '<U+00DC>': 'Ü'
        }

        for col in ['Site', 'Country', 'Species']:
            for bad, good in replacements.items():
                mast[col] = mast[col].str.replace(bad, good, regex=False)

        countries = [
        'Belgium', 'Switzerland', 'Czechia', 'Germany', 'Finland', 'France', 'Italy'
        ]

        for species_name in ['Fagus sylvatica', 'Quercus petraea', 'Quercus robur', 'Larix decidua', 'Abies alba', 'Picea abies', 'Pinus sylvestris']:
            if species_name in ['Fagus sylvatica', 'Quercus petraea', 'Quercus robur']:
                leaf = 'B' # european beech (faggio), sessile oak (rovere), pedunculate oak (farnia)
            elif species_name == 'Larix decidua':
                # european larch (larice)
                leaf = 'ND' # needleleaf deciduous
            elif species_name in ['Abies alba', 'Picea abies', 'Pinus sylvestris']:
                # european silver fir (abete bianco), european spruce (abete rosso / peccio), european red pine (pino silvestre)
                leaf = 'NE' # needleleaf evergreen
            species_full = leaf + '_' + species_name.lower().replace(' ', '_')

            subset = (mast[(mast['Country'].isin(countries)) &
                (mast['Species'] == species_name)]
                [['Country', 'Site', 'Latitude', 'Longitude']]
                .drop_duplicates()
                .assign(
                    Latitude=lambda x: x['Latitude'].round(4),
                    Longitude=lambda x: x['Longitude'].round(4)
                )
                .sort_values(['Country', 'Site'])
            )
            subset.columns = subset.columns.str.upper()
            subset.to_csv(species_full + '.csv', sep=',', index=False)
            
            match_fluxnet_to_sites(DATA_DIR / '_FLUXNET.csv', species_full + '.csv', radius_km=10)
            
            match_full = leaf + '_' + species_name.lower().replace(' ', '_') + '_match'
            extract_mast_data(match_full + '.csv', DATA_DIR / '_MASTREEplus.csv', species_name)

        rows = []

    if do_masting:

        # Read refined masting data

        df_mast = pd.read_csv('_FULL_MAST.csv')

        results = []

        for tower, sub in df_mast.groupby('FLXTOWER'):
            mstsite = sub['MSTSITE'].iloc[0]
            species = sub['SPECIES'].iloc[0]
            vartype = sub['VARTYPE'].iloc[0]

            if vartype == 'O':
                max_val = sub['MAX_VALUE'].iloc[0]
                mast_years = sub.loc[sub['VALUE'] >= max_val - 1, 'YEAR'].astype(int).tolist()

            elif vartype == 'C':
                mean_val = sub['VALUE'].mean()
                sd_val = sub['VALUE'].std()
                threshold = mean_val + 2 * sd_val
                mast_years = sub.loc[sub['VALUE'] > threshold, 'YEAR'].astype(int).tolist()

            else:
                raise ValueError(f'Unknown VARTYPE: {vartype}')

            all_years = sub['YEAR'].astype(int).tolist()
            normal_years = sorted(list(set(all_years) - set(mast_years)))

            results.append({
                'FLXTOWER': tower,
                'MSTSITE': mstsite,
                'SPECIES': species,
                'mast_years': sorted(mast_years),
                'normal_years': normal_years
            })

        out_mast = pd.DataFrame(results)
        out_mast = out_mast.set_index('FLXTOWER')
        

    slopes = {}

    if show_slopes:
        # Settings
        csv_path = 'slopes.csv'
        v_name = 'GPP'
        col_var = f'a_{v_name}'

        # Forest type map
        forest_type = {
            # Deciduous broadleaf
            'BE-Lcr': 'deciduous',
            'CZ-Lnz': 'deciduous',
            'DE-Hai': 'deciduous',
            'DE-HoH': 'deciduous',
            'DK-Sor': 'deciduous',
            'IT-BFt': 'deciduous',

            # Evergreen needleleaf
            'DE-Tha': 'evergreen',
            'FI-Hyy': 'evergreen',
            'RU-Fyo': 'evergreen',
            'IT-Lav': 'evergreen',
            'CH-Dav': 'evergreen',
            'CZ-BK1': 'evergreen',
            'CZ-RAJ': 'evergreen',
            'DE-Obe': 'evergreen',
            'DE-RuW': 'evergreen',

            # Mixed forests
            'CH-Lae': 'mixed',
            'BE-Vie': 'mixed'
        }

        # Marker shapes by forest type
        forest_markers = {
            'deciduous': 'o',
            'evergreen': 'X',
            'mixed': 'P'
        }

        # Colors by season
        season_colors = {
            'Spring': 'hotpink',
            'Summer': 'green',
            'Autumn': 'darkorange',
            'Winter': 'blue'
        }

        # Read csv
        df = pd.read_csv(csv_path, comment='#')
        seasons = ['Spring', 'Summer', 'Autumn', 'Winter']

        plt.figure(figsize=(15, 7))

        for season in seasons:

            df_season = df[df['season'] == season]

            x = df_season['a_dist'].values
            y = df_season[col_var].values
            sites = df_season['site'].values

            for xi, yi, site in zip(x, y, sites):
                ftype = forest_type.get(site, 'mixed')
                plt.scatter(
                    xi, yi,
                    color=season_colors[season],
                    marker=forest_markers[ftype],
                    s=90,
                    label=season if site == sites[0] else None  # avoid duplicate legend entries
                )
                plt.text(xi, yi, site, fontsize=8, ha='left', va='bottom')

        plt.xlabel('a_dist')
        plt.ylabel(col_var)
        #plt.title(f'a_dist vs {col_var} across all seasons')

        plt.grid(alpha=0.3)
        plt.legend(title='Season')
        plt.tight_layout()
        plt.savefig(f'{col_var}_all_seasons.png', dpi=300, bbox_inches='tight')

        # 2×2 subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for ax, season in zip(axes, seasons):

            df_season = df[df['season'] == season]

            x = df_season['a_dist'].values
            y = df_season[col_var].values
            sites = df_season['site'].values

            for xi, yi, site in zip(x, y, sites):
                ftype = forest_type.get(site, 'mixed')
                ax.scatter(
                    xi, yi,
                    color=season_colors[season],
                    marker=forest_markers[ftype],
                    s=90
                )
                ax.text(xi, yi, site, fontsize=8, ha='left', va='bottom')

            ax.set_title(season)
            ax.set_xlabel('a_dist')
            ax.set_ylabel(col_var)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{col_var}_panel_seasons.png', dpi=300, bbox_inches='tight')

    matrix_dict = {}

    #plot_matrix(f'matrix_dict_{method}_{YEAR}.csv')
    
    for site_name, var_list in site_var_dict.items():

        var_name = f'{len(var_list)}_' + '_'.join(var_list)

        if site_name not in slopes:
            slopes[site_name] = {}

        print(f'Processing {clean(var_name)} in {site_name} with {method}...')

        #---------------------------- READ FLUXNET DATA ----------------------------#
        
        if True:

            # 0. Load and timestamp
            
            file_path = DATA_DIR / f'{site_name}.csv'
            df = pd.read_csv(file_path)

            df['TIME'] = pd.to_datetime(
                df['TIMESTAMP_START'],
                format='%Y%m%d%H%M',
                errors='coerce'
            )

            # Ensure sorted, unique timestamps
            df = df.sort_values('TIME').drop_duplicates(subset='TIME').reset_index(drop=True)

            
            # 1. Load raw series + QC handling

            series = {}
            qc_values = {}
            valid_masks = {}
            time_series = {}

            for var in var_list:

                data_col, qc_col = var_map[var]

                vals = df[data_col].astype(float).to_numpy()
                qc   = df[qc_col].astype(float).to_numpy()
                time = df['TIME'].to_numpy()

                # A. Value-based invalidity
                value_invalid = vals == -9999

                # B. Fix QC if value is valid but QC is -9999
                qc[(qc == -9999) & (~value_invalid)] = 3.0

                # C. Validity mask for this variable
                valid_mask = ~value_invalid

                series[var] = vals
                qc_values[var] = qc
                valid_masks[var] = valid_mask
                time_series[var] = time


            # 2. GLOBAL validity mask (all variables must be valid)

            mask_matrix = np.vstack([valid_masks[var] for var in var_list])
            global_valid = mask_matrix.all(axis=0)


            # 3. Apply global mask

            for var in var_list:
                series[var] = series[var][global_valid]
                qc_values[var] = qc_values[var][global_valid]
                time_series[var] = time_series[var][global_valid]


            # 4. Build aligned DataFrame (half-hourly clean data)

            time = time_series[var_list[0]]
            df_clean = pd.DataFrame(
                np.vstack([series[var] for var in var_list]).T,
                index=pd.to_datetime(time),
                columns=var_list
            )

            df_clean_qc = pd.DataFrame(
                np.vstack([qc_values[var] for var in var_list]).T,
                index=pd.to_datetime(time),
                columns=[f'{var}_QC' for var in var_list]
            )


            # 5. Choose temporal resolution

            if do_daily:
                df_out = df_clean.resample('D').mean()
                df_qc_out = df_clean_qc.resample('D').mean()

            elif not do_daily:
                df_out = df_clean.copy()
                df_qc_out = df_clean_qc.copy()

            elif yearly_medians:
                df_out = df_clean.resample('YE').mean()
                df_qc_out = df_clean_qc.resample('YE').mean()

            else:
                raise ValueError('Temporal resolution not chosen')


            # 6. Convert back to NumPy

            X_av = df_out.to_numpy().T
            QC_av = df_qc_out.to_numpy().T

            time = df_out.index
            years = time.year
            days = time.dayofyear



        ''' ------------------ Mask & Lowpass ------------------ '''

        if True:

            if do_daily:
                samples_per_day = 1
                min_days = 30
            else:
                samples_per_day = 48
                max_freq = 4
                min_days = 1/max_freq
            
            if do_lowpass:
                X_or = fft_lowpass(X_av, samples_per_day, max_freq)
            else:
                X_or = X_av

            #nyq = samples_per_day / 2
            #X_or = butter_lowpass(X_av, max_freq, nyq)

            if yearly_medians:
                samples_per_day = 1/365
                X_or = X_av


            if do_season:
                if True:
                    spring_equinox = pd.Timestamp('2000-03-20').dayofyear
                    spring_half = pd.Timestamp('2000-05-05').dayofyear # Beltane
                    summer_solstice = pd.Timestamp('2000-06-21').dayofyear
                    autumn_equinox = pd.Timestamp('2000-09-22').dayofyear
                    autumn_half = pd.Timestamp('2000-11-05').dayofyear # Samhain
                    winter_solstice = pd.Timestamp('2000-12-21').dayofyear

                    Spring = (days >= spring_equinox) & (days < summer_solstice)
                    Spring_A = (days >= spring_equinox) & (days < spring_half)
                    Spring_B = (days >= spring_half) & (days < summer_solstice)
                    Summer = (days >= summer_solstice) & (days < autumn_equinox)
                    Autumn = (days >= autumn_equinox) & (days < winter_solstice)
                    Autumn_A = (days >= autumn_equinox) & (days < autumn_half)
                    Autumn_B = (days >= autumn_half) & (days < winter_solstice)
                    Winter = (days < spring_equinox) | (days >= winter_solstice)

                Seasons_list = {
                    'Spring': Spring,
                    #'Spring_A': Spring_A,
                    #'Spring_B': Spring_B,
                    'Summer': Summer,
                    'Autumn': Autumn,
                    #'Autumn_A': Autumn_A,
                    #'Autumn_B': Autumn_B,
                    'Winter': Winter
                }

                Seasons_prefix = {
                    'Spring': 1,
                    #'Spring_A': 1,
                    #'Spring_B': 1,
                    'Summer': 2,
                    'Autumn': 3,
                    #'Autumn_A': 3,
                    #'Autumn_B': 3,
                    'Winter': 4
                }

                if do_spring:
                    Seasons_list = {
                    'Spring': Spring,
                    #'Spring_A': Spring_A,
                    #'Spring_B': Spring_B,
                    #'Summer': Summer,
                    #'Autumn': Autumn,
                    #'Autumn_A': Autumn_A,
                    #'Autumn_B': Autumn_B,
                    #'Winter': Winter
                    }
                    
                    Seasons_prefix = {
                        'Spring': 1,
                        #'Spring_A': 1,
                        #'Spring_B': 1,
                        #'Summer': 2,
                        #'Autumn': 3,
                        #'Autumn_A': 3,
                        #'Autumn_B': 3,
                        #'Winter': 4
                    }



        ''' ------------------ MAST analysis ------------------ '''

        if do_masting:

            # ANALYSIS WITH YEARLY MEDIAN

            # Compute yearly means
            unique_years = np.unique(years)

            all_medians = np.column_stack([
                np.median(X_or[:, years == y], axis=1)
                for y in unique_years
            ])

            #yearly_means = {y: all_means[:, i] for i, y in enumerate(unique_years)} # can access single year e.g. yearly_means[2000]

            # Mast vs normal years
            mast_years   = np.array(out_mast.loc[site_name, 'mast_years'], dtype=int)
            normal_years = np.array(out_mast.loc[site_name, 'normal_years'], dtype=int)
            species      = out_mast.loc[site_name, 'SPECIES']

            mask_mast   = np.isin(years, mast_years)
            mask_normal = np.isin(years, normal_years)

            mast_medians_vec   = np.median(X_or[:, mask_mast], axis=1)
            normal_medians_vec = np.median(X_or[:, mask_normal], axis=1)

            mast_medians   = {var: mast_medians_vec[i] for i, var in enumerate(var_list)}
            normal_medians = {var: normal_medians_vec[i] for i, var in enumerate(var_list)}

            if select_normal:
                X_or = X_or[:, mask_normal]
                time = time[mask_normal]
                years = time.year
            elif select_mast:
                X_or = X_or[:, mask_mast]
                time = time[mask_mast]
                years = time.year

            var_names = ['NEE', 'RECO', 'GPP', 'SW_IN', 'TA', 'TS', 'SWC', 'P']

            V = {}

            for name in var_names:
                if name in var_list:
                    idx = var_list.index(name)
                    V[name] = X_or[idx]
                else:
                    V[name] = None
            '''
            if V['SW_IN'] is not None:
                SW_IN = V['SW_IN']
                zenith, tau = compute_zenith_and_tau(DATA_DIR / '_FLUXNET.csv', site_name, df_out, SW_IN)
                S0 = 1361 # solar constant W/m2
                SW_clear = S0 * tau * np.cos(zenith)
                sunny = (SW_IN >= 0.7 * SW_clear)
                sunny = pd.Series(sunny, index=df_out.index)
                # Daily grouping
                dates = df_out.index.normalize()
                # Daily sunny mask
                mask_sunny = sunny.groupby(dates).mean() >= 0.5
            '''


            if yearly_medians:
                # Build extra row from df_mast
                df_site = df_mast[df_mast['FLXTOWER'] == site_name]

                extra_row = np.full(len(unique_years), np.nan)

                for i, y in enumerate(unique_years):
                    match = df_site[df_site['YEAR'] == y]
                    if len(match) > 0:
                        extra_row[i] = match['VALUE'].iloc[0]

                # Stack vertically to form X_full
                X_full = np.vstack([all_medians, extra_row])

                # Identify columns where the extra row is NOT NaN
                valid_cols = ~np.isnan(extra_row)

                # Filter X_full and the corresponding years
                X_full = X_full[:, valid_cols]
                years_full = unique_years[valid_cols]
                time_full = pd.to_datetime(unique_years, format='%Y')[valid_cols]
                var_list.append('MAST')

            if csv_masting:
                # Compute percentage differences
                for var in var_list:
                    nm = normal_medians[var]
                    mm = mast_medians[var]
                    pct = 100 * (mm - nm) / nm if nm != 0 else np.nan

                    rows.append([
                        site_name,
                        species,
                        var,
                        float(normal_medians[var].round(3)),
                        float(mast_medians[var].round(3)),
                        f'{pct.round(3)}%'
                    ])

            if cumulative_masting:

                all_cum_normal = []
                all_cum_mast = []

                for row in range(len(var_list)):
                    X_var = X_av[row, :]
                    v_name = var_list[row]
                    
                    plt.figure(figsize=(12,6))
                    plt.hist(X_var)
                    plt.savefig(SAVE_DIR / f'{site_name}_{v_name}_histogram.png', dpi=300, bbox_inches='tight')
                    
                    VAR_by_year = [X_var[years == y] for y in unique_years]
                    cum_by_year = [np.cumsum(arr) for arr in VAR_by_year]
                    
                    normal_years_set = set(normal_years)
                    mast_years_set   = set(mast_years)

                    cum_normal = [cum_by_year[i] for i, y in enumerate(unique_years) if y in normal_years_set]
                    cum_mast = [cum_by_year[i] for i, y in enumerate(unique_years) if y in mast_years_set]
                    all_cum_normal.append(cum_normal)
                    all_cum_mast.append(cum_mast)

                    plt.figure(figsize=(12,6))

                    for arr in cum_normal:
                        t = np.linspace(0, 1, len(arr))
                        plt.plot(t, arr, color='blue', alpha=0.3)

                    for arr in cum_mast:
                        t = np.linspace(0, 1, len(arr))
                        plt.plot(t, arr, color='red', alpha=0.3)

                    if len(cum_normal) > 0:
                        mean_normal = np.median([np.interp(np.linspace(0,1,1000), np.linspace(0,1,len(arr)), arr) for arr in cum_normal], axis=0)
                        plt.plot(np.linspace(0,1,1000), mean_normal, color='blue', linewidth=3, label='Normal (median)')

                    if len(cum_mast) > 0:
                        mean_mast = np.median([np.interp(np.linspace(0,1,1000), np.linspace(0,1,len(arr)), arr) for arr in cum_mast], axis=0)
                        plt.plot(np.linspace(0,1,1000), mean_mast, color='red', linewidth=3, label='Mast (median)')

                    plt.xlabel('Normalized year (0-1)')
                    plt.ylabel(f'Cumulative {v_name}')
                    plt.title(f'Cumulative {v_name} per year in {site_name}')
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(SAVE_DIR / f'{site_name}_{v_name}_cumulative.png', dpi=300, bbox_inches='tight')


                def median_curve(cum_list):
                    return np.median([
                        np.interp(np.linspace(0,1,1000),
                                np.linspace(0,1,len(arr)),
                                arr)
                        for arr in cum_list
                    ], axis=0)

                for row_x in [3,4]:
                    for row_y in [0,1,2]:
                        median_normal_x = median_curve(all_cum_normal[row_x])
                        median_normal_y = median_curve(all_cum_normal[row_y])

                        plt.figure(figsize=(12,6))
                        plt.plot(median_normal_x, median_normal_y)
                        plt.xlabel(f'Cumulative {var_list[row_x]} (normal)')
                        plt.ylabel(f'Cumulative {var_list[row_y]} (normal)')
                        plt.title(f'Cumulative comparison in {site_name}')
                        plt.grid(True)
                        plt.savefig(SAVE_DIR / f'{site_name}_vs_{var_list[row_y]}_{var_list[row_x]}.png', dpi=300, bbox_inches='tight')
                                
                for y_m in mast_years:

                    for row in range(len(var_list)):
                        X_var = X_av[row, :]
                        v_name = var_list[row]

                        if (y_m in unique_years) and ((y_m - 1) in unique_years):

                            # Extract the two years
                            var_mast = X_var[years == y_m]
                            var_prev = X_var[years == (y_m - 1)]

                            # Compute cumulative curves
                            cum_mast = np.cumsum(var_mast)
                            cum_prev = np.cumsum(var_prev)

                            # Normalize time axis to [0,1]
                            t_mast = np.linspace(0, 1, len(cum_mast))
                            t_prev = np.linspace(0, 1, len(cum_prev))

                            # Plot
                            plt.figure(figsize=(12,6))
                            plt.plot(t_prev, cum_prev, color='blue', label=f'{y_m-1} (previous)')
                            plt.plot(t_mast, cum_mast, color='red', label=f'{y_m} (mast)')
                            plt.title(f'Cumulative {v_name}: mast year {y_m} vs previous year {y_m-1}')
                            plt.xlabel('Normalized year (0-1)')
                            plt.ylabel(f'Cumulative {v_name}')
                            plt.grid(True)
                            plt.legend()
                            plt.tight_layout()
                            plt.savefig(SAVE_DIR / f'{site_name}_{v_name}_mast_vs_prev_{y_m}.png', dpi=300, bbox_inches='tight')



        ''' ------------------ DMD-SSA analysis ------------------ '''

        if time_series:
            SAVE_SERIES = Path(r'C:\Users\gemmy\Documents\Multivariate_DMD-SSA\time_series')
            SAVE_SERIES.mkdir(parents=True, exist_ok=True)
            for variable_row in [2, 3, 4, 5, 6]:
                if variable_row == 2:
                    row_name = 'GPP'
                elif variable_row == 3:
                    row_name = 'SW_IN'
                elif variable_row == 4:
                    row_name = 'TA'
                elif variable_row == 5:
                    row_name = 'TS'
                elif variable_row == 6:
                    row_name = 'SWC'
                plot_time_series(time, X_av[variable_row], QC_av[variable_row], site_name, row_name, single_year=True, SAVE_DIR=SAVE_SERIES)

        
        if (do_dmd or do_ssa) and do_season:

            for Season_name in Seasons_list:
                
                print(Season_name)

                if Season_name not in slopes[site_name]:
                    slopes[site_name][Season_name] = {}

                Folder = f'{Seasons_prefix[Season_name]}_{Season_name}'

                SAVE_SEASON = Path(fr'C:\Users\gemmy\Documents\Multivariate_DMD-SSA\{method}_{YEAR}') / Folder
                if not csv_harmonics:
                    SAVE_SEASON.mkdir(parents=True, exist_ok=True)

                # Set matrix to use
                
                if select_year:
                    Season = Seasons_list[Season_name]
                    X_use = X_or[:, Season]
                    time_use = time[Season]
                    mask = (time_use.year == YEAR)
                    X_use = X_use[:, mask]
                    time_use = time_use[mask]
                        
                elif select_month:
                    mask = ((time.year == YEAR) & (time.month == MONTH) & (time.day <= 10))
                    X_use = X_or[:, mask]
                    time_use = time[mask]
                
                elif do_median:
                    X_median, time_median = seasonal_halfhour(X_or, time, Seasons_list, Season_name, use_median=True, plot_hist=False)
                    X_use = X_median
                    time_use = time_median
                
                elif do_masting:
                    X_use = X_or[:, mask_normal]
                    time_use = time[mask_normal]
                    if yearly_medians:
                        X_use = X_full
                        time = time_full
                        h = N - 2
                        m, n = X_emb.shape
                        rank = min(m, n) # max rank

                else:
                    X_use = X_or
                    time_use = time

                X_use = np.atleast_2d(X_use)
                M, N = X_use.shape

                # Hankel matrix: 2 <= h <= N-2

                if do_daily:
                    h = 365
                else:
                    h = 48            

                # Embedding and selected rank
                
                X_emb = hankel(X_use, h) # (M(N-h+1), h-1)
                X_emb_, row_means, row_stds = normalize_rows(X_emb)
                X_use_ = un_hankel(X_emb_, M).real

                rank = 16
                    
                if do_dmd:
                    eigs, freqs, modes, dynamics, b, X_rec_emb_, Rank = dmd(X_emb_, samples_per_day, svd_rank=rank, only_harmonics=True)

                if do_ssa:
                    eigs, modes, dynamics, X_rec_emb_, Rank = ssa(X_emb_, samples_per_day, svd_rank=rank, only_harmonics=True)
                    freqs = np.zeros(Rank)
                
                # Reconstruction
                X_rec_ = un_hankel(X_rec_emb_, M).real
                rel_err = float(np.linalg.norm(X_use_ - X_rec_) / np.linalg.norm(X_use_))

                X_rec_emb = X_rec_emb_ * row_stds + row_means
                X_rec = un_hankel(X_rec_emb, M).real

                
                if csv_harmonics:
                    num = 0
                    fund = False
                    for i in range(Rank):
                        signal = dynamics[i, :].real
                        is_har, is_fund = check_harmonicity(signal, samples_per_day)
                        num += 1 if is_har else 0
                        if is_fund:
                            fund = True

                    matrix_dict[(site_name, Season_name)] = {
                        'num': num,
                        'fund': fund,
                        'err': rel_err
                    }
                

                # --------- PLOTS ---------
                
                #DMD: 4, 6, 8, 8, 10, 10, 12, 6, 2, 2, 2
                #SSA: 9, 2, 3, 4,  5,  7,  8, 8, 8, 9, 9

                #har_indices, distances = plot_fft_dynamics(Rank, samples_per_day, dynamics, freqs, site_name, var_name, SAVE_DIR=SAVE_SEASON)
                
                if csv_slopes:
                    plt.figure(figsize=(12,6))
                    distances = np.array(distances)
                    y = distances[::2] # every 2nd distance
                    x = np.arange(len(y))
                    plt.plot(x, y, color='black', linewidth=2)
                    plt.xlabel('Harmonic frequency [cycles/day]')
                    plt.ylabel('|DMD freq - FFT freq|')
                    plt.title('Distances of DMD frequencies from FFT frequencies')

                    # xticks: positions = x, labels = x+1 (integers)
                    plt.xticks(x, (x+1).astype(int))

                    # compute best-fit line
                    a, b = np.polyfit(x, y, 1)
                    plt.plot(x, a*x + b, color='red', linewidth=2, linestyle='--', label=f'Best fit: y = {a:.2f}x + {b:.2f}')
                    plt.legend()

                    plt.tight_layout()
                    plt.savefig(SAVE_SEASON / f'{site_name}_{var_name}_distances.png', dpi=300, bbox_inches='tight')
                    
                    slopes[site_name][Season_name]['a_dist'] = float(a)

                for row in range(len(var_list)):
                    
                    X_var_av = X_av[row, :]
                    X_var = X_use[row, :]
                    X_var_ = X_use_[row, :]
                    X_var_rec = X_rec[row, :]
                    X_var_rec_ = X_rec_[row, :]
                    v_name = var_list[row]   

                    
                    if row == 2: # only save GPP
                        """
                        # FFT ON ORIGINAL DATA
                        plot_fft_data(samples_per_day, freqs, X_var_, X_var_rec_, site_name, var_name, SAVE_DIR=SAVE_SEASON, save=True)
                        
                        # TIME SERIES
                        plt.figure(figsize=(12,6))
                        plt.plot(time_use, X_var)
                        plt.xlabel(f'{Season_name}')
                        plt.ylabel(clean(v_name))
                        #plt.title(f'Time series ({Season_name}) for {clean(v_name)} in {site_name}')
                        plt.tight_layout()
                        plt.savefig(SAVE_SEASON / f'{site_name}_{v_name}_series.png', dpi=300, bbox_inches='tight')
                        """
                        # RECONSTRUCTION
                        if Season_name == 'Spring':
                            plot_reconstruction(time_use, X_var, X_var_rec, site_name, f'{len(var_list)}__{v_name}', SAVE_DIR=SAVE_SEASON, save=True)
                    
                    if csv_slopes:
                        # Compute daily averages
                        #X_norm = (X_var - X_var.mean()) / X_var.std()
                        daily_avg = X_var.reshape(-1, samples_per_day).mean(axis=1)

                        # Day indices for fitting
                        day_idx = np.arange(len(daily_avg))

                        # X-axis in half-hours (center of each day)
                        x = day_idx * samples_per_day + samples_per_day//2

                        plt.figure(figsize=(12,6))

                        # Plot original half-hourly series
                        plt.plot(time_use, X_var, color='lightgray', linewidth=1, label='Half-hourly')

                        # Plot daily averages
                        plt.plot(x, daily_avg, 'o-', color='black', linewidth=2, label='Daily average')

                        # Best-fit line (fit on days, plot on half-hour axis)
                        a, b = np.polyfit(day_idx, daily_avg, 1)
                        y_fit = a * day_idx + b
                        plt.plot(x, y_fit, color='red', linewidth=2, linestyle='--', label=f'Best fit: y = {a:.2f}x + {b:.2f}')

                        plt.xlabel(f'{Season_name} [half-hours]')
                        plt.ylabel(clean(v_name))
                        plt.title(f'Daily averages ({Season_name}) for {clean(v_name)} in {site_name}')

                        plt.legend()
                        plt.tight_layout()

                        plt.savefig(SAVE_SEASON / f'{site_name}_{v_name}_dailyavg_series.png', dpi=300, bbox_inches='tight')
                        
                        slopes[site_name][Season_name][f'a_{v_name}'] = float(a)
                    
                """
                plot_dynamics(Rank, dynamics, freqs, site_name, var_name, SAVE_DIR=SAVE_SEASON)
                
                if do_dmd:
                    plot_eigenvalues_circle(eigs, site_name, var_name, SAVE_DIR=SAVE_SEASON)
                
                else:
                    plot_expl_variance(Rank, eigs, site_name, var_name, SAVE_DIR=SAVE_SEASON)

                    # SINE-COSINE DECOUPLING
                    k_values = list(range(0, Rank-1, 2))

                    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
                    axes = axes.flatten()

                    for idx, k in enumerate(k_values[:8]):
                        ax = axes[idx]
                        ax.plot(dynamics[k, :], dynamics[k+1, :])
                        #ax.set_title(f'k = {k}, k+1 = {k+1}')
                        ax.set_xlabel(fr'row {k+1} of $V^\top$', fontsize=22)
                        ax.set_ylabel(fr'row {k+2} of $V^\top$', fontsize=22)
                        ax.set_xticks([])
                        ax.set_yticks([])

                    # remove unused axes if r < 16
                    for ax in axes[len(k_values):]:
                        ax.axis('off')

                    plt.tight_layout()
                    plt.savefig(SAVE_DIR / f'{site_name}_sine_cosine_decoupling.png', dpi=300, bbox_inches='tight')
                
                """
                if cross_corr:
                    SAVE_CROSS_CORR = Path(r'C:\Users\gemmy\Documents\Multivariate_DMD-SSA\cross_corr')
                    SAVE_CROSS_CORR.mkdir(parents=True, exist_ok=True)
                    plot_cross_analysis(X_use[3], X_use[2], 'SW_IN', 'GPP', radiation_units, carbon_units, site_name, Season_name.lower(), SAVE_DIR=SAVE_CROSS_CORR)

                #plot_channel_part(M, N-h+1, modes, site_name, var_name, Season_name.lower(), var_list, SAVE_DIR=SAVE_SEASON)
                
                plt.close('all')
        

        if do_control_dmd:

            # Set variables to be used
            V_main = 'GPP'
            V_control = 'SW_IN'
            var_name = f'{V_main}_{V_control}'
            X_use = V[V_main]
            Y_use = V[V_control]
            X_use = np.atleast_2d(X_use)
            Y_use = np.atleast_2d(Y_use)
            M, N = X_use.shape

            # Embed in higher dimension
            if do_daily:
                h = 365
            else:
                h = 48

            X_emb = hankel(X_use, h)
            X_emb_, row_means_x, row_stds_x = normalize_rows(X_emb)

            Y_emb = hankel(Y_use, h)
            Y_emb_, row_means_y, row_stds_y = normalize_rows(Y_emb)

            # Fit DMD with control
            rank = 20
            lam, phi, X_rec_emb_, Rank_r, Rank_w, = control_dmd(X_emb_, Y_emb_, svd_rank_r=rank, svd_rank_w=rank)
            Rank = Rank_r
            X_rec_emb = X_rec_emb_ * row_stds_x + row_means_x
            X_rec = un_hankel(X_rec_emb, M).real
            rel_err = float(np.linalg.norm(X_use[:, :-1] - X_rec) / np.linalg.norm(X_use[:, :-1]))
            print(f'Rec. relative error: {rel_err}')

            # Continuous-time eigenvalues per day
            omega = np.log(lam) * samples_per_day

            # Cycles per day: 1 repeats every day, 2 twice a day, 0.5 every 2 days, 1/365 every year
            freqs = np.imag(omega) / (2 * np.pi)

            # Plots

            plot_eigenvalues_circle(lam, site_name, var_name)

            plot_dynamics(Rank, phi.T, freqs, site_name, var_name)

            plot_fft_dynamics(Rank, samples_per_day, phi.T, freqs, site_name, var_name)
            

        if do_split_dmd:

            for y_m in mast_years:

                    for row in range(len(var_list)):
                        X_var = X_or[row, :]
                        v_name = var_list[row]

                        if (y_m in unique_years) and ((y_m - 1) in unique_years):

                            # Extract the two years
                            var_x = X_var[years == (y_m - 1)]
                            var_y = X_var[years == (y_m)]

                            # Set matrix to use
                            '''
                            m, n = 50, 200
                            sigma = 0.01
                            rng = np.random.default_rng(0)
                            noise = rng.normal(loc=0.0, scale=sigma, size=(m, n))
                            X_use = rng.normal(loc=0.0, scale=1.0, size=(m, n))
                            Y_use = X_use + noise
                            '''
                            X_use = var_x
                            Y_use = var_y
                            X_use = np.atleast_2d(X_use)
                            Y_use = np.atleast_2d(Y_use)
                            M, N = X_use.shape

                            # Embed into higher dimensions using Hankel matrix: 2 <= h <= N-2

                            if do_daily:
                                h = 10
                            else:
                                h = 48
                            
                            X_emb = hankel(X_use, h) # (M(N-h+1), h-1)
                            X_emb_, row_means_x, row_stds_x = normalize_rows(X_emb) # normalize row by row

                            Y_emb = hankel(Y_use, h)
                            Y_emb_, row_means_y, row_stds_y = normalize_rows(Y_emb)

                            # Fit DMD
                            
                            rank = 16
                                
                            eigs, modes, dynamics, X_rec_emb_, Rank = split_dmd(X_emb_, Y_emb_, svd_rank=rank)
                            X_rec_emb = X_rec_emb_ * row_stds_x + row_means_x
                            X_rec = un_hankel(X_rec_emb, M).real
                            rel_err = float(np.linalg.norm(X_use - X_rec) / np.linalg.norm(X_use))
                            print(f'Rec. relative error for {v_name} in {y_m}: {rel_err}')
                            
                            # Continuous-time eigenvalues per day
                            omega = np.log(eigs) * samples_per_day

                            # Cycles per day: 1 repeats every day, 2 twice a day, 0.5 every 2 days, 1/365 every year
                            freqs = np.imag(omega) / (2 * np.pi)

                            plot_eigenvalues_circle(eigs, site_name, v_name + f'_{y_m}')

                            plot_dynamics(Rank, dynamics, freqs, site_name, v_name + f'_{y_m}')

                            plot_fft_dynamics(Rank, samples_per_day, dynamics, freqs, site_name, v_name + f'_{y_m}')

            
        print(f'Done {clean(var_name)} in {site_name} with {method}.\n')


    if csv_masting:
        with open('_mast_vs_normal_pct.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['FLXTOWER', 'SPECIES', 'VARIABLE', 'normal_mean', 'mast_mean', 'pct_difference'])
            writer.writerows(rows)

    if csv_slopes:

        # Build rows
        rows = []
        for site, seasons in slopes.items():
            for season, vars_dict in seasons.items():
                row = {'site': site, 'season': season}
                row.update(vars_dict)
                rows.append(row)

        # Collect all column names
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())

        fieldnames = ['site', 'season'] + sorted(k for k in all_keys if k not in ('site', 'season'))

        # Write CSV
        with open('slopes.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    if csv_harmonics:
        with open(f'matrix_dict_{method}_{YEAR}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['site', 'season', 'num', 'fund', 'err'])  # header

            for (site, season), entry in matrix_dict.items():
                writer.writerow([
                    site,
                    season,
                    entry['num'],
                    entry['fund'],
                    entry['err']
                ])


if __name__ == '__main__':
    main()