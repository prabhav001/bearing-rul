"""
src/features.py
===============
Extracts 18 features per snapshot from IMS bearing signals following Wang (2012).

Features:
    Time domain        : Max, RMS, Kurtosis (3)
    Frequency energy   : E_N, E_BPFO, E_BPFI, E_BSF, E_FTF (5)
    Envelope Band 1    : same 5 energy features on 600-2000 Hz envelope
    Envelope Band 2    : same 5 energy features on 3000-5000 Hz envelope

Input:
    signals            : numpy array (n_snapshots, 20480, n_channels)

Output:
    features           : pandas DataFrame (n_snapshots, 18)
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert

# ── Bearing & signal parameters ───────────────────────────────
FS        = 20480       # sampling frequency (Hz)
N_SAMPLES = 20480       # samples per snapshot

# ── Shaft and fault frequencies ───────────────────────────────
SHAFT_HZ = 2000 / 60    # 33.33 Hz

D     = 2.815
d     = 0.331
n     = 16
alpha = np.radians(15.17)
ratio = (d / D) * np.cos(alpha)

FTF  = (SHAFT_HZ / 2)      * (1 - ratio)
BPFO = (n / 2) * SHAFT_HZ  * (1 - ratio)
BPFI = (n / 2) * SHAFT_HZ  * (1 + ratio)
BSF  = (D / (2*d)) * SHAFT_HZ * (1 - ratio**2)

FAULT_FREQS = {
    "N"    : SHAFT_HZ,
    "BPFO" : BPFO,
    "BPFI" : BPFI,
    "BSF"  : BSF,
    "FTF"  : FTF,
}

# ── Envelope filter bands ─────────────────────────────────────
ENVELOPE_BANDS = [
    {"name": "band1", "low": 600,  "high": 2000},
    {"name": "band2", "low": 3000, "high": 5000},
]

# ── Smoothing ─────────────────────────────────────────────────
ALPHA_SMOOTH = 0.1      # exponential smoothing factor

# ── FPT ───────────────────────────────────────────────────────
#FPT_IDX = 530           # first prediction time — from EDA

def compute_fft(signal, fs=FS):
    """
    Compute FFT of signal.
    Returns positive frequencies and their magnitudes.
    
    Returns:
        freqs : (N//2,) array of frequencies in Hz
        mag   : (N//2,) array of magnitudes (NOT squared)
    """
    N     = len(signal)
    freqs = fftfreq(N, d=1/fs)[:N//2]
    mag   = np.abs(fft(signal))[:N//2]
    return freqs, mag

def find_nearest_bin(freqs, target_freq):
    """
    Find index of FFT bin closest to target_freq.
    
    Args:
        freqs       : frequency array from compute_fft
        target_freq : target frequency in Hz
    
    Returns:
        index of nearest bin
    """
    return np.argmin(np.abs(freqs - target_freq))

def normalized_energy(mag, freqs, target_freqs):
    """
    Compute normalized energy at a set of target frequencies.
    
    E = Σ(A_f²) / Σ(A_F²)   for F > 0
    
    Args:
        mag          : FFT magnitude array (N//2,)
        freqs        : frequency array (N//2,)
        target_freqs : list of frequencies to sum energy over
    
    Returns:
        scalar normalized energy value
    """
    # total power — exclude DC bin (index 0)
    total_power = np.sum(mag[1:] ** 2)

    if total_power == 0:
        return 0.0

    # sum power at each target frequency
    numerator = 0.0
    for f in target_freqs:
        idx        = find_nearest_bin(freqs, f)
        numerator += mag[idx] ** 2

    return numerator / total_power

def bandpass_filter(signal, low, high, fs=FS, order=4):
    """
    Butterworth bandpass filter.
    
    Args:
        signal : 1D numpy array
        low    : lower cutoff frequency (Hz)
        high   : upper cutoff frequency (Hz)
        fs     : sampling frequency (Hz)
        order  : filter order
    
    Returns:
        filtered signal (same shape as input)
    """
    nyq  = fs / 2
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, signal)

def compute_envelope(signal, low, high, fs=FS):
    """
    Compute envelope of signal after bandpass filtering.

    Steps:
        1. Bandpass filter around resonance band
        2. Hilbert transform → analytic signal
        3. Magnitude → envelope

    Args:
        signal : 1D numpy array
        low    : bandpass lower cutoff (Hz)
        high   : bandpass upper cutoff (Hz)
        fs     : sampling frequency

    Returns:
        envelope signal (same shape as input)
    """
    filtered  = bandpass_filter(signal, low, high, fs)
    analytic  = hilbert(filtered)
    envelope  = np.abs(analytic)
    return envelope

def extract_features_single(signal, fs=FS):
    """
    Extract 18 features from a single snapshot signal.

    Args:
        signal : 1D numpy array of shape (N_SAMPLES,)
        fs     : sampling frequency

    Returns:
        dict of 18 features
    """
    features = {}

    # ── 1. Time domain features ───────────────────────────────
    features["max"]      = np.max(np.abs(signal))
    features["rms"]      = np.sqrt(np.mean(signal ** 2))
    features["kurtosis"] = kurtosis(signal, fisher=True)

    # ── 2. Raw FFT features ───────────────────────────────────
    freqs, mag = compute_fft(signal, fs)

    for name, freq in FAULT_FREQS.items():
        harmonics             = [1*freq, 2*freq, 3*freq, 4*freq, 5*freq]
        features[f"E_{name}"] = normalized_energy(mag, freqs, harmonics)

    # ── 3. Envelope features — Band 1 and Band 2 ─────────────
    for band in ENVELOPE_BANDS:
        env             = compute_envelope(signal, band["low"], band["high"], fs)
        freqs_env, mag_env = compute_fft(env, fs)

        for name, freq in FAULT_FREQS.items():
            harmonics                             = [1*freq, 2*freq, 3*freq, 4*freq, 5*freq]
            features[f"E_{name}_{band['name']}"]  = normalized_energy(mag_env, freqs_env, harmonics)

    return features

def extract_features_all(signals, fs=FS):
    """
    Extract features for all snapshots.
    Automatically handles 1 or 2 channel bearings.

    For 1 channel  → 18 features
    For 2 channels → 36 features (ch0 features + ch1 features)

    Args:
        signals : numpy array (n_snapshots, N_SAMPLES, n_channels)

    Returns:
        DataFrame (n_snapshots, 18 or 36) — unsmoothed
    """
    n_snapshots = signals.shape[0]
    n_channels  = signals.shape[2]
    rows        = []

    for i in range(n_snapshots):
        row = {}

        for ch in range(n_channels):
            signal        = signals[i, :, ch]
            ch_features   = extract_features_single(signal, fs)

            if n_channels == 1:
                # no suffix needed — keeps column names clean
                row.update(ch_features)
            else:
                # suffix _ch0, _ch1 to distinguish channels
                for key, val in ch_features.items():
                    row[f"{key}_ch{ch}"] = val

        rows.append(row)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_snapshots} snapshots...")

    return pd.DataFrame(rows)

# Equivalent to your function
def smooth_features(df, alpha=ALPHA_SMOOTH):
    return df.ewm(alpha=alpha, adjust=False).mean()

def run(test_folder, bearing, output_path):
    """
    Full pipeline for one bearing:
        load signals → extract features → smooth → save to CSV
    """
    import h5py

    H5_PATH = "./data/processed/ims_data.h5"
    output_path = f"{output_path}/features_ims_{test_folder}_{bearing}.csv"
    print(f"\nLoading {test_folder}/{bearing} from {H5_PATH}")
    with h5py.File(H5_PATH, "r") as f:
        grp        = f[f"{test_folder}/{bearing}"]
        signals    = grp["signals"][:]
        n_channels = grp.attrs["n_channels"]
        print(f"  signals shape : {signals.shape}")
        print(f"  n_channels    : {n_channels}")
        print(f"  failed        : {grp.attrs['failed_bearing']}")
        print(f"  failure mode  : {grp.attrs['failure_mode']}")

    n_features = 24 * n_channels
    print(f"\nExtracting {n_features} features ({n_channels} channel(s))...")
    df_raw     = extract_features_all(signals)
    print(f"  raw features shape    : {df_raw.shape}")

    print(f"\nSmoothing features (alpha={ALPHA_SMOOTH})...")
    df_smoothed = smooth_features(df_raw, alpha=ALPHA_SMOOTH)

    print(f"\nSaving to {output_path}")
    df_smoothed.to_csv(output_path, index=False)
    print(f"  Done.")

    return df_smoothed


if __name__ == "__main__":
    run(
        test_folder = "1st_test",
        bearing     = "B4",
        output_path = "./data/processed"
    )
