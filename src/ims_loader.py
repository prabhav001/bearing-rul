"""
ims_loader.py
=============
Converts raw IMS bearing dataset (text snapshot files) into a structured HDF5 file.

Dataset:
    UCI IMS Bearing Dataset
    Source: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
    3 test runs, 4 bearings each, run-to-failure experiments

Usage:
    python src/ims_loader.py

Output:
    data/processed/ims_data.h5

HDF5 Structure:
    ims_data.h5
    ├── 1st_test/
    │   ├── B1/
    │   │   ├── signals       (2156, 20480, 2)  float32  [x-axis, y-axis]
    │   │   └── timestamps    (2156,)            string
    │   ├── B2, B3, B4 ...
    ├── 2nd_test/
    │   ├── B1/
    │   │   ├── signals       (984, 20480, 1)   float32  [x-axis only]
    │   │   └── timestamps    (984,)             string
    │   ├── B2, B3, B4 ...
    └── 3rd_test/
        └── (same structure as 2nd_test)

Attributes on each bearing group:
    fs             : int   - sampling frequency (20000 Hz)
    n_snapshots    : int   - number of snapshot files loaded
    n_channels     : int   - 2 for 1st_test, 1 for 2nd and 3rd test
    failed_bearing : bool  - whether this bearing failed during the test
    failure_mode   : str   - 'inner race', 'outer race', 'roller element', or 'none'

Known failures:
    1st_test — B3: inner race,  B4: roller element
    2nd_test — B1: outer race   ← primary working bearing (cleanest run-to-failure)
    3rd_test — B3: outer race
"""

import os
import numpy as np
import h5py

IMS_ROOT    = "data/raw/IMS"
OUTPUT_PATH = "data/processed/ims_data.h5"
FS          = 20000
N_SAMPLES   = 20480

TEST_COLS = {
    "1st_test": {"B1":[0,1], "B2":[2,3], "B3":[4,5], "B4":[6,7]},
    "2nd_test": {"B1":[0],   "B2":[1],   "B3":[2],   "B4":[3]},
    "3rd_test": {"B1":[0],   "B2":[1],   "B3":[2],   "B4":[3]},
}

TEST_NCOLS = {
    "1st_test": 8,
    "2nd_test": 4,
    "3rd_test": 4,
}

FAILURE_INFO = {
    "1st_test": {
        "B1": {"failed_bearing": False, "failure_mode": "none"},
        "B2": {"failed_bearing": False, "failure_mode": "none"},
        "B3": {"failed_bearing": True,  "failure_mode": "inner race"},
        "B4": {"failed_bearing": True,  "failure_mode": "roller element"},
    },
    "2nd_test": {
        "B1": {"failed_bearing": True,  "failure_mode": "outer race"},
        "B2": {"failed_bearing": False, "failure_mode": "none"},
        "B3": {"failed_bearing": False, "failure_mode": "none"},
        "B4": {"failed_bearing": False, "failure_mode": "none"},
    },
    "3rd_test": {
        "B1": {"failed_bearing": False, "failure_mode": "none"},
        "B2": {"failed_bearing": False, "failure_mode": "none"},
        "B3": {"failed_bearing": True,  "failure_mode": "outer race"},
        "B4": {"failed_bearing": False, "failure_mode": "none"},
    },
}

def get_sorted_files(test_dir):
    files = [f for f in os.listdir(test_dir) if not f.startswith('.')]
    files.sort()
    return files

def load_snapshot(filepath, expected_cols):
    try:
        data = np.loadtxt(filepath, dtype=np.float32)
    except Exception as e:
        print(f"  [WARN] Could not load {filepath}: {e}")
        return None

    if data.shape != (N_SAMPLES, expected_cols):
        print(f"  [WARN] Skipping {filepath} — shape {data.shape}, expected ({N_SAMPLES}, {expected_cols})")
        return None

    return data

def convert_test(h5file, test_folder):
    test_dir      = os.path.join(IMS_ROOT, test_folder)
    col_map       = TEST_COLS[test_folder]
    expected_cols = TEST_NCOLS[test_folder]
    failure_info  = FAILURE_INFO[test_folder]

    print(f"\nProcessing {test_folder} from {test_dir}")

    filenames = get_sorted_files(test_dir)
    n_files   = len(filenames)
    print(f"Found {n_files} files")

    signals    = []
    timestamps = []

    for i, fname in enumerate(filenames):
        fpath    = os.path.join(test_dir, fname)
        snapshot = load_snapshot(fpath, expected_cols)

        if snapshot is None:
            continue

        signals.append(snapshot)
        timestamps.append(fname)

        if (i + 1) % 200 == 0:
            print(f"  Loaded {i+1}/{n_files} files...")

    print(f"  {len(signals)} valid snapshots ({n_files - len(signals)} skipped)")

    # stack into one array: (n_valid, N_SAMPLES, expected_cols)
    all_signals = np.stack(signals, axis=0)

    for bearing, cols in col_map.items():
        grp_path = f"{test_folder}/{bearing}"
        grp      = h5file.create_group(grp_path)

        # extract channels for this bearing → (n_valid, N_SAMPLES, n_channels)
        bearing_signals = all_signals[:, :, cols]

        grp.create_dataset(
            "signals",
            data=bearing_signals,
            compression="gzip",
            compression_opts=4,
            chunks=(1, N_SAMPLES, len(cols))
        )

        dt = h5py.string_dtype(encoding="utf-8")
        grp.create_dataset(
            "timestamps",
            data=np.array(timestamps, dtype=object),
            dtype=dt
        )

        info = failure_info[bearing]
        grp.attrs["fs"]             = FS
        grp.attrs["n_snapshots"]    = len(signals)
        grp.attrs["n_channels"]     = len(cols)
        grp.attrs["failed_bearing"] = info["failed_bearing"]
        grp.attrs["failure_mode"]   = info["failure_mode"]

        print(f"  Written {grp_path}  shape {bearing_signals.shape}")

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with h5py.File(OUTPUT_PATH, "w") as f:
        f.attrs["dataset"]  = "IMS Bearing Dataset"
        f.attrs["fs"]       = FS
        f.attrs["n_tests"]  = 3

        for test_folder in ["1st_test", "2nd_test", "3rd_test"]:
            convert_test(f, test_folder)

    print(f"\nDone. Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()