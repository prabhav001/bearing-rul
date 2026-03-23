# bearing-rul

## Data

Raw data files are not tracked by git. Download and place them as follows:

## IMS Bearing Dataset
- Download: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
- Place contents in: `data/raw/IMS/`

## FEMTO-ST / PRONOSTIA Dataset
- Download: https://www.femto-st.fr/en/Research-departments/AS2M/Research-groups/PHM
- Place contents in: `data/raw/FEMTO/`

## Wind Turbine Dataset
- Shared via Google Drive: (add your link here)
- Place contents in: `data/raw/wind_turbine/`

## Processed Data
- Run `src/ims_loader.py` to convert raw files → HDF5
- Output `.h5` files will appear in `data/processed/`