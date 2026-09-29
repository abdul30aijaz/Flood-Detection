# Real-Time Flood Detection

A college mini-project for the Building Enterprise Applications subject. The planned workflow uses K-Means to add spatial risk patterns, XGBoost to estimate flood probability, and Streamlit to serve predictions.

## Project Status

Stage 1 establishes the project skeleton and model feature contract. No dataset has been downloaded or generated, and model training has not started.

## Structure

- `data/raw/`: unchanged source files downloaded by the project author
- `data/processed/`: cleaned and feature-engineered training data
- `notebooks/`: preparation, clustering, classification, and evaluation work
- `models/feature_columns.json`: ordered features expected by training and serving
- `requirements.txt`: lightweight app dependencies only
- `requirements-training.txt`: app dependencies plus notebook/GIS tools

## Feature Contract

The final model input order is stored in `models/feature_columns.json`:

`rainfall`, `soil_moisture`, `humidity`, `ndvi`, `ndwi`, `elevation`, `slope`, `temperature`, `cluster_label`.

`cluster_label` is derived by the K-Means notebook; it should not be mistaken for a raw input field. The dataset's flood label and its meaning (for example, observed flood occurrence) still need to be confirmed before preparation code is written.

## Data Needed Before Stage 2

Use real, documented observations with a defensible flood outcome label. To create the requested feature set, the data will need compatible observations for rainfall, soil moisture, humidity, NDVI, NDWI, elevation, slope, and temperature. Record the source, license, units, dates, and location/coordinate reference for each file. Where sources differ, their spatial and temporal alignment must be addressed in the preparation notebook.

Do not substitute a generic flood-prediction CSV unless its data dictionary confirms the required predictors and label. Many datasets with "flood prediction" in the name use a different feature set and cannot support this project as specified.

After selecting and downloading the real source file(s), keep the original files unchanged in `data/raw/` and note their filenames and source links. No specific download link is prescribed yet because the source and flood-label definition have not been selected; choosing one without checking its columns and license could make the later model invalid.

## Dependencies

For the app environment, install `requirements.txt`. Install `requirements-training.txt` only in the separate training environment. GDAL installation can depend on the operating system and Python version; if pip cannot install it on Windows, use a compatible Conda package rather than adding GIS libraries to the app environment.

## Planned Risk Bands

The initial probability bands are Low below 0.33, Medium from 0.33 through 0.66, and High above 0.66. These are provisional and should be reviewed against validation probabilities and the confirmed flood-label definition during evaluation.
