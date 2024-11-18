"""KNMI Meteo Transformation

This script makes use of TODO:

This script requires that `pandas` and `numpy` be installed within 
the Python environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * TODO: FILL FUNCTIONS

"""

import os
import json

import pandas as pd
import numpy as np

from knmi_mode_validation import validate_mode


def load_tf_json(filename: str) -> list[dict] | dict:
    """Load 'transform' JSON to Python object."""
    with open(os.path.join("transform", filename)) as f:
        json_obj = json.load(f)
    
    return json_obj


def transform_param_values(df: pd.DataFrame, mode="day",
                           nullify_small_sumvals=False) -> pd.DataFrame:
    """TODO: Create text here."""

    # Validate mode for correct filename link
    mode = validate_mode(mode)

    if mode == "daily":
        param_filename = "transform_params_day.json"
    elif mode == "hourly":
        param_filename = "transform_params_hour.json"
        raise NotImplementedError("Configure 'transform_params_hour.json' before proceeding.")

    # Load JSON file with transformation mapping as list of dict items
    param_tf_map = load_tf_json(param_filename)

    # Placeholder for column renaming mapper
    rename_cols = {}

    for colname in df:
 
        # Find linked item on key 'parameter_code'
        map_item = next((d for d in param_tf_map
                         if d["parameter_code"] == colname), None)
        
        # Give warning, skip this col if no linked item found
        if not map_item:
            if colname not in ["STN", "YYYYMMDD", "HH"]:
                print(f"Warning: no transform dict item found for colname "
                      f"'{colname}' in corresponding file '{param_filename}'"
                      f" - transform(s) for this col skipped.")
            continue

        # Assign item to column rename mapping dictionary (done later)
        rename_cols[colname] = map_item["parameter_name"]

        # Additional transformations for summable parameters
        if map_item["is_summable"]:

            # Change all '-1' observations to 0 or 0.25 (indicates cat. 0 - 0.5)
            if nullify_small_sumvals:
                df[colname] = df[colname].mask(df[colname] == -1, 0.)
            else:
                df[colname] = df[colname].mask(df[colname] == -1, 0.25)

            # Change leftover negative values to NaNs (summables have to be >= 0)
            df[colname] = df[colname].mask(df[colname] < 0, np.NaN)

        # Multiply the column by its 'parameter_unit' to get base units
        df[colname] *= map_item["parameter_mult"]

    # Reformat STN, YYYYMMDD and possible HH columns
    df.rename(columns={"STN": "station_code", "YYYYMMDD": "date",
                       "HH": "hour"}, inplace=True)
    # Update all columns with prettified names in one go
    df.rename(columns=rename_cols, inplace=True)

    return df


# TODO: Update stations transform to be part of this script, using 'transform_stations.json' as colmapper
    #stations = (df_stations.rename(columns=col_map)
    #                       .to_dict(orient="records"))

