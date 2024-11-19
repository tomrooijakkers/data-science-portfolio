"""KNMI Meteo Transformation

This script contains the general transformation functions for
the 'raw' / ingested data returned by the KNMI web service.

This script requires that `pandas` and `numpy` be installed within 
the Python environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * load_tf_json - load 'transform' JSON to Python object
    * transform_stations - transform raw station data to cleaned df
    * transform_param_values - transform raw values to cleaned df

"""

import os
import json

import pandas as pd
import numpy as np


def load_tf_json(filename: str) -> list[dict] | dict:
    """Load 'transform' JSON to Python object."""
    with open(os.path.join("transform", filename)) as f:
        json_obj = json.load(f)
    
    return json_obj


def transform_stations(df_stn: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw station data to a cleaned Pandas DataFrame.

    Prettifies the column names of the stations DataFrame.

    Parameters
    ----------
    df_stn : pd.DataFrame
        DataFrame with raw station (column name) data.

    Returns
    -------
    pd.DataFrame
        Cleaned station (column name) DataFrame.

    """
    # Load JSON file with transformation mapping as list of dict items
    stn_tf_map = load_tf_json("transform_stations.json")

    return df_stn.rename(columns=stn_tf_map)


def transform_param_values(df: pd.DataFrame,
                           nullify_small_sumvals=False) -> pd.DataFrame:
    """
    Transform raw values to a cleaned Pandas DataFrame.

    In this function various general data cleaning steps are performed. 
    The column names are prettified, and the values are converted to whole
    units where possible.

    For parameters that can be summed, all negative values are
    converted to NaNs, nullified, or replaced by small values
    (the latter two only in case of -1).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with raw values and one param per col.
    nullify_small_values : bool
        Specify whether to convert small summable values to the
        average of its category (0.025), or to nullify them.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with parameter cols and values.
    """
    # Validate mode for correct filename link
    if "HH" not in df.columns:
        mode = "daily"
        param_filename = "transform_params_day.json"
    else:
        mode = "hourly"
        param_filename = "transform_params_hour.json"

    # Load JSON file with transformation mapping as list of dict items
    param_tf_map = load_tf_json(param_filename)

    # Placeholder for column renaming mapper
    rename_cols = {}

    for colname in df:
 
        # Find linked item on key 'parameter_code'
        map_item = next((d for d in param_tf_map
                         if d["parameter_code"] == colname), None)
        
        # Give warning and skip this col if no linked item found
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
    
    # Create date or datetime col with time
    if mode == "daily":
        df["date"] = (pd.to_datetime(df["date"], format="%Y%m%d").dt.date)

    elif mode == "hourly":
        # Ensure that hours always fill up two characters
        df["hour_filled"] = (df["hour"].astype(int) - 1).astype(str).str.zfill(2)
        df["datetime"] = (pd.to_datetime(
                            df[["date", "hour_filled"]]
                            .astype(str).sum(axis=1),
                            format="%Y%m%d%H"))

        df.drop(columns=['hour_filled'], inplace=True)

    # Update all other cols with prettified names in one go
    df.rename(columns=rename_cols, inplace=True)

    return df
