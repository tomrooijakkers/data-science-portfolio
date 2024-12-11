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
    * knmi_hourslot_percentage_df - aggregate KNMI data to hour slots
    * get_multiyr_hourslot_percentages - make multi-year hour slot df
"""

import os
import json
import datetime

import pandas as pd
import numpy as np

from functools import reduce

import knmi_meteo_ingest


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


def knmi_hourslot_percentage_df(start_date: datetime.date, 
                                end_date: datetime.date,
                                in_season: bool = False,
                                param_col: str = "max_rain_hour_sum",
                                hourslot_col: str = "hour_slot_max_rain_hour_sum",
                                param_min_cutoff_val: float = 0.1,
                                max_station_na_frac: float = 0.1,
                                return_as_counts: bool = False) -> pd.DataFrame:
    """
    Convert KNMI data to hour slot occurrences as percentages.

    This function fetches meteorological data from the KNMI web service, 
    applies transformations, and processes the data to calculate the percentage
    occurrence of specified parameters across hourly slots for each station in 
    a given date range. 
    
    The function returns a DataFrame where the index represent station codes,
    and columns represent the hour slots, with each value indicating the 
    percentage of occurrences for that particular hour slot.

    Parameters
    ----------
    start_date : datetime.date
        The start date for the data range to fetch from KNMI.
    end_date : datetime.date
        The end date for the data range to fetch from KNMI.
    in_season : bool, optional
        Set to True to only load data between `start_date` and 
        `end_date` for each year in the selection. If False,
        all days and/or hours in the range between `start_date`
        and `end_date` are selected (default behavior).
    param_col : str, optional
        The name of the column representing the meteorological 
        parameter to analyze. Default is "max_rain_hour_sum".
    hourslot_col : str, optional
        The name of the column representing the hour slots in 
        the data. Default is "hour_slot_max_rain_hour_sum".
    param_min_cutoff_val : float, optional
        The minimum value threshold for the parameter to consider
        for analysis. Any values below this threshold will be labeled 
        with hour slot -1 (indicating missing data). Default is 0.1.
    max_station_na_frac : float, optional
        The maximum allowed fraction of missing data per station. Stations
        with a higher fraction of missing values will be excluded.
        Default is 0.1.
    return_as_counts : bool, optional
        Indicate whether to keep the DataFrame as overview of counts.
        Default is False (meaning percentages will be returned).

    Returns
    -------
    pd.DataFrame
        A DataFrame with hour slots as the columns and station codes as 
        index, where each value represents the percentage occurrence 
        of that hour slot for the station of interest.

    Notes
    -----
    - The resulting DataFrame contains only stations with fewer than 
    the specified fraction of missing values.
    - Only works for daily KNMI data, as the hour slot-based data
    summaries are only directly available in 'daily' mode.
    - The output percentages are normalized per station, based on
    total occurrences.
    """
    # Translate parameter names back to parameter codes for request
    par_dct = load_tf_json("transform_params_day.json")

    # Use key 'parameter_name' to find associated parameter codes
    param_codes = []
    for col in [param_col, hourslot_col]:
        par_item = next((d for d in par_dct
                         if d["parameter_name"] == col), None)

        param_codes.append(par_item["parameter_code"])

    # Get dataset from KNMI web script service
    df_day = knmi_meteo_ingest.knmi_meteo_to_df(meteo_stns_list=None,
                                                meteo_params_list=param_codes,
                                                start_date=start_date,
                                                end_date=end_date,
                                                in_season=in_season)
    
    # Apply transformations to the raw dataset
    df_day_cleaned = transform_param_values(df_day)

    # Only select columns of interest
    sel_cols = ["date", "station_code"] + [param_col, hourslot_col]
    df_h = df_day_cleaned[sel_cols]

    # Separately label observationless days with hour slot -1
    is_cutoff = df_h[param_col] < param_min_cutoff_val
    df_h.loc[is_cutoff, hourslot_col] = -1
    
    # Remove parameter column (not needed further)
    df_h = df_h.copy().drop(columns=[param_col])

    # Pivot data with 'date' as index, 'stn_code' as cols
    df_h_pivot = (df_h.pivot(index="date",
                             columns="station_code"))

    # Flatten pivot table to single index
    df_h_pivot.columns = (df_h_pivot.columns
                          .get_level_values(1))

    # Only keep stations with less than fraction of values missing
    keep_cols = [col for col in df_h_pivot.columns 
                 if (df_h_pivot[col].isna().sum() 
                     <= (max_station_na_frac 
                         * len(df_h_pivot)))]

    # Apply the station column selection
    df_h_pivot = df_h_pivot[keep_cols]

    # Create DataFrame for storing counts / occurrences
    hr_idxs = [-1] + list(range(1, 25))
    df_h_counts = pd.DataFrame(index=hr_idxs)

    # Build up the 'counts' DataFrame col by col
    for col in df_h_pivot.columns:
        df_h_counts[col] = (df_h_pivot[col]
                            .value_counts())

    # Drop cutoff observations (-1) from the dataset (if any)
    cutoff_idx = -1

    if cutoff_idx in df_h_counts.index.values:
        df_h_counts.drop(cutoff_idx, axis='index',
                         inplace=True)
    
    # Do not convert to percentages if 'return_as_counts'
    if return_as_counts:
        df_h_slots = df_h_counts.copy()
    
    # Else, normalize data by total counts per column
    else:
        df_h_slots = df_h_counts.apply(lambda x: 100 * x / x.sum())

    # Simplify / prettify index columns
    df_h_slots.index = (df_h_slots.index
                                  .astype(int)
                                  .astype(str))
    
    # Fill missing values using 0, since that means that 
    # an hourslot did not occur in given period @ station
    df_h_slots = df_h_slots.copy().fillna(0)
    
    # Switch index and cols and return the overview
    return df_h_slots.T


def get_multiyr_hourslot_percentages(year_start: int, 
                                     year_end: int, **kwargs):
    """
    Create a multi-year view of KNMI hour slot percentages.

    This function retrieves hour slot data from the KNMI for a specified
    range of years and calculates the percentage distribution of hourly 
    occurrences across all years. The underlying `knmi_hourslot_percentage_df`
    function is used to fetch data for each year separately, and the results 
    are aggregated into a single DataFrame. Future modifications to the 
    fetching logic can be accommodated via the `**kwargs` parameter.

    This function splits the fetching of meteorological data from the KNMI web
    service for each year between `year_start` and `year_end` (inclusive).

    The function returns a DataFrame where the index represent station codes,
    and columns represent the hour slots, with each value indicating the 
    percentage of occurrences for that particular hour slot.

    Parameters
    ----------
    year_start : int
        The first year in the range of years to retrieve data for.
    year_end : int
        The last year in the range of years to retrieve data for (inclusive).
    **kwargs : dict, optional
        Additional keyword arguments to pass to the underlying 
        `knmi_hourslot_percentage_df` function that is called.

    Returns
    -------
    pd.DataFrame
        A DataFrame with hour slots as the columns and station codes as 
        index, where each value represents the percentage occurrence 
        of that hour slot for the station of interest.

    Notes
    -----
    - Function `knmi_hourslot_percentage_df` requires the following 
      parameters: `start_date`, `end_date`, and `return_as_counts`. Additional 
      arguments can be passed via `**kwargs`.
    - The data is aggregated using a summation of counts, and percentages are 
      calculated column-wise after aggregation.
    """
    # Fetch KNMI data year by year (to prevent overflow of service error)
    df_h_list = []
    for yr in range(year_start, year_end+1):
        
        df_h_year = knmi_hourslot_percentage_df(
            start_date=datetime.date(yr, 1, 1), 
            end_date=datetime.date(yr, 12, 31),
            return_as_counts=True,
            **kwargs)
        
        df_h_list.append(df_h_year)

    # Use a 'reduce' function to sum all count-dfs from result list at once
    df_h_counts = reduce(lambda x, y: x.add(y, fill_value=0), df_h_list)

    # Convert the summed counts to percentages (apply over columns)
    df_h_slots = df_h_counts.apply(lambda x: 100 * x / x.sum(), axis=1)

    return df_h_slots
