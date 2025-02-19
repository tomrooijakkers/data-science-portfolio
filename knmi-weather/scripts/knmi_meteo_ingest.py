"""KNMI Meteo Ingestion

This script makes use of the KNMI script data retrieval
service through Python and loads the data to Pandas DataFrames.

This script requires that `pandas` and `requests` be installed within 
the Python environment you are running this script in.

Furthermore a stable internet connection and the availability of the
KNMI script data retrieval service are required.

This file can also be imported as a module and contains the following
functions:

    * load_md_json_to_df - load metadata JSON to pandas DataFrame (df)
    * knmi_response_content_to_df - parse KNMI byte-response to df
    * knmi_load_meteo_stations - load meteo stations as df
    * knmi_load_daily_parameters - load daily parameters as df
    * knmi_load_hourly_parameters - load hourly parameters as df
    * knmi_meteo_to_df - handle KNMI 'get' request, load result to df 

For more info on the KNMI script data retrieval services, please see:
https://www.knmi.nl/kennis-en-datacentrum/achtergrond/data-ophalen-vanuit-een-script
"""

import io
import os
import json
import datetime
import requests
import pandas as pd

from knmi_mode_validation import validate_mode


def load_md_json_to_df(filename: str, datakey: str) -> pd.DataFrame:
    """Load metadata JSON to pd.DataFrame."""
    script_dir = os.path.dirname(__file__)
    with open(os.path.join(script_dir, "metadata", filename)) as f:
        data = json.load(f)
        df = pd.DataFrame.from_records(data[datakey])

    return df


def knmi_load_meteo_stations() -> pd.DataFrame:
    """Load meteo station metadata as pd.DataFrame."""
    return load_md_json_to_df("knmi_meteo_stations.json", "stations")


def knmi_load_daily_parameters() -> pd.DataFrame:
    """Load daily parameter metadata as pd.DataFrame."""
    return load_md_json_to_df("knmi_parameters_daily.json", "parameters")


def knmi_load_hourly_parameters() -> pd.DataFrame:
    """Load hourly parameter metadata as pd.DataFrame."""
    return load_md_json_to_df("knmi_parameters_hourly.json", "parameters")


def knmi_response_content_to_df(knmi_response_content: bytes) -> pd.DataFrame:
    """
    Load byte-like response content from the KNMI site to a Pandas DataFrame.

    Parameters
    ----------
    knmi_response_content : bytes
        Byte-stream containing the raw (csv-like) data returned by the KNMI.

    Returns
    -------
    df_data : pd.DataFrame
        Pandas DataFrame containing the KNMI data (in CSV-style).

    """
    # Write the requested data to an in-memory string file
    with io.StringIO() as string_buffer:
        string_buffer.write(knmi_response_content.decode("utf-8"))

        # Go to the start of the in-memory file
        string_buffer.seek(0)

        # The string buffer should not start with HTML declaration
        first_line = string_buffer.readline()
        
        if 'DOCTYPE html' in first_line:
            # string_buffer.truncate(0)
            raise AssertionError("KNMI service returned HTML instead of data - "
                                 "request less data. If that does not help, "
                                 "check whether the service is available "
                                 "at this moment.")

        # If all OK, return to the start of the in-memory file
        string_buffer.seek(0)

        # STN and YYYYMMDD are always returned as data: provides good cut-off
        split_text = "# STN,YYYYMMDD"

        # Only keep string portion after cutoff, replace first '#'
        string_data = (split_text.replace("#", "")
                       + string_buffer.getvalue().split(split_text)[-1])

        # Erase the old content; write the leftover (data) content to file
        string_buffer.truncate(0)
        string_buffer.write(string_data)

        # Return to the start of the file; convert data to Pandas DataFrame
        string_buffer.seek(0)
        df_data = pd.read_csv(string_buffer, sep=r"\s*,\s*", engine="python",
                              index_col=False, on_bad_lines='skip',
                              encoding='utf-8')
        
    # Remove any possible double-entry columns
    df_data.drop(df_data.columns[df_data.columns.str.endswith('.1')],
                 axis=1, inplace=True)

    return df_data


def knmi_meteo_to_df(meteo_stns_list: list | None,
                     meteo_params_list: list[str] | None,
                     start_date: datetime.date | datetime.datetime,
                     end_date: datetime.date | datetime.datetime,
                     mode: str = 'day',
                     in_season: bool = False) -> pd.DataFrame:
    """
    Convert day/hr KNMI automatic meteo station data to pd.DataFrame.

    Parameters
    ----------
    meteo_stns_list : list or None
        List of station IDs to get data for. If None, uses "ALL".
    meteo_params_list : list or None
        List of parameter codes to include. If none, uses "ALL".
    start_date : date or datetime
        First day for which data should be downloaded.
    end_date : date or datetime
        Last day for which data should be downloaded.
    mode : str, optional
        Specify whether to load day- or hour-based values.
    in_season : bool, optional
        Set to True to only load data between `start_date` and 
        `end_date` for each year in the selection. If False,
        all days and/or hours in the range between `start_date`
        and `end_date` are selected (default behavior).

    Returns
    -------
    df_meteo : pd.DataFrame
        Pandas DataFrame containing the downloaded meteo data.

    """
    # Enforce specification of correct mode, get corresponding URL
    mode = validate_mode(mode)

    if mode == "daily":
        meteo_url = "https://www.daggegevens.knmi.nl/klimatologie/daggegevens"
    elif mode == "hourly":
        meteo_url = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"

    # Get unique station IDs and define variables for data request
    if meteo_stns_list:
        meteo_stns = list(set(meteo_stns_list))
        meteo_stns = ":".join([str(stn).zfill(3) for stn in meteo_stns])
    else:
        meteo_stns = "ALL"

    if meteo_params_list:
        meteo_params = list(set(meteo_params_list))
        meteo_params = ":".join([prm for prm in meteo_params])
    else:
        meteo_params = "ALL"

    # Collect specifications in a 'post-data' dictionary
    post_data = {"stns": meteo_stns,
                 "vars": meteo_params,
                 "start": start_date.strftime('%Y%m%d'),
                 "end": end_date.strftime('%Y%m%d'),
                 "fmt": "csv"}
    
    # Add 'inseason' parameter if in_season is True
    if in_season:
        post_data["inseason"] = "Y"

    # Send the POST request including all specifications
    response = requests.post(meteo_url, data=post_data)

    # Raise error if something went wrong with the request
    response.raise_for_status()

    # Parse the KNMI response content and return it
    df_meteo = knmi_response_content_to_df(response.content)

    return df_meteo
