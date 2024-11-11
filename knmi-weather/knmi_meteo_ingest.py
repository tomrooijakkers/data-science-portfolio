import io
import datetime
import requests
import pandas as pd

from knmiclasses import (KNMIMeteoStation,
                         KNMIMeteoStationList)


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

        # Go to the start of the in-memory file;
        string_buffer.seek(0)

        # STN and YYYYMMDD are always returned, so provides a good cut-off
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

    return df_data


def knmi_meteo_to_df(meteo_stns_list: list | None,
                     meteo_params_list: list[str] | None,
                     start_date: datetime.date | datetime.datetime,
                     end_date: datetime.date | datetime.datetime,
                     mode: str = 'day') -> pd.DataFrame:
    """
    Convert day/hr KNMI automatic meteo station data to a Pandas DataFrame.

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
    mode : str
        Specify whether to load daily or hour-based values.

    Returns
    -------
    df_meteo : pd.DataFrame
        Pandas DataFrame containing the downloaded meteo data.

    """
    # Enforce specification of correct mode
    valid_modes = ['day', 'daily', 'dag', 'd', 'hour', 'hourly', 'h', 'hr',
                   'uur', 'u']

    mode_err_msg = f"Please enter a valid 'mode': {', '.join(valid_modes)}"
    assert mode.lower() in valid_modes, mode_err_msg

    # Get day or hourly data depending on the user's preference
    if mode.lower() in ['day', 'daily', 'dag', 'd']:
        meteo_url = "https://www.daggegevens.knmi.nl/klimatologie/daggegevens"

    elif mode.lower() in ['hour', 'hourly', 'h', 'hr', 'uur', 'u']:
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

    # Send the POST request including all specifications
    response = requests.post(meteo_url, data=post_data)

    # Transform the content of the KNMI response to a DataFrame and return it
    df_meteo = knmi_response_content_to_df(response.content)

    return df_meteo
