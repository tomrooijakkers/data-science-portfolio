import io
import urllib
import datetime
import requests
import pandas as pd
import lxml.html as lh

from knmiclasses import (KNMIMeteoStation,
                         CachedKNMIMeteoStations)


# TIP: This function is only needed to update metadata / dataset -> maybe move to "metadata_update" o.i.d.?
def knmi_get_all_station_codes(url: str) -> list[dict]:
    """Retrieve all station codes from the KNMI URL."""
    # Get HTML content of web site
    html_content = urllib.request.urlopen(url).read()

    # Find all station elements on the page (using class "station")
    station_elements = lh.fromstring(html_content).find_class("station")

    stations = []

    # Extract text for each station, convert to list of dicts
    for station_el in station_elements:
        station_text = station_el.text_content().strip().split(": ")
        stations.append({"station_code": station_text[0],
                         "location_name": station_text[-1]})
    
    # Raise an error if less than two stations were found (something must have gone wrong then)
    if len(stations) < 2:
        raise AssertionError(f"Incomplete or empty station list (len: {len(stations)} returned.")
    
    return stations


# TIP: This function is only needed to update metadata / dataset -> maybe move to "metadata_update" o.i.d.?
def knmi_get_all_daily_stations() -> list[dict]:
    """Retrieve all daily station codes from KNMI URL."""
    stations = knmi_get_all_station_codes("https://daggegevens.knmi.nl/klimatologie/daggegevens")
    return stations


# TIP: This function is only needed to update metadata / dataset -> maybe move to "metadata_update" o.i.d.?
def knmi_get_all_hourly_stations() -> list[dict]:
    """Retrieve all hourly station codes from KNMI URL."""
    stations = knmi_get_all_station_codes("https://daggegevens.knmi.nl/klimatologie/uurgegevens")
    return stations


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


# TIP: STNS mag ook "ALL" zijn => is het dan nog wel nodig dit zo moeilijk te maken?
# TIP: dag en uurstations zijn hetzelfde, alleen de params kunnen verschillen!
def knmi_meteo_to_df(meteo_stns_list: list,
                     start_date: datetime.date | datetime.datetime,
                     end_date: datetime.date | datetime.datetime,
                     mode: str = 'day') -> pd.DataFrame:
    """
    Convert day/hr KNMI automatic meteo station data to a Pandas DataFrame.

    Parameters
    ----------
    meteo_stns_list : list
        List of station IDs to get data for.
    start_date : date or datetime
        First day for which data should be downloaded.
    end_date : date or datetime
        Last day for which data should be downloaded.
    mode : str
        Specify whether to download daily or hour-based values.

    Returns
    -------
    df_meteo : pd.DataFrame
        Pandas DataFrame containing the downloaded meteo data.

    """
    # Enforce specification of correct mode
    valid_modes = ['day', 'daily', 'dag', 'd', 'hour', 'hourly', 'h', 'hr',
                   'uur', 'u']

    mode_err_msg = f"Please enter a valid 'mode': {', '.join(valid_modes)}"
    assert mode in valid_modes, mode_err_msg

    # Get day or hourly data depending on the user's preference
    if mode.lower() in ['day', 'daily', 'dag', 'd']:
        meteo_url = "https://www.daggegevens.knmi.nl/klimatologie/daggegevens"

    elif mode.lower() in ['hour', 'hourly', 'h', 'hr', 'uur', 'u']:
        meteo_url = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"

    # Get unique station IDs and define variables for data request
    meteo_stns = list(set(meteo_stns_list))

    # Collect specifications in a 'post-data' dictionary
    post_data = {"stns": ":".join([str(stn).zfill(3) for stn in meteo_stns]),
                 "vars": "ALL",
                 "start": start_date.strftime('%Y%m%d'),
                 "end": end_date.strftime('%Y%m%d'),
                 "fmt": "csv"}

    # Send the POST request including all specifications
    response = requests.post(meteo_url, data=post_data)

    # Transform the content of the KNMI response to a DataFrame and return it
    df_meteo = knmi_response_content_to_df(response.content)

    return df_meteo
