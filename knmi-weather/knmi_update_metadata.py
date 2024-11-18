"""KNMI Update Metadata

This script updates and overwrites the files in subfolder `metadata`.

This script requires that `pandas`, `requests`, `urllib`, `lxml`
and `cssselect` be installed within the Python environment you
are running this script in.

Furthermore a stable internet connection and the availability of the
KNMI script data retrieval service are required.

This file can also be imported as a module and contains the following
functions:

    * knmi_get_all_parameter_codes - scrape all param codes from URL
    * write_to_json_datafile - load dictlist to JSON in folder
    * knmi_update_parameter_metadata - scrape and save param data
    * knmi_station_content_to_df - parse KNMI byte-response to df
    * knmi_update_meteo_station_metadata - fetch and save station data
    * knmi_update_all_metadata - wrapper to run all above functions


For more info on the KNMI script data retrieval services, please see:
https://www.knmi.nl/kennis-en-datacentrum/achtergrond/data-ophalen-vanuit-een-script
"""

import os
import io
import re
import json
import urllib
import requests
import datetime

import pandas as pd
import lxml.html as lh


def knmi_get_all_parameter_codes(url: str) -> list[dict]:
    """Retrieve all param codes from the KNMI URL."""
    # Get HTML content of web site
    html_content = urllib.request.urlopen(url).read()

    # Find all param elements on the page (using parent "fields_container")
    css_sel = ".fields-container > :not(:nth-child(1))"
    param_elements = lh.fromstring(html_content).cssselect(css_sel)

    param_list = []

    # Extract text for each parameter, convert to list of dicts
    for param_el in param_elements:
        param_text = param_el.text_content().strip().split(": ")
        param_desc = param_text[1].strip().replace(". Meer info", "")
        
        # Edge case: some descriptions contain multiple colon-seps; then only skip first item
        if len(param_text) > 2:
            param_desc = ": ".join(param_text[1:-1]).strip().replace(". Meer info", "")
        
        param_list.append({"parameter_code": param_text[0],
                           "parameter_desc": param_desc})

    # Raise an error if less than two stations were found (something must have gone wrong then)
    if len(param_list) < 2:
        raise AssertionError(f"Incomplete/empty param. list (len: {len(param_list)} returned.")
    
    return param_list


def write_to_json_datafile(sourceurl: str, datalist: list[dict],
                           datakey: str, datafile: str) -> None:
    """Write dictlist to JSON file in 'metadata' folder."""
    # Define location for output file
    datafile_loc = os.path.join('metadata', datafile)

    # Add the update timestamp (in UTC)
    updated_dt = (datetime.datetime.now(datetime.timezone.utc)
                  .strftime('%Y-%m-%d %H:%M %Z'))
    
    # Define object for saving as JSON
    json_obj = {"last_updated": updated_dt,
                "source": sourceurl}
    json_obj[datakey] = datalist

    # Write the JSON object to file
    with open(datafile_loc, 'w', encoding='utf-8') as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def knmi_update_parameter_metadata(sourceurl: str, datafile: str) -> None:
    """Update parameter metadata from KNMI web source."""
    # Fetch all current parameters from KNMI's script service
    parameters = knmi_get_all_parameter_codes(sourceurl)

    # Write parameter data to JSON file in 'metadata' folder
    write_to_json_datafile(sourceurl, parameters,
                           "parameters", datafile)


def knmi_station_content_to_df(knmi_response_content: bytes) -> pd.DataFrame:
    """
    Load byte-like station content from the KNMI site to a Pandas DataFrame.

    Parameters
    ----------
    knmi_response_content : bytes
        Byte-stream containing the raw (csv-like) data returned by the KNMI.

    Returns
    -------
    df_stations : pd.DataFrame
        Pandas DataFrame containing KNMI station data.

    """
    # Write the requested data to an in-memory string file
    with io.StringIO() as string_buffer:
        string_buffer.write(knmi_response_content.decode("utf-8"))

        # Go to the start of the in-memory file
        string_buffer.seek(0)

        # STN and first alphabetic param form the station data cut-offs
        split_text = "# STN "

        # Only keep string portion between 1st and 2nd cutoff, replace first '#'
        string_data = (split_text.replace("# ", "")
                       + (string_buffer.getvalue().split(split_text)[1]))

        # Keep the first line containing the column names
        string_first_line = string_data.split('\n', 1)[0]

        # Only keep lines with station data (regex-based)
        re_pattern = r"\s*#\s+[0-9]{3}\s[^\n\r\t]*"
        matched_lines = re.findall(re_pattern, string_data)

        # Add column (first) line back to data-string
        string_data = ((string_first_line + "\n"
                       + "".join(matched_lines))
                       .replace("#", ""))

        # Erase the old content; write the leftover (data) content to file
        string_buffer.truncate(0)
        string_buffer.write(string_data)

        # Return to the start of the file; convert data to Pandas DataFrame
        string_buffer.seek(0)
        df_stations = pd.read_csv(string_buffer, sep=r"\s{2,}",
                                  engine="python",
                                  index_col=False,
                                  on_bad_lines="skip",
                                  encoding="utf-8")

    return df_stations


def knmi_update_meteo_station_metadata(sourceurl: str, 
                                       datafile: str) -> None:
    """Update KNMI automatic meteo station metadata."""
    # Set start, end dates in future, to only return metadata
    start_date = datetime.date.today() + datetime.timedelta(days=1)
    end_date = datetime.date.today() + datetime.timedelta(days=2)

    # Collect specifications in a 'post-data' dictionary
    post_data = {"stns": "ALL",
                 "vars": "ALL",
                 "start": start_date.strftime('%Y%m%d'),
                 "end": end_date.strftime('%Y%m%d'),
                 "fmt": "csv"}

    # Send the POST request including all specifications
    response = requests.post(sourceurl, data=post_data)

    # Transform the content of the KNMI response to a DataFrame
    df_stations = knmi_station_content_to_df(response.content)
    
    # Convert Pandas DataFrame into a list of dicts
    stations = df_stations.to_dict(orient="records")

    # Write the parameter data to JSON file in 'metadata' folder
    write_to_json_datafile(sourceurl, stations,
                           "stations", datafile)


def knmi_update_all_metadata() -> None:
    """Update all KNMI metadata files."""
    d_sourceurl = "https://daggegevens.knmi.nl/klimatologie/daggegevens"
    h_sourceurl = "https://daggegevens.knmi.nl/klimatologie/uurgegevens"

    # Update daily and hourly parameter data
    print("Updating daily parameter metadata...")
    knmi_update_parameter_metadata(d_sourceurl, "knmi_parameters_daily.json")
    print("Success.")

    print("Updating hourly parameter metadata...")
    knmi_update_parameter_metadata(h_sourceurl, "knmi_parameters_hourly.json")
    print("Success.")

    # Update station data (by doing a data-less request to KNMI service)
    print("Updating meteo station metadata...")
    knmi_update_meteo_station_metadata(d_sourceurl, "knmi_meteo_stations.json")
    print("Success.")
    