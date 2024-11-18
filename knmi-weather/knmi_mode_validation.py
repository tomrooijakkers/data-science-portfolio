"""KNMI Mode Validation

Script containing the validation functions that are used by
multiple of the other scripts.

This file can also be imported as a module and contains the following
functions:

    * validate_mode_get_source - Validate inserted mode ('daily' or 'hourly')

"""

def validate_mode_get_source(mode: str) -> str:
    """
    Validate inserted mode ('daily' or 'hourly').

    Also returns the KNMI source URL associated with either the
    daily or the hourly values.

    Parameters
    ----------
    mode : str
        Specifies whether to load daily or hour-based values.

    Returns
    -------
    meteo_url : str
        Pandas DataFrame containing the KNMI data (in CSV-style).

    Raises
    ------
    AssertionError
        In case the 'mode' cannot be mapped to 'daily' or 'hourly'.

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

    # If we reached here, return the correct source
    return meteo_url
