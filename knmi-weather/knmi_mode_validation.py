"""KNMI Mode Validation

Script containing the validation functions that are used by
multiple of the other scripts.

This file can also be imported as a module and contains the following
functions:

    * validate_mode - Validate inserted mode ('daily' or 'hourly')

"""

def validate_mode(mode: str) -> str:
    """
    Validate inserted mode ('daily' or 'hourly').

    Parameters
    ----------
    mode : str
        Specifies whether to load daily or hour-based values.

    Returns
    -------
    mode : str
        Cleaned version of the 'mode': 'daily' or 'hourly'.

    Raises
    ------
    AssertionError
        Raised if the 'mode' cannot be mapped to 'daily' or 'hourly'.

    """
    # Enforce specification of correct mode
    valid_modes = ['day', 'daily', 'dag', 'd', 'hour', 'hourly', 'h', 'hr',
                   'uur', 'u']

    mode_err_msg = f"Please enter a valid 'mode': {', '.join(valid_modes)}"
    assert mode.lower() in valid_modes, mode_err_msg

    # Get day or hourly data depending on the user's preference
    if mode.lower() in ['day', 'daily', 'dag', 'd']:
        cleaned_mode = "daily"

    elif mode.lower() in ['hour', 'hourly', 'h', 'hr', 'uur', 'u']:
        cleaned_mode = "hourly"

    # Return the correct value
    return cleaned_mode
