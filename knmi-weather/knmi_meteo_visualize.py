import datetime
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial, reduce
from matplotlib import colors as cl
from matplotlib import colormaps as cm

import knmi_meteo_ingest
import knmi_meteo_transform


def centered_symmetric_linear(N: float, N_max: float,
                              f_min: float, f_max: float) -> float:
    """
    Symmetric linear function peaking at the midpoint of range [0, N_max].

    This function generates a value based on a symmetric linear gradient:
    - Starts at `f_min` when `N` is 0.
    - Peaks at `f_max` when `N` is at the midpoint (`N_max / 2`).
    - Returns to `f_min` when `N` is at `N_max`.

    Parameters
    ----------
    N : float
        The input value for which the function is evaluated. 
        Should be in the range [0, N_max] for correct results.
    N_max : float
        The maximum value of the range.
    f_min : float
        The minimum output value at the endpoints (0 and N_max).
    f_max : float
        The maximum output value at the midpoint (N_max / 2).

    Returns
    -------
    float
        The computed function value.

    Examples
    --------
    >>> centered_symmetric_linear(0, 10, 0, 1)
    0
    >>> centered_symmetric_linear(5, 10, 0, 1)
    1
    >>> centered_symmetric_linear(10, 10, 0, 1)
    0
    """
    # Get slope of function (twice the slope of a normal linear function)
    a = 2 * (f_max - f_min) / N_max
    
    # Apply positive slope for N in [0, N_max / 2]
    if N < N_max / 2:
        return a * N + f_min
    # Apply negative slope for N in [N_max / 2, N_max]
    else:
        return -a * (N - N_max / 2) + f_max
    

def blend_colormaps_cyclically(n_cats: int, cmap_name_a: str,
                               cmap_name_b: str,
                               c_min: float, c_max: float) -> list[str]:
    """
    Generate a list of cyclic color hex codes by combining two colormaps.

    This function creates a cyclic color palette by blending two colormaps. 
    It uses a symmetric linear function to ensure a smooth transition 
    through the colormaps. Colors from the first colormap are used for the 
    first half of the categories, and colors from the inverted(!)
    second colormap are used for the second half.

    Parameters
    ----------
    n_cats : int
        The total number of categories (colors) to generate.
    cmap_name_a : str
        The name of the first colormap (from Matplotlib's colormap library).
    cmap_name_b : str
        The name of the second colormap (from Matplotlib's colormap library).
    c_min : float
        The minimum value for the normalized range (e.g., 0.125 or 0).
    c_max : float
        The maximum value for the normalized range (e.g., 0.85 or 1).

    Returns
    -------
    list of str
        A list of hexadecimal color codes corresponding to the cyclic colormap.

    Notes
    -----
    - The colormaps are accessed using Matplotlib's `cm.get_cmap` function.
    - The `centered_symmetric_linear` function is used to compute the 
      normalized value for each category, ensuring symmetry and smooth
      transitions.
    - For the best result it is advised to choose the two colormaps in such
      a way that the color of the end of colormap A is very similar to the
      color of the end of colormap B.
    - Also set c_min around 0.15 and c_max around 0.85, as this filters out 
      the extreme ends of each colormap.

    Examples
    --------
    >>> generate_cyclic_color_hexes(10, 'viridis', 'plasma', 0, 1)
    ['#440154', '#482475', '#414487', '#31688e', '#21918c', '#fde725', 
     '#f7d13d', '#f29e2e', '#e8641e', '#d43d51']
    """
    # Define colormaps for first and second halves of plotting
    cmap_a = cm.get_cmap(cmap_name_a)
    cmap_b = cm.get_cmap(cmap_name_b)

    # Set cyclic run through parts of the colormap (symmetrically);
    # preset all variables (except for iteration value i)
    partial_f = partial(centered_symmetric_linear,
                        N_max=n_cats, f_min=c_min, f_max=c_max)

    # Build up the colormap colors; walk through colormap A
    # in first half & through colormap B in second half
    colors = []
    for i in range(n_cats):
        is_first_half = ((i // (n_cats // 2)) == 0)
        cmap = (cmap_a if is_first_half else cmap_b)
        colors.append(cl.to_hex(cmap(partial_f(i))))

    # Return the generated list with color hex codes
    return colors


def cyclic_hourslot_boxplot(df_h_slot: pd.DataFrame,
                            title_text: str,
                            subtitle_text: str = "", 
                            cmap_name_a: str = "inferno", 
                            cmap_name_b: str = "inferno",
                            c_max: float = 0.85,
                            c_min: float = 0.125) -> plt.Axes:
    """
    Create hourslot boxplot of KNMI data using two colormap(s).

    Traverses through the first colormap (a) in linear order
    and through the second colormap (b) in reverse linear order.
    The edges of each colormap are set by `c_min` and `c_max`. 

    Parameters
    ----------
    df_h_slot : pd.DataFrame
        DataFrame with stations as index and hourslots as cols.
    title_text : str
        The text to display on the main title of the figure.
    subtitle_text : str, optional
        The text to display below the main title of the figure.
    cmap_name_a : str
        The name of the colormap for the first half of data.
    cmap_name_b : str
        The colormap name for the second half of the data; this
        colormap will be traversed through in reverse order.
    c_max : float
        Maximum (normalized) color value to pick a color from
        the colormap from. Set this around 0.85 to prevent
        the display of 'extreme' colors from the colormaps.
    c_min : float
        Maximum (normalized) color value to pick a color from
        the colormap from. Set this around 0.15 to prevent
        the display of 'extreme' colors from the colormaps.

    Notes
    -----
    - The colormap should be available via Matplotlib's `cm.get_cmap` function.
    - For the best result it is advised to choose the two colormaps in such
      a way that the color at the end of colormap (a) is very similar to the
      color at the end of colormap (b).
    """  
    # Define custom styles for plot text and ticks
    textfont = {"fontname": "Palatino"}
    tickfont = {"fontname": "Georgia"}

    # Create boxplot with Pandas & Matplotlib
    ax, bp = df_h_slot.plot(
        # Create a boxplot
        kind="box",
        # Make markers for the means
        showmeans=True,
        # Properties for boxes
        boxprops=dict(linestyle='-',
                      linewidth=1.5),
        # Properties for outliers
        flierprops=dict(marker='x',
                        linestyle='none',
                        linewidth=0),
        # Properties for mean symbols
        meanprops=dict(marker='o',
                       markeredgecolor='none',
                       markerfacecolor='w',
                       markersize=2),
        # Properties for median lines
        medianprops=dict(linestyle='-', 
                         color='w',
                         linewidth=1),
        # Properties for IQR whisker lines
        whiskerprops=dict(linestyle='-',
                          linewidth=0.75),
        # Properties for IQR cap lines
        capprops=dict(linestyle='-', linewidth=0.5),
        # Show outliers, grid and/or apply rotation
        showfliers=True, grid=True, rot=0,
        # Return dict to allow for custom plot settings
        patch_artist=True,
        # Return figure and its elements for customization
        return_type='both')
    
    # Number of categories is equal to number of boxes to draw
    n_cats = len(bp["boxes"])

    # Get colormap hex codes from colormap blending
    colors = blend_colormaps_cyclically(n_cats, cmap_name_a,
                                        cmap_name_b,
                                        c_min, c_max)

    # Set boxplot + outlier item colors based on colormaps
    for i in range(n_cats):
        bp['boxes'][i].set_facecolor(colors[i])
        bp['boxes'][i].set_edgecolor(colors[i])
        bp['fliers'][i].set_markeredgecolor(colors[i])

        # Whiskers & caps occur twice per box; correct for that
        for item in ['whiskers', 'caps']:
            bp[item][i].set_color(colors[i // 2])
            bp[item][n_cats + i].set_color(colors[(n_cats + i) // 2])

    # Add horizontal line at "expected" hour slot value if 
    # rainfall were perfectly uniform each day (= 100%/24)
    ax.axhline(y=100/n_cats, color=colors[0],
               linestyle='--',
               linewidth=1, zorder=1)

    # Add x, y labels and title; use custom font
    ax.set_xlabel("Hour slot (1= 0-1 UT; 2= 1-2 UT, ...)", **textfont)
    ax.set_ylabel("Occurrence (in %)", **textfont)

    # Set title and subtitle text and styles
    #title_text = f"Hour slots with maximum daily rainfall - Dutch KNMI stations ({YEAR})"
    #subttl_text = ("Means: circles, medians: lines, uniform reference: dashdot line,"
    #               " outliers (non-IQR): crosses")

    # Get central x-pos; max y-pos as title plotting locations
    x_mid = sum(ax.get_xlim()) / len(ax.get_xlim())
    y_max = ax.get_yticks()[-1]

    # Add title and subtitles as text on the Axes object
    ax.text(x_mid, 1.11*y_max, title_text, fontsize=14,
            ha='center', va='top', **textfont)
    ax.text(x_mid, 1.0375*y_max, subtitle_text, fontsize=9,
            ha='center', va='top', **textfont)

    # Customize font of the tick markers
    ax.set_yticks(ax.get_yticks(), ax.get_yticklabels(), **tickfont)
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), **tickfont)

    # Customize the grid layout
    ax.grid(True, color="grey", linewidth=0.15, linestyle="-")

    # Cut off any potential negative y-values (should not occur)
    ax.set_ylim(0)

    # Return the Axes object for plotting
    return ax


def knmi_hourslot_percentage_df(start_date: datetime.date, end_date: datetime.date,
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
    par_dct = knmi_meteo_transform.load_tf_json("transform_params_day.json")

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
    df_day_cleaned = knmi_meteo_transform.transform_param_values(df_day)

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


def get_multiyr_hourslot_percentages(year_start, year_end, **kwargs):
    """ """

    # TODO: WRITE DOCUMENTATION FOR THIS FUNCTION

    # Fetch KNMI data year by year to prevent service overflow
    df_h_list = []
    for yr in range(year_start, year_end+1):
        df_h_year = knmi_hourslot_percentage_df(start_date=datetime.date(yr, 1, 1), 
                                                end_date=datetime.date(yr, 12, 31),
                                                return_as_counts=True,
                                                **kwargs)
        df_h_list.append(df_h_year)

    # Use a reduce function to sum all count-dfs from the result list at once
    df_h_counts = reduce(lambda x, y: x.add(y, fill_value=0), df_h_list)

    # Convert the summed counts to percentages (apply over columns)
    df_h_slots = df_h_counts.apply(lambda x: 100 * x / x.sum(), axis=1)

    return df_h_slots