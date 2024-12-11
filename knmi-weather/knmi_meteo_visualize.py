"""KNMI Meteo Visualization

This script contains the data visualization functions for the data
returned by the KNMI web service which has been transformed in 
script `knmi_meteo_transform.py`.

This script requires that `pandas` and `matplotlib` be installed within
the Python environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * centered_symmetric_linear - symm. lin-func. with peak at N_max/2
    * blend_colormaps_cyclically - mix two cmaps to get cyclic hex color codes
    * cyclic_hourslot_boxplot - create KNMI cyclic-cmap h-slot boxplot of data
"""

import pandas as pd
import matplotlib.pyplot as plt

from functools import partial

from matplotlib import colors as cl
from matplotlib import colormaps as cm


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
