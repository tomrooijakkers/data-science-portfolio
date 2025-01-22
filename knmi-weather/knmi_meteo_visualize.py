"""KNMI Meteo Visualization

This script contains the data visualization functions for the data
returned by the KNMI web service which has been transformed in 
script `knmi_meteo_transform.py`.

This script requires that `numpy`, `pandas`, `matplotlib`, `pykrige` 
and `basemap` be installed within the Python environment you are running
this script in.

This file can also be imported as a module and contains the following
functions:

    * centered_symmetric_linear - symm. lin-func. with peak at N_max/2
    * blend_colormaps_cyclically - mix two cmaps to get cyclic hex color codes
    * cyclic_hourslot_boxplot - create KNMI cyclic-cmap h-slot boxplot of data
    * ordinary_kriging_nl_plot - create interpolation plot of The Netherlands
    * standardized_index_heatmap - plot heatmap of historic SP(E)I values

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import colors as cl
from matplotlib import colormaps as cm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.basemap import Basemap

from functools import partial
from pykrige.ok import OrdinaryKriging


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
    The color range edges of each colormap are set by `c_min` and `c_max`.

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
    - The colormap should be available via Matplotlib function `cm.get_cmap`.
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


def ordinary_kriging_nl_plot(locs_x, locs_y, values,
                             variogram_model : str = "linear",
                             cmap : str = "YlOrRd",
                             grid_dim_xy : int = 500,
                             map_resolution : str = "h",
                             plot_title : str = "<Plot title>",
                             val_label : str = "<Value label>",
                             loc_color : str = "k",
                             loc_marker : str = "o",
                             loc_msize : int = 8) -> plt.Axes:
    """
    Create a Kriging-based geospatial interpolation plot for The Netherlands.

    This function uses Ordinary Kriging to interpolate values based on
    geospatial data and visualizes the results on a map of The Netherlands.
    
    Customizable options for variogram modeling, colormap, grid
    resolution, and map appearance are included.

    Parameters
    ----------
    locs_x : array-like
        Longitudes of the input data points.
    locs_y : array-like
        Latitudes of the input data points.
    values : array-like
        Values associated with the input data points to be interpolated.
    variogram_model : str, optional
        The variogram model to use for Kriging interpolation.
        Default is "linear". See the PyKrige documentation for all options.
    cmap : str, optional
        The Matplotlib colormap to use for visualizing interpolated values.
        Default is "YlOrRd". See Matplotlib docs for all options.
    grid_dim_xy : int, optional
        The resolution of the interpolation grid (number of cells
        along each axis). Default is 500. Reduce to speed up calculation.
    map_resolution : str, optional
        The resolution of the map. Options include "c" (crude), "l" (low),
        "i" (intermediate), "h" (high), "f" (full). Default is "h".
    plot_title : str, optional
        Title text for the plot. Default is "<Plot title>".
    val_label : str, optional
        Label for the color bar to indicate the value that is visualized.
        Default is "<Value label>".
    loc_color : str, optional
        Color of the markers for input data locations. Default is "k" (black).
    loc_marker : str, optional
        Marker style for input data locations. Default is "o" (circle).
    loc_msize : int, optional
        Marker size for input data locations. Default is 8.

    Returns
    -------
    plt.Axes
        The matplotlib Axes object containing the interpolated plot.

    Notes
    -----
    - The function masks interpolated values outside the land boundaries
      of The Netherlands using a Basemap land-sea mask.
    - Any values outside of The Netherlands should be treated with
      caution, since those are often solely based on the edge locations.
    - The variogram model of choice depends highly on the data to plot.
      Try to vary between models if the spatial outcome does not make sense.
    """
    # Define custom styles for plot text and ticks
    textfont = {"fontname": "Palatino"}
    tickfont = {"fontname": "Georgia",
                "size": 12}

     # Set up Axes for plotting
    ax = plt.gca()

    # Create grid around NL for geospatial interpolation
    min_x, max_x = 3.0, 8.0
    min_y, max_y = 50.0, 54.0

    grid_x, grid_y = np.meshgrid(np.linspace(min_x, max_x, grid_dim_xy, 
                                             endpoint=True), 
                                 np.linspace(min_y, max_y, grid_dim_xy, 
                                             endpoint=True))

    # Set up OrdinaryKriging (2D-based) to populate grid with values
    OK = OrdinaryKriging(locs_x, locs_y, values,
                         variogram_model=variogram_model,
                         verbose=False, enable_plotting=False,
                         coordinates_type="geographic")

    # Execute Ord. Kriging algorithm to fill grid with interpolated values
    # (Note: the variance grid output is not used here; hence the '_')
    grid_z, _ = OK.execute("grid", grid_x[0, :], grid_y[:, 0])

    # Create a Basemap for The Netherlands & surrounding area
    m = Basemap(projection='merc', llcrnrlat=50.7, urcrnrlat=53.7, 
                llcrnrlon=3.3, urcrnrlon=7.3, resolution=map_resolution,
                ax=ax)
    
    # Add country borders, rivers and coastlines
    m.drawcountries(linewidth=0.5, linestyle='-.', color='k')
    m.drawrivers(linewidth=0.3, linestyle='solid', color='darkblue')
    m.drawcoastlines(linewidth=0.5, linestyle='solid', color='k')

    # Convert grid coordinates to map projection coordinates
    x, y = m(grid_x, grid_y)

    # Mask values outside land borders using the Basemap land-sea mask
    land_mask = np.vectorize(m.is_land)(x, y)
    land_grid_z = np.ma.masked_where(np.logical_not(land_mask), grid_z)

    # Plot interpolated values on the land parts of the map
    cs = m.contourf(x, y, land_grid_z, cmap=cmap)

    # Add color bar
    cbar = m.colorbar(cs, location='right', pad="10%")
    cbar.set_label(val_label, fontsize=14,  **textfont)

    # Add location markers to map plot as well
    locs_mx, locs_my = m(locs_x, locs_y)
    m.scatter(locs_mx, locs_my, color=loc_color, marker=loc_marker,
              s=loc_msize, ax=ax)
 
    # Customize font of the colorbar's tick markers
    cbar.ax.set_yticks(cbar.ax.get_yticks(), cbar.ax.get_yticklabels(), 
                       **tickfont)
    cbar.ax.set_xticks(cbar.ax.get_xticks(), cbar.ax.get_xticklabels(), 
                       **tickfont)
    
    # Add the title and return the final Axes object for plotting
    plt.title(plot_title, fontsize=18, **textfont)
    
    return ax


def standardized_index_heatmap(df_sp_data : pd.DataFrame,
                               sp_col : str,
                               stn_loc_name : str,
                               size_factor : int = 2):
    """
    Plot heatmap of historic SP(E)I values for marking drought/wetness.

    Expects SP(E)I input for a single location (measurement station).

    Dry periods are labeled with yellow / orange / red colors, "normal"
    periods are shown in white, and wet periods in shades of blue. NaNs
    are shown in grey.

    The SP(E)I values are categorized using the following definitions:
    - Less than 1.0 offset from the mean: "normal" conditions
    - At least 1.0 offset: drought / wetness period
    - At least 1.5 offset: severe drought / wetness
    - At least 2.0 offset: extreme drought / wetness
    - At least 3.0 offset: exceptional drought / wetness

    Parameters
    ----------
    df_sp_data : pd.DataFrame
        DataFrame with SPE(I) timeseries, in which columns `sp_col`,
        `month` and `year` need to be present to produce a plot.
    sp_col : str
        Column name of the col containing the SP(E)I values, e.g.: 'spi_3'.
    stn_loc_name : str
        Description of the location to use for the plot's title, e.g.:
        'De Bilt (260)'.
    size_factor : int, optional
        Determines the size of the plot; other items should scale along
        accordingly. The default value is 2.

    Returns
    -------
    plt.Axes
        The matplotlib Axes object containing the heatmap plot.

    Notes
    -----
    - The input DataFrame needs to have "month" and "year" columns.
    - Using a different SP(E)I timescale may yield vastly different results.
    """
    # Pivot the DataFrame for heatmap; keep NaNs (mark as grey later on)
    heatmap_data = df_sp_data.pivot_table(index="year", columns="month",
                                          values=sp_col, dropna=False)

    # Create prettified text version for SP(E)I variant
    sp_text = sp_col.upper().replace("_", "-")

    # Define custom styles for plot text and ticks
    textfont = {"fontname": "Palatino"}
    tickfont = {"fontname": "Georgia"}

    # Define category-to-color mapping
    cat_colors = {
        -9.9: "#820e0e",
        -3.0: "#c1251b",
        -2.0: "#fb7936",
        -1.5: "#fde069",
        -1.0: "white",
         0.0: "white",
         1.0: "#daedff",
         1.5: "#38a1f7",
         2.0: "#1953b7",
         3.0: "#1953b7",
         9.9: "#072f75"
    }

    # Pre-set X (month) and Y (year) indicators
    xs = heatmap_data.columns
    ys = heatmap_data.index

    # Set x-axis labels for the months
    xlabels = ["J", "F", "M", "A", "M", "J", 
               "J", "A", "S", "O", "N", "D"]

    # Automatic plot scaling to ensure "square"-like heatmap results
    scale_factor = 0.75 * len(ys)/len(xs)
    fig, ax = plt.subplots(figsize=(size_factor, int(scale_factor 
                                                     * size_factor)))

    # Create custom colormap using the defined cats and bounds
    cmap = cl.ListedColormap([cat_colors[cat] for cat in cat_colors])
    bounds = list(cat_colors.keys())
    norm = cl.BoundaryNorm(bounds, cmap.N)

    # Plot NaN values first; use a gray tint to mark those points
    ax.imshow(
        np.isnan(heatmap_data),
        cmap=cl.ListedColormap(["white", "#b5b5b5"]),
        aspect="auto",
        interpolation="nearest")

    # Use 'imshow' to superimpose non-NaN values of the heatmap on the plot
    heatmap = ax.imshow(
        heatmap_data,
        cmap=cmap,
        norm=norm,
        aspect="auto",  
        interpolation="nearest")

    # Add colorbar with corresponding category labels
    cbar = fig.colorbar(heatmap, ax=ax, boundaries=bounds[1:-1], 
                        ticks=bounds[1:-1], spacing="uniform")
    cbar.ax.set_yticklabels([f"{c}" for c in list(cat_colors.keys())[1:-1]],
                            fontsize=10, **tickfont)
    cbar.set_label(f"Category ({sp_text})", fontsize=10, **textfont)

    # Set title of plot; try to align centrally within the plot
    ax.set_title(f"{sp_text} Heatmap - {stn_loc_name}", fontsize=14, 
                 x=0.7, y=1.015, **textfont)

    # Customize x-ticks (months)
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels(xlabels, fontsize=8, **tickfont)

    # Customize y-ticks (years)
    ax.set_yticks(np.arange(len(ys)))
    ax.set_yticklabels(ys, fontsize=10, **tickfont)

    # Set y-axis ticks dynamically: try to always get multiples of 5 years
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=(len(ys)//5)+1, 
                                           prune="both"))

    # Add subtle grid lines and minor ticks for better readability
    ax.set_xticks(np.arange(-0.5, len(xs)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ys)), minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="--", linewidth=0.15)
    ax.tick_params(which="minor", size=1.5)

    return ax
