"""KNMI SPI Utils

This script contains the utility / helper functions for the SPI / SPEI
analysis of KNMI weather data.

This script requires that `pandas`, `numpy`, `scipy` and `sklearn`
be installed within the Python environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * validate_station_code - get stn. details for a given station code
    * param_code_from_col - get param. code associated with `param_col`
    * get_measured_stn_param_values - get measured values from a KNMI station
    * check_years_to_impute - find years for imputation from NaNs in series
    * find_imputation_stations - find (high-) corr. stations for imputation
    * filter_datacols_to_impute - filter df cols to match target, feature cols
    * impute_vals_from_targetcol - impute df values by use of a MICE algorithm
    * merge_measured_and_imputed_data - merge measured and imputed dfs to one
    * measured_imputed_to_one_series - combine measured and imputed cols to one
    * imputation_workflow - run full imputation workflow for a KNMI stn. param
    * calc_rain_min_evap_df - calculate (rain - evap) for SPEI
    * agg_to_grouped - aggregate DataFrame timeseries to N-monthly bins
    * fit_distr_to_series - fit distribution(s) to a (grouped) timeseries
    * cdf_bestfit_to_z_scores - convert CDF values to Z-scores from invnorm
    * calculate_nmonth_spi - calculate SPI-N scores for precip data
    * calculate_nmonth_spei - calculate SPEI-N scores for (rain - evap) data
    * get_events_from_z_scores - Get drought or wetness events from SP(E)I-N

"""

import datetime
import numpy as np
import pandas as pd
import scipy.stats as scs

import knmi_meteo_ingest
import knmi_meteo_transform
import mice_imputation_utils


def validate_station_code(stn_code: int) -> dict:
    """
    Get station details for a given station code.

    Can be used to check whether the input station code is valid.
    
    Parameters
    ----------
    stn_code : int
        Three-digit KNMI station code (ID), e.g.: 260 for De Bilt.

    Returns
    -------
    stn_dict : dict
        Basic dictionary with 'raw' (uncleaned) station attributes.

    Raises
    ------
    AssertionError
        Occurs if the station code is not found.
        Retry using a valid KNMI station code (ID), e.g.: 260 for De Bilt.

    """
    # Get station details for chosen code (sanity check)
    stations_raw = knmi_meteo_ingest.knmi_load_meteo_stations()

    # Show details of chosen station (should be non-empty)
    stn_dict = (stations_raw[stations_raw["STN"] == stn_code]
                .to_dict(orient="list"))

    # AssertionError message with valid options if stn. code invalid
    valid_stns_str = ", ".join(str(x) for x in stations_raw["STN"])
    err_msg = ("Invalid station code (integer) - valid options: "
               f"{valid_stns_str}.")

    assert len(stn_dict["NAME"]) > 0, err_msg

    # Return details of chosen station
    return stn_dict


def param_code_from_col(param_col: str) -> str:
    """Retrieve parameter code associated with `param_col`."""
    # Translate parameter names back to parameter codes for request
    params_file = "transform_params_day.json"
    param_dct = knmi_meteo_transform.load_tf_json("transform_params_day.json")

    # Use key 'parameter_name' to find associated parameter codes
    param_code = next((d["parameter_code"] for d in param_dct if
                       d["parameter_name"] == param_col), None)

    # Raise AssertionError if no matching parameter_code was found
    assert_err = (f"No parameter match found for '{param_col}' in transform "
                  f" file '{params_file}'; please use a valid `param_col`.")
    assert param_code, assert_err

    # If all OK, return the found `param_code`
    return param_code


def get_measured_stn_param_values(stn_code: int,
                                  param_col: str = "rain_sum",
                                  start_year: int = 1901,
                                  end_year: int | None = None) -> pd.DataFrame:
    """
    Obtain measured parameter values from a specified KNMI station.
    
    Automatically fetches data for a long period, breaking up the request
    to the KNMI web service in years, then combining the result.

    Primary cleaning of data also takes place within this function.

    Furthermore, this function ensures that long periods of NaNs are filtered
    out, while at the same time returning a DataFrame that always starts at
    the first day of the first non-NaN month and ends at the last day of the 
    last non-NaN month.

    Parameters
    ----------
    stn_code : int
        Three-digit KNMI station code (ID), e.g.: 260 for De Bilt.
    param_col : str, optional
        Name of the cleaned parameter column to fetch data for.
        The default value is "rain_sum".
    start_year : int, optional
        Starting year to retrieve KNMI data for.
        The default value is 1901 (first year of data @ De Bilt).
    end_year : int or None, optional
        Last year (inclusive) to retrieve KNMI data for.
        If not specified, the previous year based on the current
        date (in UTC) will be used. The default value is None.

    Returns
    -------
    df_clean : pd.DataFrame
        Pre-cleaned KNMI dataset containing the fetched dataset.

    Notes
    -----
    - Running this function takes 30 secs - 1 min per ~100 yrs of data.
    - To speed up, please select a shorter timeframe.
    """
    # Parse year to last year if no final year was specified
    if end_year is None:
        end_year = datetime.utcnow().year - 1

    # Find associated parameter code for KNMI request
    param_code = param_code_from_col(param_col)

    # Get daily data from KNMI web script service
    df_list_y = []

    # Split the request in individual years to prevent service overflow
    for year in range(start_year, end_year+1):
        df_y = (knmi_meteo_ingest
                     .knmi_meteo_to_df(meteo_stns_list=[stn_code],
                                       meteo_params_list=[param_code],
                                       start_date=datetime.date(year, 1, 1),
                                       end_date=datetime.date(year, 12, 31),
                                       mode="day"))
    
        df_list_y.append(df_y)

    # Concatenate each non-empty yearly series to full history
    # Note: 'ignore_index' used to deduplicate indexes from yearly DataFrames
    df_raw = pd.concat([df for df in df_list_y if not df.empty],
                       ignore_index=True)
    
    # Apply transformations to clean the raw dataset
    df_clean = knmi_meteo_transform.transform_param_values(df_raw)

    # Cut off leading and trailing NaNs from history dataset;
    # in this way we get our actual historical start and end dates
    min_idx = (df_clean[[param_col]]
               .apply(pd.Series.first_valid_index).max())
    max_idx = (df_clean[[param_col]]
               .apply(pd.Series.last_valid_index).min())
    df_clean = df_clean.loc[min_idx: max_idx, :]

    # Now, re-index DataFrame so that it:
    # 1. Always starts at first day of the first month found
    first_date = df_clean["date"].iloc[0]
    last_date = df_clean["date"].iloc[-1]
    target_first_date = datetime.date(year=first_date.year, 
                                      month=first_date.month, 
                                      day=1)

    # 2. Always ends on last day of last month found
    target_last_date = (datetime.date(
        year=(last_date.year + (last_date.month // 12)),
        month=(last_date.month % 12) + 1,
        day=1) - datetime.timedelta(days=1))
    
    # 3. Always has NaNs for missing indexes in the range
    full_mnths_index = pd.date_range(target_first_date,
                                     target_last_date, freq="D")

    # Set range of collected non-NaN dates as initial index
    df_clean.index = pd.to_datetime(df_clean["date"])

    # Reindex to full months
    df_clean = df_clean.reindex(full_mnths_index)

    # Remove station code and "incomplete" date column
    drop_cols = ["date", "station_code"]
    df_clean = df_clean.loc[:,~df_clean.columns.isin(drop_cols)]

    # Rename full-month index to "date"
    df_clean = df_clean.rename_axis(index="date")

    # Return DataFrame with "date" as index, `param_col` as column
    return df_clean


def check_years_to_impute(df_clean: pd.DataFrame,
                          param_col: str = "rain_sum") -> tuple[bool, list]:
    """
    Find full years to run imputation for based on NaNs in a series.

    The series is grouped into months, where every full year with at least
    one missing value for the month will be marked as a year to impute for.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Pre-cleaned KNMI dataset containing the data of interest.
    param_col : str, optional
        Name of the cleaned parameter column for which to run the check.
        The default value is "rain_sum".

    Returns
    -------
    tuple[bool, list]
        Result tuple containing the following items:
        - `try_impute`: bool indicating whether to continue the imputation.
        - `years_to_impute`: full years to fetch imputation data for.

    Notes
    -----
    - Instead of imputing, it is advised to drop data if either only
    a few datapoints are missing (to save time) or when almost the 
    entire years of data are missing (too few data-points to test for).
    
    """
    # Set non-requirement of imputation as default
    try_impute = False

    # Safeguard function from changing input DataFrame in any way
    df_c = df_clean.copy()

    # Ensure that "date" is in the columns
    if df_c.index.name == "date":
        df_c = df_c.reset_index()

    # Create custom aggregator for counting NaN percentage per group
    agg_lambda = lambda x: 100.0 * np.mean(np.isnan(x))
    agg_func = pd.NamedAgg(column=param_col, aggfunc=agg_lambda)
    
    # Define grouper to group on dates by month, keep month starts as label
    grouper_obj = pd.Grouper(key="date", freq="MS")

    # Perform the grouping to the DataFrame to get monthly NaN percentages
    df_c_gr = df_c.groupby(grouper_obj).agg(month_nan_perc=agg_func)

    # Add column to indicate the year
    df_c_gr["year"] = df_c_gr.index.year

    # Find unique years to run imputation for
    years_to_impute = df_c_gr[df_c_gr["month_nan_perc"] > 0]["year"].tolist()
    years_to_impute = list(set(years_to_impute))

    # Sort the years chronologically (if any)
    years_to_impute.sort()

    # Mark 'try_impute' as True if there are years with missing data found
    if len(years_to_impute) > 0:
        try_impute = True

    # Return whether to look for years to impute for; and if so, which ones
    return (try_impute, years_to_impute)


def find_imputation_stations(stn_code: int,
                             years_to_impute: list[int],
                             param_col: str = "rain_sum",
                             max_nr_impute_stns: int = 5,
                             min_imp_corr: float = 0.5
                             ) -> tuple[bool, list, pd.DataFrame]:
    """
    Find (highly-)correlating KNMI stations for potential imputation.

    Retrieves data of the same parameter from other KNMI stations to be
    able to be used as an imputation matrix in a later step. This dataset
    will cover the full years in which any missing data was found, as
    specified in `years_to_impute`.

    Parameters
    ----------
    stn_code : int
        Three-digit KNMI station code (ID), e.g.: 260 for De Bilt.
    years_to_impute : list[int]
        List of years to fetch other KNMI station data for.
    param_col : str, optional
        Name of the cleaned parameter column for which to fetch values.
        The default value is "rain_sum".
    max_nr_impute_stns : int, optional
        Maximum number of other-station data to use for running the
        imputation algorithm in a later step. The default value is 5.
    min_imp_corr : float, optional
        Minimum (absolute) correlation that is required between the
        data of the station of interest and the other stations to be
        part of the final result. The default value is 0.5.

    Returns
    -------
    tuple[bool, list, pd.DataFrame]
        Result tuple containing the following items:
        - `run_impute`: bool indicating whether to run the imputation next.
        - `impute_stn_codes`: Three-digit station codes to use for imputation.
        - `df_imp`: df with original (to-impute) and fetched series.

    Notes
    -----
    - Limit the number of stations to use as imputation stations, since each
     additional station will increase the running time of the later imputation
     algorithm non-linearly. Between 3 to 5 should usually be more than enough.
    - The returned data might have less than `max_nr_impute_stns` columns to
    return; this happens if less stations having >= `min_imp_corr` were found.
    - Negative correlations might still yield good imputation results; for this
    reason the `min_imp_corr` is evaluated in terms of its absolute value.
    
    """
    # Find associated parameter code for KNMI request
    param_code = param_code_from_col(param_col)
    
    # Placeholders for yearly df list and imputation DataFrame
    df_implist_y = []
    df_imp_raw = pd.DataFrame()

    # Split the request in individual years to prevent service overflow
    for year in years_to_impute:
        df_imp_y = (knmi_meteo_ingest
                    .knmi_meteo_to_df(meteo_stns_list=None,
                                      meteo_params_list=[param_code],
                                      start_date=datetime.date(year, 1, 1),
                                      end_date=datetime.date(year, 12, 31),
                                      mode="day"))
    
        df_implist_y.append(df_imp_y)

    # Concatenate each non-empty yearly series to full history
    # Note: 'ignore_index' used to deduplicate indexes from yearly DataFrames
    if len(df_implist_y) > 0:
        df_imp_raw = pd.concat([df for df in df_implist_y if not df.empty],
                                ignore_index=True)
    
    # Apply transformations to clean the raw dataset
    df_imp = knmi_meteo_transform.transform_param_values(df_imp_raw)

    # Pivot data with "date" as index, "station_code" as cols
    df_imp = (df_imp.pivot(index="date",
                           columns="station_code"))

    # Flatten pivot table to single index
    df_imp.columns = df_imp.columns.get_level_values(1)

    # Get correlation scores
    corr_scores = (df_imp
                   .corrwith(df_imp[stn_code])
                   .sort_values(ascending=False))
    
    # Drop NaNs from the correlation scores
    corr_scores = corr_scores[~corr_scores.isna()]

    # Find up to N best-correlating stations for imputation years;
    # note: less stations will be used in case of empty results

    # Get maximum index: minimum of N and non-NaN corr elements
    max_corr_idx = min(1 + max_nr_impute_stns, len(corr_scores))
    impute_stn_codes = corr_scores[1 : max_corr_idx].index.tolist()

    # Set flag to run imputation procedure at False by default
    run_impute = False

    # Print warning message in case of all-NaN correlations
    warn_msg = ("No matching stations found for MICE imputation; use "
               "another method or drop missing data instead of imputing.")
    if len(impute_stn_codes) == 0:
        print(warn_msg)
    else:
        # Only run impute if max. found correlation is at least `min_imp_corr`
        if max(np.abs(corr_scores[1 : max_corr_idx])) >= min_imp_corr:
            # If all OK, set `run_impute` to True
            run_impute = True
        else:
            warn_msg = (f"None of the stations show >= |{min_imp_corr}| corr.;"
                        " use another method or drop missing data instead.")
            print(warn_msg)

    # Return whether to run impute later, which stns, and imp. dataset 
    return (run_impute, impute_stn_codes, df_imp)


def filter_datacols_to_impute(df_imp: pd.DataFrame,
                              target_col: str | int,
                              feature_cols: list) -> pd.DataFrame:
    """
    Filter DataFrame cols to always match target and feature cols.

    Ensures that the target column for imputation is always presented
    as the first column; this may speed up imputation later on.

    Parameters
    ----------
    df_imp : pd.DataFrame
        DataFrame containing at least columns `target_col` and `feature_cols`.
    target_col : str or int
        Target column for imputation later on (e.g. 260 for De Bilt).
    feature_cols : list
        Feature columns to use to impute target column with.

    Returns
    -------
    df_imp_filt : pd.DataFrame
        DataFrame with columns [`target_col`] + `feature_cols`, with
        `target_col` always as the first column.

    """
    # Define colums to keep for imputation calculation
    keep_cols = [target_col] + feature_cols

    # Apply column filter
    df_imp_filt = df_imp.loc[:, df_imp.columns.isin(keep_cols)]

    # Ensure that our target impute column is always the first; note:
    # this is no hard requirement, but may speed up imputation later on
    df_imp_filt = df_imp_filt[[target_col] + feature_cols]

    return df_imp_filt


def impute_vals_from_targetcol(df_imp: pd.DataFrame, 
                               target_col: str | int,
                               param_col: str = "rain_sum",
                               r2_min_thresh: float = 0.50,
                               print_progress: bool = True) -> pd.DataFrame:
    """
    Impute DataFrame values from `target_col` by use of a MICE algorithm.

    Tries to fit a wide variety of MICE imputation models and returns the
    result of the best-scoring fit (assuming R^2 exceeds `r2_min_thresh`).

    Parameters
    ----------
    df_imp : pd.DataFrame
        DataFrame containing only `target_col` and feature columns.
    target_col : str or int
        Target column for imputation (e.g. 260 for De Bilt).
    param_col : str, optional
        Parameter column in which the imputation vals will be stored.
        The default value is "rain_sum".
    r2_min_thresh : float, optional
        Minimum R^2 value that the best-fit model should score in order
        for calc. imputed values to be interpreted as useful at all.
        The default value is 0.50.
    print_progress : bool, optional
        Whether to print statement of the intermediary steps, mainly when
        applying the fits to the imputation models.
        The default value is True.

    Returns
    -------
    df_imputed or None: pd.DataFrame or None
        DataFrame containing only the filled values for `target_col`, 
        renamed to "`param_col`_imputed". 
        If None is returned, the imputation R^2 threshold on `target_col`
        was not reached.

    Notes
    -----
    - This procedure may take a while to run because of hyperparameter
    tuning of many different models to find the best imputation model.
    Change the `get_models_params_grids()` function in the MICE
    imputation script to exclude gradient boosting and random forests
    for a significant speed-up. The use of simpler models should give
    sufficiently accurate results already in most cases.
    - This function also takes relatively longer to run if there
    are very few non-NaN values to calculate the imputations for.
    - In any production environment: store the best-fit model reference
    and only run MICE-algorithm for that best fit (since fitting the best
    model takes by far most runtime). You will need to write your own
    function to do so.

    """
    # Fit best MICE imputation model on 'target_col' in dataset
    (best_imputer, results_dict, _) = (
        mice_imputation_utils.fit_best_df_imputer_on_targetcol(
            df_imp=df_imp,
            target_col=target_col,
            print_progress=print_progress))

    # Only use imputed data if the r2 is good enough!
    if results_dict["r2"] < r2_min_thresh:
        warn_msg = (f"Best-fit R^2 ({results_dict["r2"].round(4)}) too low to "
                    "produce reliable imputes; continuing without imputing "
                    "NaNs. If you still wish to use imputed data in this "
                    "case, re-run and decrease threshold `r2_min_thresh`.")
        print(warn_msg)

        # Return an empty result to show that no imputation was performed
        return None

    # Fit and run the best imputer on the *full* to-impute dataset
    df_imputed = mice_imputation_utils.run_best_imputer_on_dataset(
        df_imp, best_imputer)

    # Only keep imputed values that were initially missing in dataset
    df_imputed["init_val"] = df_imp[target_col]
    df_imputed = df_imputed[df_imputed["init_val"].isna()]
    df_imputed = df_imputed.loc[:, [target_col]]

    # Convert index type to DateTime
    df_imputed.index = pd.to_datetime(df_imputed.index)

    # Rename columns and re-instate 'date' column; return the result
    df_imputed = (df_imputed
                  .rename(columns={target_col: param_col + "_imputed"})
                  .reset_index())

    return df_imputed


def merge_measured_and_imputed_data(df_data: pd.DataFrame, 
                                    df_imputed: pd.DataFrame | None,
                                    param_col: str,
                                    round_to_n_decimals: int = 2
                                    ) -> pd.DataFrame:
    """
    Merge measured and imputed DataFrames to a single DataFrame.

    The returned result will contain separate columns for measured
    and imputed data.

    Parameters
    ----------
    df_data : pd.DataFrame
        DataFrame containing the original data in col `param_col`.
    df_imputed : pd.DataFrame or None
        DataFrame with the imputed data in col "`param_col` + _imputed".
    param_col : str
        Parameter column in which the values (measured / imputed) are stored.
    round_to_n_decimals : int, optional
        Number of decimals to round the final results to.
        The default value is 2.

    Returns
    -------
    df_data_all : pd.DataFrame
        DataFrame with separate cols for measured and (any) imputed vals.

    Notes
    -----
    - Also recommended to use as part of workflows in which imputation
    was not needed or successful, to guarantee a fixed df format,
    regardless of whether imputation took place or not.

    """

    # Merge imputed data with measured data if imputation was run
    if isinstance(df_imputed, pd.DataFrame):
        df_data_all = (df_data.merge(df_imputed, how="left", on="date")
                       .round(round_to_n_decimals))
    
    # If no imputation was run, only add all-NaN imputation column
    else:
        df_data_all = df_data.round(round_to_n_decimals)

        # Ensure that "date" is a separate column in the output df
        if "date" not in df_data_all.columns:
            df_data_all = df_data.reset_index()

        df_data_all[param_col + "_imputed"] = np.nan

    return df_data_all


def measured_imputed_to_one_series(df_data_all: pd.DataFrame, 
                                   param_col: str) -> pd.DataFrame:
    """
    Combine measured and imputed data to a single series.

    Parameters
    ----------
    df_data_all : pd.DataFrame
        DataFrame of measured and imputed values that should at least contain
        columns `param_col` (measured) and "`param_col` + _imputed" (imputed).
    param_col : str
        Parameter column in which the (measured / imputed) values are stored.

    Returns
    -------
    df_data : pd.DataFrame
        DataFrame with column "`param_col`" with measured and imputed values,
        and column "is_imputed" to indicate whether the value in the same row
        was imputed (True) or measured (False).

    """
    # Copy DataFrame object to prevent changing input df in-place
    df_data = df_data_all.copy()

    # Keep track of fact whether data was imputed or not
    df_data["is_imputed"] = (df_data[param_col].isna() 
                             & ~df_data[param_col + "_imputed"].isna())

    # Create a fully-filled column with mixed real and imp. values;
    # start by filling the column with the originally measured data
    df_data[param_col + "_all"] = df_data[param_col].copy()

    # Fill the "_all" column with imputed data wherever applicable
    filt_idxs = df_data[df_data[param_col + "_all"].isna()].index
    df_data.loc[filt_idxs, param_col + "_all"] = (
        df_data[param_col + "_imputed"].copy())

    # Only keep date, parameter column and imputation label (yes/no)
    keep_cols = ["date", param_col + "_all", "is_imputed"]
    df_data = df_data[keep_cols]

    # Rename filled column back to original parameter name
    rename_cols = {param_col + "_all": param_col}
    df_data = df_data.rename(columns=rename_cols)

    # Return the result
    return df_data


def imputation_workflow(df_cleaned: pd.DataFrame,
                        param_col: str,
                        stn_code: str | int,
                        print_progress: bool = True) -> pd.DataFrame:
    """
    Runs a full imputation workflow for one KNMI station's parameter.
    
    The steps of the full workflow are summarized as follows:
    - Step 1: Find whether there are missing values; get covering years.
    - Step 2: If missing values found, find any non-NaN values of other stns.
    - Step 3: If reasonably correlating other stns found: use to impute,
    using the best-fit model for iterative (MICE) data imputation.
    - Step 4: Merge measured and imputed data to one DataFrame (if available)
    - Step 5: Combine measured and imputed data to one series (if available)

    The format of the output DataFrame will be  the same, regardless of 
    whether imputation took place or not. In case of no imputation,
    "is_imputed" will be all-False.

    Parameters
    ----------
    df_cleaned : pd.DataFrame
        DataFrame with pre-cleaned data fetched from the KNMI web service.
    param_col : str
        Column name of the parameter to run the imputation procedure for,
        e.g.: "rain_sum" or "evap_ref".
    stn_code : str | int
        Three-digit KNMI station code (ID), e.g.: 260 for De Bilt.
    print_progress : bool, optional
        Whether to print statement of the intermediary steps, mainly when
        applying the fits to the imputation models.
        The default value is True.

    Returns
    -------
    df_data_sel : pd.DataFrame
        DataFrame with column "`param_col`" with measured and imputed values,
        and column "is_imputed" to indicate whether the value in the same row
        was imputed (True) or measured (False).

    Notes
    -----
    - This procedure may take a while to run because of hyperparameter
    tuning of many different models to find the best imputation model.
    Change the `get_models_params_grids()` function in the MICE
    imputation script to exclude Gradient Boosting and Random Forests
    for a significant speed-up. The use of simpler models should give
    sufficiently accurate results already in most cases.
    - This function also takes relatively longer to run if there
    are very few non-NaN values to calculate the imputations for.
    - In any production environment: store the best-fit model reference
    and only run MICE-algorithm for that best fit (since fitting the best
    model takes by far most runtime). You will need to write your own
    function to do so.

    """
    # Step 1: Find whether there are missing values; get covering years
    (try_impute, yrs_to_imp) = check_years_to_impute(df_cleaned,
                                                     param_col = param_col)

    # Step 2: If missing values found, find any non-NaN values of other stns
    run_impute = False
    if try_impute:
        (run_impute, impute_stns, df_imp) = (
            find_imputation_stations(stn_code = stn_code, 
                                     years_to_impute = yrs_to_imp,
                                     param_col = param_col)
        )

    # Set up empty DataFrame of imputed values (in case of no imputation)
    df_imputed = None

    # Step 3: If reasonably correlating other stns found: use to impute,
    # using the best-fit model for iterative (MICE) data imputation
    if run_impute:
        df_imp = filter_datacols_to_impute(df_imp, stn_code, impute_stns)
        df_imputed = impute_vals_from_targetcol(
            df_imp, stn_code, param_col, print_progress = print_progress)
    
    # Step 4: Merge measured and imputed data to one DataFrame (if available)
    df_data_all = merge_measured_and_imputed_data(df_cleaned, df_imputed, 
                                                  param_col = param_col)
    
    # Step 5: Combine measured and imputed data to one series (if available)
    df_data_sel = measured_imputed_to_one_series(df_data_all, param_col)
    
    return df_data_sel


def calc_rain_min_evap_df(df_rain_data: pd.DataFrame,
                          df_evap_data: pd.DataFrame,
                          rain_param_col: str = "rain_sum",
                          evap_param_col: str = "evap_ref") -> pd.DataFrame:
    """
    Calculate precipitation minus evaporation (rain - evap).

    This transformation is needed as a preparation step for calculating the
    SPEI values given historic rainfall and evaporation.

    Any imputed value labels (True / False) will be aggregated to averages
    in the output DataFrame. E.g.: if one of the two (rain, evap) is missing,
    the value for "is_imputed" for that row will be reported as 0.5.

    Parameters
    ----------
    df_rain_data : pd.DataFrame
        Precipitation dataset; should at least contain cols `rain_param_col`
        and "is_imputed".
    df_evap_data : pd.DataFrame
        Evaporation dataset; should at least contain cols `evap_param_col`
        and "is_imputed".
    rain_param_col : str, optional
        Column name used to identify the precipitation (rainfall) data.
        The default value is "rain_sum".
    evap_param_col : str, optional
        Column name used to identify the evaporation data.
        The default value is "evap_ref".

    Returns
    -------
    df_pcp_data : pd.DataFrame
        DataFrame with columns "date", calculated "rain_min_evap" and 
        row-averaged "is_imputed" (0, 0.5 or 1 for each row).

    Notes
    -----
    - Make sure that the precipitation and evaporation datasets are both
    from the same unit (e.g. mm) to ensure correct subtraction.
    """
    # Merge rain and evap datasets on date (use "left" to keep index)
    df_pcp_data = df_rain_data.merge(df_evap_data, on="date", how="left",
                                     suffixes=('_rain', '_evap'))

    # Summarize final imputation marker column as average imp. per row
    imp_cols = ["is_imputed_rain", "is_imputed_evap"]
    df_pcp_data["is_imputed"] = (df_pcp_data[imp_cols].astype(float)
                                 .sum(axis=1, skipna=False) / len(imp_cols))

    # Calculate (precipitation - evaporation) from the rain and evap columns
    df_pcp_data["rain_min_evap"] = (df_pcp_data[rain_param_col] 
                                    - df_pcp_data[evap_param_col])

    # Only keep transformed columns; return the result
    keep_cols = ["date", "rain_min_evap", "is_imputed"]
    df_pcp_data = df_pcp_data[[col for col in keep_cols]]

    return df_pcp_data


def agg_to_grouped(df_data: pd.DataFrame, param_col: str, N_months: int
                   ) -> pd.DataFrame:
    """
    Aggregate DataFrame timeseries to N-monthly bins.

    Use a one-month Grouper on key "date", where the translation to 
    N-monthly bins is performed using a rolling window over the previous
    N-1 months and the month itself.

    This function can be used as a step to calculate SP(E)I-`N` values.

    Parameters
    ----------
    df_data : pd.DataFrame
        DataFrame of timeseries values with columns "date", `param_col`
        and "is_imputed".
    param_col : str
        Column name of the parameter to group values for, e.g.: "rain_sum".
    N_months : int
        Number of months to use for each grouped aggreagation bin.

    Returns
    -------
    df_time_grouped : pd.DataFrame
        Monthly DataFrame with N-month rolling-window values as output.

    """

    # Set aggregation rules; any aggregate that still has any NaN(s) 
    # in its value is summed to NaN as a whole (to prevent dist-fit errors)
    lambda_sum_f = lambda x: np.nan if x.isnull().any() else x.sum()
    agg_dict = {param_col: lambda_sum_f, "is_imputed": "mean"}
    
    # Aggregate dataset to monthly sums
    month_grouper = pd.Grouper(key="date", freq="MS")
    df_month_grouped = (df_data
                        .groupby(month_grouper)
                        .agg(agg_dict))
    
    # Use rolling window to get aggregated totals for N months
    df_time_grouped = df_month_grouped.rolling(window=N_months).agg(agg_dict)

    return df_time_grouped


def fit_distr_to_series(df_grouped: pd.DataFrame, param_col: str,
                        distr_name: str = "best") -> tuple[pd.DataFrame, str]:
    """
    Fit one or more distributions to a (grouped) timeseries.

    Uses the L2-norm distance between the fitted distribution's CDF and the
    actual empirical CDF (ECDF) to evaluate the quality of the distr. fit.

    If a `distr_name` is specified, a fit will only be made for that
    specific distribution. The reason for this is that it is customary
    in literature to fit only to e.g. a "gamma" distribution for SPI.

    Parameters
    ----------
    df_grouped : pd.DataFrame
        DataFrame containing (grouped) timeseries in column `param_col`.
    param_col : str
        Column name of the parameter to fit values for, e.g.: "rain_sum".
    distr_name : str, optional
        Name of the distribution to apply a fit for to the data. If set
        at "best", this function will try to find the best-fit to the data 
        in `df_grouped[param_col]` from the available distributions. If set
        at one of the available distributions, a fit will only be made using
        that specific distribution. The default value is "best".

    Returns
    -------
    tuple[pd.DataFrame, str]
        Result tuple containing the following items:
        - `df_distr`: DataFrame with calc. distr. PDFs, CDFs, and the ECDF
        - `best_distr`: Name of best-fit distribution.

    Notes
    -----
    - A correction of +0.1 mm is used for zero values to ensure compliance
    with e.g. fitting a Gamma distribution (not defined at 0).
    - The available distributions to fit for are the same as those used in
    the following scientific paper: https://doi.org/10.1175/JAMC-D-14-0032.1.

    """
    # Define list of supported distributions (same names as in "scipy"!)
    supported_distrs = ["gamma", "expon", "lognorm", "weibull_min"]
    
    if distr_name.lower() == "best":
        # Fit to all supported distributions and choose the best one (default)
        dist_names = supported_distrs
    elif distr_name.lower() in supported_distrs:
        # Only fit to desired distribution if set as explicit input
        dist_names = [distr_name.lower()]
    else:
        # Raise error if any other distribution was chosen as input
        err_msg = (f"Distribution '{dist_name}' is not supported; if needed, "
                   "extend the list of supported distributions and test if "
                   "the code also works for your example. Currently supported"
                   f" are: {', '.join(d for d in supported_distrs)}")
        raise NotImplementedError(err_msg)

    # Prepare DataFrame for storing results for each distribution (_pdf, _cdf)
    df_distr = pd.DataFrame()
    df_distr[param_col] = df_grouped[param_col].copy()

    # Set any total of 0 mm to 0.1 mm (avoid zero-issues with gamma)
    # Important: in arid regions, this may not suffice for a good fit
    filter_idxs = df_distr[df_distr[param_col] == 0.0].index
    df_distr[df_distr.loc[filter_idxs, param_col]] = 0.1

    # Get PDFs and CDFs for each best-fit distr. (using Max. Likelihood Est.)
    for dist_name in dist_names:
        # Get distribution object from SciPy
        dist = getattr(scs, dist_name)

        # Fit distribution parameters using MLE method to get best params
        params = dist.fit(df_distr[param_col].dropna(), method="mle", loc=0)
    
        # Use best-params result to calculate PDF and CDF for this distr.
        df_distr[dist_name + "_pdf"] = dist.pdf(df_distr[param_col], *params)
        df_distr[dist_name + "_cdf"] = dist.cdf(df_distr[param_col], *params)

    # Also calculate empirical CDF (ECDF) for checking goodness-of-fit
    ecdf_res = scs.ecdf(df_grouped[param_col].dropna())
    df_distr["ecdf"] = ecdf_res.cdf.evaluate(df_grouped[param_col])

    # Ensure empirical CDF (ECDF) is always NaN for NaN-values (and not 1.0)
    df_distr.loc[df_distr[param_col].isna(), "ecdf"] = np.nan

    # Find best-fitting distr. if 'best' is chosen (default)
    best_distr = distr_name
    if distr_name.lower() == "best":

        # Use L2-norm distance to ECDF to determine best fit
        distances = {}
        for dist_name in dist_names:
            # Squared differences (L2-norm)
            squared_diff = (df_distr["ecdf"]
                            - df_distr[dist_name + "_cdf"]) ** 2
            # Sum of squared differences (L2-norm distance)
            distances[dist_name] = squared_diff.sum()

        best_distr = min(distances, key=distances.get)

    return (df_distr, best_distr)


def cdf_bestfit_to_z_scores(df_distr: pd.DataFrame,
                            best_distr: str) -> np.array:
    """Convert CDF percentile to Z-scores using inv. norm."""
    # Apply inverse normal distribution to best-fit CDF;
    # this will give us a Z-score ('Standardized Index')
    norm_ppf = scs.norm.ppf(df_distr[best_distr + "_cdf"])
    norm_ppf[np.isinf(norm_ppf)] = np.nan

    # Return result (Z-score; or 'Standardized Index')
    return norm_ppf


def calculate_nmonth_spi(df_data: pd.DataFrame, 
                         param_col: str,
                         N_monthlist: list[int] = [1, 3, 6, 9, 12, 24],
                         distr_name: str = "best") -> pd.DataFrame:
    """
    Calculate SPI-N scores for a series of precipitation data.

    Parameters
    ----------
    df_data : pd.DataFrame
        Precipitation dataset with values listed in column `param_col`.
    param_col : str
        Column name of the precipitation parameter, e.g.: "rain_sum".
    N_monthlist : list[int], optional
        List of N-months to calculate SPI values for.
        The default value is [1, 3, 6, 9, 12, 24].
    distr_name : str, optional
        Name of the distribution to use for fitting. If "best", the
        best-fit distribution is used to fit the data.
        The default value is "best".

    Returns
    -------
    df_month : pd.DataFrame
        DataFrame containing monthly calculated SPI-N indexes and 
        aggregated values from input column `param_col`.

    """
    # Create a monthly dataset (so that later indexes will align)
    df_month = agg_to_grouped(df_data, param_col, 1)

    # Now, calculate SPIs and fit best distributions for N-month series
    for N in N_monthlist:
        df_grouped = agg_to_grouped(df_data, param_col, N)
        (df_distr, best_distr) = fit_distr_to_series(df_grouped, param_col, 
                                                     distr_name = distr_name)
        df_month[f"spi_{N}"] = cdf_bestfit_to_z_scores(df_distr, best_distr)

    return df_month


def calculate_nmonth_spei(df_data: pd.DataFrame, 
                          param_col: str,
                          N_monthlist: list[int] = [1, 3, 6, 9, 12, 24],
                          distr_name: str = "best") -> pd.DataFrame:
    """
    Calculate SPEI-N scores for a series of (rain - evap) data.

    The difference with the SPI-N calculation is that negative values
    (indicating precipitation deficits) are now possible. This means that
    we need to offset all data by the minimum value to fit a distribution
    and finally use the inverse of this offset to obtain our final result.

    Parameters
    ----------
    df_data : pd.DataFrame
        Excess precipitation data (rain - evap) with values listed in
        column `param_col`.
    param_col : str
        Column name of the (rain - evap) parameter, e.g.: "rain_min_evap".
    N_monthlist : list[int], optional
        List of N-months to calculate SPEI values for.
        The default value is [1, 3, 6, 9, 12, 24].
    distr_name : str, optional
        Name of the distribution to use for fitting. If "best", the
        best-fit distribution is used to fit the data.
        The default value is "best".

    Returns
    -------
    df_month : pd.DataFrame
        DataFrame containing monthly calculated SPEI-N indexes and 
        aggregated values from input column `param_col`.

    """
    # Create a monthly dataset (so that later indexes will align)
    df_month = agg_to_grouped(df_data, param_col, 1)

    # Add offset to all datapoints to avoid < 0 values for distr. fitting!
    df_minval = df_data[param_col].min()
    df_data_mod = df_data.copy()

    # Only apply offset if min. val. is actually negative (otherwise, proceed)
    if df_minval < 0:
        df_data_mod.loc[:, param_col] += np.abs(df_minval)

    # Now, calculate SPEIs and fit best distributions for N-month series
    for N in N_monthlist:
        df_grouped = agg_to_grouped(df_data_mod, param_col, N)
        (df_distr, best_distr) = fit_distr_to_series(df_grouped, param_col, 
                                                     distr_name = distr_name)
        df_month[f"spei_{N}"] = cdf_bestfit_to_z_scores(df_distr, best_distr)

    return df_month


def get_events_from_z_scores(df_si: pd.DataFrame,
                             vals_col: str,
                             event: str) -> pd.DataFrame:
    """
    Get drought or wetness events from calculated SP(E)I-N values.

    Identifies first and last start indexes of each event, their
    duration (e.g. months) and total magnitude (sum of SP(E)I-N scores).

    Events are defined as subsequent periods in which the SP(E)I-N score
    is at least as far from the mean as the event threshold. This means,
    for example, that a "severe_wetness" event starts in the first month
    with a score above 1.5, and ends whenever the score starts falling
    below 1.5 again.

    Parameters
    ----------
    df_si : pd.DataFrame
        DataFrame containing SP(E)I scores in column `vals_col`.
    vals_col : str
        Column in `df_si` containing the , e.g. "spi-3".
    event : str
        The event type of interest, e.g.: "severe_drought".

    Returns
    -------
    events_df : pd.DataFrame
        DataFrame with listed events, including duration and magnitude.
        The column labels are prepended with `event` and `vals_col`
        for traceability purposes.
    
    Notes
    -----
    - The thresholds indicated in this function are in line with the WMO
    (World Meteorological Organization) definitions at the moment of writing.
    - In some other definitions, events are only marked as ended when the
    SP(E)I-N index completely flips sign (0.0). If you need to apply that 
    definition, please implement your own custom function for this. 
    """
    # Define events and Z-score thresholds
    ev_thrshs = {"extreme_wetness": 2.0, 
                 "severe_wetness": 1.5, 
                 "wetness": 1.0, 
                 "drought": -1.0, 
                 "severe_drought": -1.5,
                 "extreme_drought": -2.0}
    
    # Show error if an incorrect event type was chosen as input
    if event not in ev_thrshs.keys():
        raise ValueError("Invalid 'event'; please choose from this list: "
                         f"{", ".join([e for e in ev_thrshs.keys()])}")
    
    # Create copy of input DataFrame (to avoid modifying input DataFrame)
    df_stzd = df_si.copy()

    # If 'date' is still the index: convert it to a column (for grouping)
    if "date" not in df_stzd.columns:
        df_stzd = df_stzd.reset_index()

    # Mark for each row whether value exceeds threshold value or not;
    # use .gt()-method if > 0; use .lt()-method if <= 0
    gt_or_lt = ("gt" if ev_thrshs[event] > 0 else "lt")
    df_stzd["is_" + event] = getattr(df_stzd[vals_col], 
                                     gt_or_lt)(ev_thrshs[event])

    # Create a unique identifier for each drought event;
    # every new event is part of the event, while row above is not
    is_new_event = (df_stzd["is_" + event] 
                    & (~df_stzd["is_" + event]
                    .shift(fill_value=False)))

    # Convert every new found event to a new ID
    df_stzd[event + "_id"] = is_new_event.cumsum()

    # Filter out (exclude) non-event values 
    df_stzd.loc[~df_stzd["is_" + event], event + "_id"] = None

    # Custom aggregation dictionary
    agg_dict = {
        # First start index of the event
        "first_start_idx": ("date", "min"),
        # Last start index of the event
        "last_start_idx": ("date", "max"),
        # Magnitude of the event (as total sums of the Z-index)
        "magnitude": (vals_col, lambda x: np.sum(np.abs(x)).round(2)),
        # Duration of the event (number of subsequent rows in same event)
        "duration": (vals_col, "count")}

    # Prepend reference for event and SP(E)I-"n" source
    agg_dict = {event + "_" + vals_col + "_" + k: v 
                for k, v in agg_dict.items()}

    # Create overview of all identified events
    events_df = (
        df_stzd[df_stzd["is_" + event]]
        .groupby(event + "_id")
        .agg(**agg_dict)
        .reset_index(drop=True))
    
    return events_df
