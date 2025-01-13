"""KNMI SPI Utils

This script contains the utility / helper functions for the SPI / SPEI
analysis of KNMI weather data.

This script requires that `pandas`, `numpy`, `scipy` and `sklearn`
be installed within the Python environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    TODO: UPDATE TO FUNCTIONS PRESENT IN THIS SCRIPT!
    * load_tf_json - load 'transform' JSON to Python object
    * transform_stations - transform raw station data to cleaned df
    * transform_param_values - transform raw values to cleaned df
    * knmi_hourslot_percentage_df - aggregate KNMI data to hour slots
    * get_multiyr_hourslot_percentages - make multi-year hour slot df
"""

import datetime
import numpy as np
import pandas as pd
import scipy.stats as scs

import knmi_meteo_ingest
import knmi_meteo_transform
import mice_imputation_utils


def validate_station_code(stn_code: int) -> dict:
    """Get station details for a given station code.

    Can be used to check whether the input station code is valid.
    
    Parameters
    ----------
    stn_code : int
        Three-digit KNMI station code (ID), e.g.: 265 for De Bilt.

    Returns
    -------
    stn_dict : dict
        Basic dictionary with 'raw' (uncleaned) station attributes.

    Raises
    ------
    AssertionError
        Occurs if the station code is not found.
        Retry using a valid KNMI station code (ID), e.g.: 265 for De Bilt.

    """
    # Get station details for chosen code (sanity check)
    stations_raw = knmi_meteo_ingest.knmi_load_meteo_stations()

    # Show details of chosen station (should be non-empty)
    stn_dict = (stations_raw[stations_raw["STN"] == stn_code]
                .to_dict(orient="list"))

    # AssertionError message with valid options if station code is not valid
    valid_stns_str = ", ".join(str(x) for x in stations_raw["STN"])
    err_msg = f"Invalid station code (integer) - valid options: {valid_stns_str}."

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


def get_measured_stn_precip_values(stn_code: int,
                                   param_col: str = "rain_sum",
                                   start_year: int = 1901,
                                   end_year: int = 2024) -> pd.DataFrame:
    """"""
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
    
    # 3. Missing indexes in the range should be filled with NaNs
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
    """"""
    # Set non-requirement of imputation as default
    try_impute = False

    # Safeguard function from changing input DataFrame in any way
    df_c = df_clean.copy()

    # Ensure that "date" is in the columns
    if df_c.index.name == "date":
        df_c = df_c.reset_index()

    # Create custom aggregator for counting NaN percentage per group
    agg_func = pd.NamedAgg(column=param_col,
                           aggfunc=lambda x: 100.0 * np.mean(np.isnan(x)))
    
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
                             years_to_impute: list,
                             param_col: str = "rain_sum",
                             max_nr_impute_stns: int = 5,
                             min_imp_corr: float = 0.25
                             ) -> tuple[bool, list, pd.DataFrame]:
    """"""
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
    """"""
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
    """"""
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
                    "case, re-run and increase threshold `r2_min_thresh`.")
        print(warn_msg)

        # Return an empty result to show that no imputation was performed
        return None

    # Fit and run the best imputer on the *full* to-impute dataset
    df_imputed = mice_imputation_utils.run_best_imputer_on_dataset(
        df_imp, best_imputer
    )

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
    """"""

    # Merge imputed data with measured data if imputation was run
    if isinstance(df_imputed, pd.DataFrame):
        df_data_all = (df_data.merge(df_imputed, how="left", on="date")
                       .round(round_to_n_decimals))
    
    # If no imputation was run, only add all-NaN imputation column
    else:
        if "date" not in df_data_all.columns:
            df_data_all = df_data.reset_index()

        df_data_all = df_data.reseround(round_to_n_decimals)
        df_data_all[param_col + "_imputed"] = np.nan

    return df_data_all


def simplify_dataset(df_data_all: pd.DataFrame, param_col: str):
    """"""
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


def agg_to_grouped(df_data: pd.DataFrame, param_col: str, N_months: int):
    """"""

    # Set aggregation rules; any aggregate that still has any NaN(s) 
    # in its value is summed to NaN as a whole (prevent dist-fit errors)
    lambda_sum_func = lambda x: np.nan if x.isnull().any() else x.sum()
    agg_dict = {param_col: lambda_sum_func,
                "is_imputed": "mean"}
    
    # Aggregate dataset to monthly sums
    month_grouper = pd.Grouper(key="date", freq="MS")
    df_month_grouped = (df_data
                        .groupby(month_grouper)
                        .agg(agg_dict))
    
    # Use rolling window to get aggregated totals for N months
    df_time_grouped = df_month_grouped.rolling(window=N_months).agg(agg_dict)

    return df_time_grouped


def fit_distr_to_series(df_grouped: pd.DataFrame, 
                        param_col: str,
                        distr_name: str = "best"):
    """"""
    # Time for fitting a distribution! 

    # We will use (just as in: https://journals.ametsoc.org/view/journals/apme/53/10/jamc-d-14-0032.1.xml):
    # 1. Gamma distribution function (note: totals should be nonzero!)
    # 2. Exponential distribution function
    # 3. Lognormal distribution function
    # 4. Weibull distribution function
    supported_distrs = ["gamma", "expon", "lognorm", "weibull_min"]
    
    if distr_name.lower() == "best":
        # Fit to all supported distributions, fit best (default)
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


    # Also calculate empirical CDF (for checking goodness-of-fit)
    ecdf_res = scs.ecdf(df_grouped[param_col].dropna())
    df_distr["ecdf"] = ecdf_res.cdf.evaluate(df_grouped[param_col])

    # Ensure empirical CDF (ECDF) is always NaN for NaN-values (and not 1.0)
    df_distr.loc[df_distr[param_col].isna(), "ecdf"] = np.nan

    # Find best-fitting distr. if 'best' is chosen (default)
    best_distr = distr_name
    if distr_name.lower() == "best":

        # Use L2-norm distance to determine best fit
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
                            best_distr: str):
    """"""
    # Apply inverse normal distribution to best-fit CDF;
    # this will give us a Z-score (Standardized Index)
    norm_ppf = scs.norm.ppf(df_distr[best_distr + "_cdf"])
    norm_ppf[np.isinf(norm_ppf)] = np.nan

    # Return result (Z-score; Standardized Index)
    return norm_ppf


def calculate_nmonth_spi(df_data: pd.DataFrame, 
                         param_col: str,
                         N_monthlist: list[int] = [1, 3, 6, 9, 12, 24],
                         distr_name: str = "best"):
    """"""
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
                          distr_name: str = "best"):
    """"""
    # Create a monthly dataset (so that later indexes will align)
    df_month = agg_to_grouped(df_data, param_col, 1)

    # Add offset to all datapoints to avoid < 0 values for distr. fitting!
    df_minval = df_data[param_col].min()
    df_data_mod = df_data.copy()

    # Only apply offset if min. val. is actually negative
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
                             event: str):
    """"""
    # Define events and Z-score thresholds
    ev_thrshs = {"extreme_wetness": 2.0, 
                 "severe_wetness": 1.5, 
                 "wetness": 1.0, 
                 "drought": -1.0, 
                 "severe_drought": -1.5,
                 "extreme_drought": -2.0}
    
    if event not in ev_thrshs.keys():
        raise ValueError("Invalid 'event'; please choose from this list: "
                         f"{", ".join([e for e in ev_thrshs.keys()])}")
    
    # Create copy of input DataFrame (to avoid changing input DataFrame)
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
        "duration": (vals_col, "count"),
    }

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