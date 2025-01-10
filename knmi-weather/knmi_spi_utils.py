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
    param_code = next((d for d in param_dct if
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
    df_rainlist_y = []

    # Split the request in individual years to prevent service overflow
    for year in range(start_year, end_year+1):
        df_rain_y = (knmi_meteo_ingest
                     .knmi_meteo_to_df(meteo_stns_list=[stn_code],
                                       meteo_params_list=[param_code],
                                       start_date=datetime.date(year, 1, 1),
                                       end_date=datetime.date(year, 12, 31),
                                       mode="day"))
    
        df_rainlist_y.append(df_rain_y)

    # Concatenate each non-empty yearly series to full history
    # Note: 'ignore_index' used to deduplicate indexes from yearly DataFrames
    df_rain_raw = pd.concat([df for df in df_rainlist_y if not df.empty],
                            ignore_index=True)
    
    # Apply transformations to clean the raw dataset
    df_rain = knmi_meteo_transform.transform_param_values(df_rain_raw)

    # Cut off leading and trailing NaNs from history dataset;
    # in this way we get our actual historical start and end dates
    min_idx = (df_rain[[param_col]]
               .apply(pd.Series.first_valid_index).max())
    max_idx = (df_rain[[param_col]]
               .apply(pd.Series.last_valid_index).min())
    df_rain = df_rain.loc[min_idx: max_idx, :]

    # Now, re-index DataFrame so that it:
    # 1. Always starts at first day of the first month found
    first_date = df_rain["date"].iloc[0]
    last_date = df_rain["date"].iloc[-1]
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
    df_rain.index = pd.to_datetime(df_rain["date"])

    # Reindex to full months
    df_rain = df_rain.reindex(full_mnths_index)

    # Remove station code and "incomplete" date column
    drop_cols = ["date", "station_code"]
    df_rain = df_rain.loc[:,~df_rain.columns.isin(drop_cols)]

    # Rename full-month index to "date"
    df_rain = df_rain.rename_axis(index="date")

    # Return DataFrame with "date" as index, `param_col` as column
    return df_rain


def check_years_to_impute(df_rain: pd.DataFrame,
                          param_col: str = "rain_sum") -> tuple[bool, list]:
    """"""
    # Set non-requirement of imputation as default
    try_impute = False

    # Safeguard function from changing input DataFrame in any way
    df_r = df_rain.copy()

    # Ensure that "date" is in the columns
    if df_r.index.name == "date":
        df_r = df_r.reset_index()

    # Create custom aggregator for counting NaN percentage per group
    agg_func = pd.NamedAgg(column=param_col,
                           aggfunc=lambda x: 100.0 * np.mean(np.isnan(x)))
    
    # Define grouper to group on dates by month, keep month starts as label
    grouper_obj = pd.Grouper(key="date", freq="MS")

    # Perform the grouping to the DataFrame to get monthly NaN percentages
    df_r_gr = df_r.groupby(grouper_obj).agg(result=agg_func)

    # Add column to indicate the year
    df_r_gr["year"] = df_r_gr.index.year

    # Find unique years to run imputation for
    years_to_impute = df_r_gr[df_r_gr["result"] > 0]["year"].tolist()
    years_to_impute = list(set(years_to_impute))

    # Sort the years chronologically (if any)
    years_to_impute.sort()

    # Mark 'try_impute' as True if there are years with missing data found
    if len(years_to_impute) > 0:
        try_impute = True

    # Return whether to look for years to impute for; and if so, which ones
    return (try_impute, years_to_impute)


def find_imputation_stations(years_to_impute: list,
                             stn_code: int,
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
            target_colname=target_col,
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
                                    param_col: bool,
                                    round_to_n_decimals: int = 2
                                    ) -> pd.DataFrame:
    """"""

    # Merge imputed data with measured data if imputation was run
    if isinstance(df_imputed, pd.DataFrame):
        df_data_all = (df_data.merge(df_imputed, how="left", on="date")
                       .round(round_to_n_decimals))
    
    # If no imputation was run, only add all-NaN imputation column
    else:
        df_data_all = df_data.round(round_to_n_decimals)
        df_data_all[param_col + "_imputed"] = np.nan

    return df_data_all


def simplify_dataset(df_data_all: pd.DataFrame, param_col: str | int):
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

    # Return the result
    return df_data


def agg_to_grouped(df_data: pd.DataFrame, param_col: str | int, N: int):
    """"""

    # Set aggregation rules; any aggregate that still has any NaN(s) 
    # in its value is summed to NaN as a whole (prevent dist-fit errors)
    lambda_sum_func = lambda x: np.nan if x.isnull().any() else x.sum()
    agg_dict = {param_col + "_all": lambda_sum_func,
                "is_imputed": "mean"}
    
    # Aggregate dataset to monthly sums
    month_grouper = pd.Grouper(key="date", freq="MS")
    df_month_grouped = (df_data
                        .groupby(month_grouper)
                        .agg(agg_dict))
    
    # Use rolling window to get aggregated totals for N months
    df_time_grouped = df_month_grouped.rolling(window=N).agg(agg_dict)

    return df_time_grouped


def fit_distr_to_series(df_grouped: pd.DataFrame, 
                        param_col: str | int,
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
    df_distr[param_col] = df_grouped[param_col + "_all"].copy()

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
    ecdf_res = scs.ecdf(df_grouped[param_col + "_all"].dropna())
    df_distr["ecdf"] = ecdf_res.cdf.evaluate(df_grouped[param_col + "_all"])

    best_distr = distr_name

    # Find best-fitting distr. if 'best' is chosen (default)
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


def inverse_normal_bestfit_cdf(df_distr: pd.DataFrame,
                               best_distr: str):
    """"""
    # Apply inverse normal distribution to best-fit CDF;
    # this will give us a Z-score (Standardized Index)
    norm_ppf = scs.norm.ppf(df_distr[best_distr + "_cdf"])
    norm_ppf[np.isinf(norm_ppf)] = np.nan

    # Return result (Z-score; Standardized Index)
    return norm_ppf
