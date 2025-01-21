"""MICE Imputation Utils

This script contains the utility / helper functions for applying a MICE
(Multivariate Imputation by Chained Equations) procedure, where the best-scoring 
model based on a single target column is selected.

This script requires that `pandas`, `numpy`, and `sklearn` be installed 
within the Python environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * get_models_params_grids - get sklearn models, param grids to test
    * fit_best_df_imputer_on_targetcol - fit best MICE model, score on one col
    * run_best_imputer_on_dataset - run best imputer on the entire dataset

"""
import copy
import numpy as np
import pandas as pd

from sklearn.linear_model import (LinearRegression, Ridge, 
                                  Lasso, BayesianRidge)
from sklearn.ensemble import (RandomForestRegressor, 
                              GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid

# Note: IterativeImputer is still experimental; import as such
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.impute._base import _BaseImputer


def get_models_params_grids(r_seed: int | None = 42) -> dict:
    """Get sklearn models and param grids to test for.
    
    The dict below can be edited to the user's preferences.
    """
    models_params_grids = {
        "LinearRegression":
            (LinearRegression(), {}),
        "Ridge": (
            Ridge(),
            {
            "alpha": [0.1, 1.0, 10.0]
            }
        ),
        "Lasso": (
            Lasso(),
            {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0]
            }
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=r_seed),
            {
            "n_estimators": [10, 20, 50, 100],
            "max_depth": [3, 5, 7, 10]
            }
        ),
        "GradientBoostingRegressor": (
            GradientBoostingRegressor(random_state=r_seed),
            {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7, 10]
            },
        ),
        "BayesianRidge": (
            BayesianRidge(),
            {
            "alpha_1": [1e-6, 1e-5],
            "alpha_2": [1e-6, 1e-5],
            "lambda_1": [1e-6, 1e-5],
            "lambda_2": [1e-6, 1e-5],
            },
        ),
        "KNeighborsRegressor": (
            KNeighborsRegressor(),
            {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"],
            },
        ),
    }

    return models_params_grids


def fit_best_df_imputer_on_targetcol(df_imp: pd.DataFrame, 
                                     target_col: str | int, 
                                     test_frac: float = 0.25,
                                     r_seed: int | None = 42,
                                     rmse_r2_alpha: float = 0.5,
                                     print_progress: bool = True
                                     ) -> tuple[IterativeImputer, dict,
                                                pd.DataFrame]:
    """
    Test and fit MICE models on dataset; return best-scoring on target col.

    Tries all prespecified models and performs hyperparameter search to
    find the best-scoring variety fitting the dataset's `target_col`.

    The exact models and hyperparameter grids to test on can be set via the
    dictionary in function `get_models_params_grids` for more flexibility.
    
    Parameters
    ----------
    df_imp : pd.DataFrame
        DataFrame with to-impute dataset; should contain 'target_col'
        and at least one other column that is not fully empty / NaN.
    target_col : str or int
        The name or ID of the target column to score the imputation on.
    test_frac : float, optional
        The fraction of data in `target_col` to randomly set to NaN.
        Used for testing the quality-of-fit of the MICE model.
        The default value is 0.25.
    r_seed : int or None, optional
        Random seed value to use in case of desired reproducilibity.
    rmse_r2_alpha : float, optional
        Weighing factor for emphasizing rmse or R^2 for finding the
        best-scoring model. Set near 0 for RMSE-only; near 1 for R^2-only.
        The default value is 0.5.
    print_progress : bool
        Whether to print statements about the fitting and testing progress.
        The default value is True.

    Returns
    -------
    tuple[IterativeImputer, dict, pd.DataFrame]
        The best-scoring imputer instance, summary of results (mse, rmse, r2),
        and the imputed DataFrame.

    Notes
    -----
    For optimal results, run the best-fitting imputer on the entire to-impute
    dataset after completing this function. The result returned here is only
    trained on (1 - test_frac) part of the available data to train on.

    Example code to run to get the optimal result:
    >>> # 1. Run this function to get best imputer, trained on e.g. 75%
    >>> res_tuple = fit_best_df_imputer_on_targetcol(df_imp, target_col)
    >>> # 2. Then, use function below on 100% of dataset as real run
    >>> df_imputed = run_best_imputer_on_dataset(df_imp, res_tuple[0])

    """
    # Set random seed value for result reproducibility (if defined)
    if r_seed:
        np.random.seed(r_seed)

    # Use min and max values from dataset as imputation value limits
    min_impval = np.nanmin(df_imp.values)
    max_impval = np.nanmax(df_imp.values)
 
    # Create dataset of "complete" data for target col, where
    # at least one of the non-target col vals is not NaN
    complete_data = (df_imp
                     .dropna(subset=[target_col])
                     .dropna(subset=(df_imp.columns
                                     .difference([target_col])),
                             how="all"))
    incomplete_data = complete_data.copy()

    # Introduce random missingness in target column (here: `test_frac`)
    missing_mask = np.random.rand(incomplete_data.shape[0]) < test_frac
    incomplete_data.loc[missing_mask, target_col] = np.nan

    # Test values are only present in the overview of complete data
    test_vals = complete_data.loc[missing_mask, target_col].to_numpy()

    # Set base imputer settings (we will only vary the model used later on)
    base_imputer = IterativeImputer(max_iter=20, tol=1e-4, random_state=r_seed,
                                    min_value=min_impval, max_value=max_impval,
                                    imputation_order="roman")

    # Get models and parameter settings to run imputation testing for
    models_and_params = get_models_params_grids(r_seed)

    # Initialize variables to track the best model and its performance
    best_model_name = None
    best_params = None
    best_score = float("inf")
    best_imputer = None

    # Loop through each model and parameter grid
    for model_name, (model, param_grid) in models_and_params.items():
        
        if print_progress:
            print(f"Testing '{model_name}' model(s):")
    
        # Iterate through all parameter combinations
        for params in ParameterGrid(param_grid):
            # Create a new estimator with the current parameters
            estimator = model.set_params(**params)
        
            # Set up the imputer with this estimator
            imputer = base_imputer.set_params(estimator=estimator)
        
            # Fit the imputer on the incomplete data
            imputed_data = imputer.fit_transform(incomplete_data)
        
            # Calculate the MSE and R^2 for the imputed values
            mse = mean_squared_error(
                test_vals,
                imputed_data[missing_mask, 0])

            r2 = r2_score(
                test_vals,  
                imputed_data[missing_mask, 0])
        
            # Combine the RMSE and R^2 into a single score
            score = (rmse_r2_alpha * np.sqrt(mse) 
                     - (1 - rmse_r2_alpha) * (1 - r2))

            if print_progress:
                print(f"  Params: {params}, MSE: {mse:.4f}, R^2: {r2:.4f}, "
                      f"score: {score:.4f}")
        
            # Update the best model if this is the best scoring one
            if score < best_score:
                best_score = score
                best_params = params
                best_model_name = model_name
                best_imputer = copy.deepcopy(imputer)

    # Print the best model and parameters
    if print_progress:
        print("\nBest Model and Parameters:")
        print(f"  Model: {best_model_name}")
        print(f"  Parameters: {best_params}")
        print(f"  Best score: {best_score:.4f}")

    # Use the best imputer for further imputation
    final_imputed_data = best_imputer.transform(incomplete_data)

    # Prepare dictionary with final results
    final_results = {}

    # Calculate MSE, RMSE and R^2 for the best-scoring model
    final_results["mse"] = mean_squared_error(
        test_vals,  # True values
        final_imputed_data[missing_mask, 0])    # Imputed values
    final_results["rmse"] = np.sqrt(final_results["mse"])

    final_results["r2"] = r2_score(
        test_vals,  # True values
        final_imputed_data[missing_mask, 0],    # Imputed values
    )

    if print_progress:
        print(f"  Imputation MSE: {final_results["mse"]:.4f}")
        print(f"  Imputation RMSE: {final_results["rmse"]:.4f}")
        print(f"  Imputation R^2: {final_results["r2"]:.4f}")

    # Only keep non-fully-NaN columns in final overview
    notnull_cols = complete_data.columns[~complete_data.isnull().all()]

    # Convert NumPy array of results (back) to DataFrame format
    filled_imp_df = pd.DataFrame(data=final_imputed_data,
                                 index=complete_data.index,
                                 columns=notnull_cols)

    # Return best imp. object, result scores, and imp. df (for tests)
    return (best_imputer, final_results, filled_imp_df)


def run_best_imputer_on_dataset(df_imp: pd.DataFrame,
                                best_imputer: _BaseImputer) -> pd.DataFrame:
    """
    Run best-scoring imputer on the full dataset; return the result.
    
    Parameters
    ----------
    df_imp : pd.DataFrame
        The dataset to run the imputation for. Strongly advised to use the
        same DataFrame as was used as an input for aforementioned function 
        `fit_best_df_imputer_on_targetcol`.
    best_imputer : sklearn.impute._base._BaseImputer
        The best-scoring imputer to use for the imputation procedure.

    Returns
    -------
    df_imputed : pd.DataFrame
        Dataset with the same columns and index as the input DataFrame,
        but now with all values in each column imputed.
    
    Notes
    -----
    - Is set up to work with any class that inherits from `_BaseImputer`.

    """
    # Fit and run the best imputer on the *full* to-impute dataset
    df_imputed = pd.DataFrame(data=best_imputer.fit_transform(df_imp),
                              columns=df_imp.columns,
                              index=df_imp.index)
    return df_imputed
