import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer 

def replace_special_missing(df: pd.DataFrame, null_dict: dict=None) -> pd.DataFrame:
    """
    Replaces special placeholder values with NaN and creates indicator features to flag where
    values were originally missing.

    Parameters:
    ------------
    df (pd.DataFrame): Dataframe containing the columns to process.
    null_dict (dict): Dictionary mapping column names to their special missing values

    Returns:
    --------
    pd.DataFrame: A copy of the dataframe with special values replaced by NaN and additional features (col_missing)

    """
    # Replace missing values with nan using cols and vals specified in special_vals dict
    df = df.copy()

    for col, val in null_dict.items():
        df[f"{col}_missing"] = (df[col] == val).astype(int)
        df[col] = df[col].replace(val, np.nan)

    return df

def impute_numeric(df: pd.DataFrame, num_cols: list, imp_vals: pd.Series) -> pd.DataFrame:
    """ 
    Fills missing values with specified imputation values in numerical columns.

    Parameters:
    ----------
    df (pd.DataFrame): Dataframe containing numerical columns to impute.
    num_cols (list): List of column names to impute on.
    imp_vals (pd.Series): Series of values to use for imputation. 

    Returns:
    --------
    df (pd.DataFrame): A copy of the dataframe with missing values in numerical columns filled.

    """
    df = df.copy()
    df[num_cols] = df[num_cols].fillna(imp_vals)
    return df
    
def impute_categorical(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """  
    Fills missing values in categorical columns with 'unknown'

    Parameters:
    ------------
    df (pd.DataFrame): Dataframe containing categorical columns to impute.
    cat_cols (list): List of categorical column names to impute.

    Returns:
    ---------
    pd.DataFrame: A copy of the dataframe with missing values in categorical columns replaced with 'unknown'
    
    """
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].fillna("unknown")
    return df 

def fit_winsor(df: pd.DataFrame, num_cols: list, lower: float = 0.05, upper:float = 0.95) -> dict:
    """  
    Computes the lower and upper percentile cutoffs for numeric columns. 
    These cutoffs are later used for winsorization to cap extreme values.

    Parameters:
    -----------
    df (pd.DataFrame): Dataframe containing numeric columns.
    num_cols (list): List of numeric column names to compute cutoffs for.
    lower (float): Lower percentile (0.01 for 1st percentile).
    upper (float): Upper percentile (0.99 for 99th percentile).

    Returns:
    --------
    cutoffs (dict): Dictionary with column names as keys and (lower_cutoff, upper_cutoff) tuples as values.

    """
    cutoffs = {}
    for col in num_cols:
        cutoffs[col] = (df[col].quantile(lower), df[col].quantile(upper))
    return cutoffs

def apply_winsor(df: pd.DataFrame, cutoffs: dict) -> pd.DataFrame:
    """   
    Applies winsorization to numeric columns based on previously computed cutoffs.
    Values below the lower bound are capped at the lower percentile.
    Values above the upper bound are capped at the upper percentile.

    Parameters:
    -----------
    df (pd.DataFrame): Dataframe containing numeric columns to winsorize.
    cutoffs (dict): Dictionary with column names as keys and (lower_cutoff, upper_cutoff) tuples as values, generated from fit_winsor.

    Returns:
    --------
    df (pd.DataFrame): Dataframe with winsorized numeric columns.

    """
    df = df.copy()
    for col, (lower, upper) in cutoffs.items():
        df[col] = df[col].clip(lower, upper)
    return df
    
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """   
    Applies feature engineering to create new numeric features for credit risk modeling.
    Assumes that numeric imputation has already been applied and no missing values exist.
        - log_income: Log transformed income.
        - loan_to_income: Ratio of loan amount to income to capture borrower leverage.
        - util_trades: Interaction term between utilization rate and number of open trades.

    Parameters:
    -----------
    df (pd.DataFrame): Input dataframe with required columns: income, loan_amount, utilization_rate, num_open_trades.

    Returns:
    --------
    df (pd.DataFrame): A copy of the input dataframe with new engineered features.

    """
    df = df.copy()

    # Income transformation
    df["log_income"] = np.log1p(df["income"])

    # Interaction terms
    df["loan_to_income"] = df["loan_amount"] / (df["income"] + 1)
    df["util_trades"] = df["utilization_rate"] * df["num_open_trades"]
    #df["income_emp_length"] = df["income"] * df["employment_length"]

    return df

def build_pipeline(num_features, cat_features, ord_features, model = None):
    """   
    Builds a scikit-learn pipeline for preprocessing and modeling.

    The pipeline performs the following transformations:
    1. Numeric features: Passed through without change
    2. Categorical features: OneHot encoded with unknown categories ignored and the first category dropped to avoid perfect multicollinearity
    3. Ordinal features: Encoded with a specified order [36, 60]
    4. Model: Optional estimator at the end of the pipeline

    Parameters:
    -----------
    num_features (List): List of numeric columns to include
    cat_features (List): List of categorical columns to onehotencode
    ord_features (List): List of ordinal columns to encode
    model (sklean estimator): Machine learning model 

    Returns:
    --------
    (sklearn.pipeline object): A pipeline that preprocesses numeric, categorical, and ordinal features with option to fit a model.

    """
    preprocessor = ColumnTransformer(
        transformers = [
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown = "ignore", sparse_output = False, drop = "first"), cat_features),
            ("ord", OrdinalEncoder(categories = [[36, 60]]), ord_features)
        ],
        remainder = "drop"
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ]).set_output(transform = "pandas")