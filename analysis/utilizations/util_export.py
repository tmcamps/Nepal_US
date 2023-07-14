import pandas as pd

def create_df(dict):
    """Function to create dataframe from dictionary with parameters as index """
    df = pd.DataFrame.from_dict(dict)
    df.index.name = 'Parameter'

    return df

def create_multilevel(df):
    """Function to create dataframe with multilevel columns by splitting values in datatype (dtype) and value """
    df_new = df.copy()
    for column in df.columns:
        # Remove parentheses
        df_new[column] = df[column].apply(lambda x: pd.Series(str(x).replace('(', '').replace(')', '')))

        # Split columns in dtype and value columns
        df_new[[column+',dtype', column+',value']] = df_new[column].apply(lambda x: pd.Series(str(x).split(",")))

        # Drop original column
        df_new = df_new.drop(columns=column)

    # Create multilevel columns based on dtype and value
    df_new.columns = df_new.columns.str.split(',', expand=True)

    return df_new


def stack_dataframe(df):
    """Function to create dataframe with multilevel index by stacking dataframe based on the sections"""
    df_new = df.copy()

    # Stack the dataframe based on the first level (the sections)
    df_new = df_new.stack(0)

    # Swap indices
    df_new = df_new.swaplevel()

    # Set index names of the multi index dataframe
    df_new.index.names = ["Section", "Parameter"]

    # Drop all rows with NaN values
    df_new = df_new.dropna()

    return df_new


def add_date(df, date):
    """Function to add date to dataframe to enable saving to excel"""
    df_new = df.copy()

    # Create column name of date
    date_column = pd.to_datetime(date, format='%Y%m%d')

    # Set columns by using multiindex from arrays
    df_new.columns = pd.MultiIndex.from_arrays([[date_column, date_column], df_new.columns])

    return df_new
