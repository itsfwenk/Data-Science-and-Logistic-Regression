import pandas as pd
import os

def load_csv(path: str) -> pd.DataFrame:
    """
    Takes a path as argument and returns it.

    Args:
        arg1 (str): The path of the data set to load.

    Returns:
        A pandas dataframe of the data set.
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: File '{path}' not found.")
        if not os.path.isfile(path):
            raise ValueError(f"Error: '{path}' is not a valid file.")
        if not path.lower().endswith(".csv"):
            raise AssertionError("file must be .csv.")

        df = pd.read_csv(path)
        return df

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except ValueError as e:
        print(f"Error: {e}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The CSV file '{path}' is empty.")
        return None
    except pd.errors.ParserError as e:
        print(
            f"Error: Could not parse '{path}': {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading '{path}': {e}")
        return None