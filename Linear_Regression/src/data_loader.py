import pandas as pd
import os


def load_datasets(data_dir):
    """
    Loads train_data.csv, val_data.csv, and test_data.csv files from the specified directory.
    Returns three DataFrames: train_df, val_df, test_df
    """
    train_path = os.path.join(data_dir, "train_data.csv")
    val_path = os.path.join(data_dir, "val_data.csv")
    test_path = os.path.join(data_dir, "test_data.csv")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df

