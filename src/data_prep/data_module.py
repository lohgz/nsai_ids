import os
import re
import pickle
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

# paths
base_path = os.getcwd()
output_path = os.path.join(base_path, "data", "output")
scaler_ = "scaler.pkl"

def load_split_data(split_tag: str, base_path: str = base_path):
    """
    Loads train/test CSVs for a given split tag.

    Parameters:
    ----------
        split_tag: e.g. '_split_2'
        base_path: root of project (default: cwd)

    Returns:
    -------
        df_train, df_test: pandas DataFrames
    """
    train_path = os.path.join(base_path, "data", "output", f"train{split_tag}.csv")
    test_path  = os.path.join(base_path, "data", "output", f"test{split_tag}.csv")
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)
    return df_train, df_test


def prepare_tensors_and_datasets(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame
):
    """
    From train/test DataFrames, extract numeric features, scale them,
    and return PyTorch TensorDatasets plus metadata copies.


    Returns:
        train_dataset: TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset:  TensorDataset(X_test_tensor,  y_test_tensor)
        df_train_meta: copy of input df_train
        df_test_meta:  copy of input df_test
    """
    # keep metadata copies
    df_train_meta = df_train.copy().reset_index(drop=True)
    df_test_meta  = df_test.copy().reset_index(drop=True)

    # separate features and labels
    X_train = df_train_meta.drop('attack', axis=1)
    y_train = df_train_meta['attack']
    X_test  = df_test_meta.drop('attack', axis=1)
    y_test  = df_test_meta['attack']

    # select numeric columns only
    num_cols_train = X_train.select_dtypes(include=['number']).columns
    num_cols_test  = X_test.select_dtypes(include=['number']).columns
    X_train_num = X_train[num_cols_train]
    X_test_num  = X_test[num_cols_test]

    # scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_scaled  = scaler.transform(X_test_num)

    # build tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor  = torch.tensor(X_test_scaled,  dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test.values,  dtype=torch.long)

    # create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset  = TensorDataset(X_test_tensor,  y_test_tensor)

    return train_dataset, test_dataset, df_train_meta, df_test_meta
