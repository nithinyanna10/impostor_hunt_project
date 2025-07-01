# Functions to load and parse train/test text files
import os
import pandas as pd

def load_train_data(data_dir="data", verbose=False):
    """
    Load training data and ground truth labels.

    Args:
        data_dir (str): Base path to data directory.
        verbose (bool): If True, print sample outputs.

    Returns:
        pd.DataFrame: DataFrame with columns: ['id', 'text_1', 'text_2', 'real']
    """
    train_csv = os.path.join(data_dir, "train.csv")
    train_dir = os.path.join(data_dir, "train")

    df = pd.read_csv(train_csv)
    samples = []

    for _, row in df.iterrows():
        article_id = row["id"]
        real_id = row["real_text_id"]

        folder_name = f"article_{str(article_id).zfill(4)}"
        path = os.path.join(train_dir, folder_name)

        try:
            with open(os.path.join(path, "file_1.txt"), encoding='utf-8') as f1, \
                 open(os.path.join(path, "file_2.txt"), encoding='utf-8') as f2:
                text_1 = f1.read()
                text_2 = f2.read()

            samples.append({
                "id": article_id,
                "text_1": text_1,
                "text_2": text_2,
                "real": real_id
            })
        except FileNotFoundError:
            print(f"⚠️ Missing files in: {folder_name}")

    df_data = pd.DataFrame(samples)
    if verbose:
        print(df_data.head())

    return df_data

import os
import glob

def load_test_data(test_dir="../data/test", verbose=False):
    """
    Loads the test data and returns a DataFrame with:
    ['id', 'text_1', 'text_2'] where 'real' is a dummy label
    """
    article_paths = sorted(glob.glob(os.path.join(test_dir, "../data/test/*")))
    samples = []

    for path in article_paths:
        article_id = int(os.path.basename(path).split("_")[1])
        try:
            with open(os.path.join(path, "file_1.txt"), encoding='utf-8') as f1, \
                 open(os.path.join(path, "file_2.txt"), encoding='utf-8') as f2:
                text_1 = f1.read()
                text_2 = f2.read()
            samples.append({
                "id": article_id,
                "text_1": text_1,
                "text_2": text_2,
            })
        except FileNotFoundError:
            print(f"⚠️ Missing files in: {path}")

    df = pd.DataFrame(samples)
    if verbose:
        print(df.head())

    return df
