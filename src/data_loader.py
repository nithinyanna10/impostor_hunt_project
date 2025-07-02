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
import pandas as pd

def load_test_data(test_dir="../data/test", verbose=False):
    """
    Load test data from article folders that contain file_1.txt and file_2.txt.

    Returns:
        pd.DataFrame with columns ['id', 'text_1', 'text_2']
    """
    samples = []

    for folder in sorted(os.listdir(test_dir)):
        folder_path = os.path.join(test_dir, folder)
        if os.path.isdir(folder_path):
            file1_path = os.path.join(folder_path, "file_1.txt")
            file2_path = os.path.join(folder_path, "file_2.txt")

            if os.path.exists(file1_path) and os.path.exists(file2_path):
                try:
                    with open(file1_path, encoding="utf-8") as f1, open(file2_path, encoding="utf-8") as f2:
                        text_1 = f1.read()
                        text_2 = f2.read()
                    samples.append({
                        "id": folder,
                        "text_1": text_1,
                        "text_2": text_2
                    })
                    if verbose:
                        print(f"✅ Loaded {folder}")
                except Exception as e:
                    print(f"⚠️ Error reading {folder}: {e}")
            else:
                print(f"⚠️ Missing one or both files in {folder}")

    df = pd.DataFrame(samples)
    if verbose:
        print(f"\n📊 Final test_df shape: {df.shape}")
        print(df.head())

    return df
