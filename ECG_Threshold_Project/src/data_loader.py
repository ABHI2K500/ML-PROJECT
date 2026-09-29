import os
import pandas as pd
import numpy as np
import wfdb
import ast

def check_dataset_exists(data_dir='data/ptb-xl/'):
    """
    Checks if the PTB-XL dataset exists in the specified directory.
    """
    metadata_path = os.path.join(data_dir, 'ptbxl_database.csv')
    if not os.path.exists(metadata_path):
        return False
    return True

def load_ptbxl_metadata(data_dir='data/ptb-xl/', max_records=5000):
    """
    Locates and loads PTB-XL metadata using pandas.
    
    Parameters:
    - data_dir: str, path to the PTB-XL dataset directory
    - max_records: int, maximum number of records to load (for dev/testing)
    
    Returns:
    - pd.DataFrame: metadata dataframe
    """
    if not check_dataset_exists(data_dir):
        print(f"Error: Dataset not found in {data_dir}.")
        print("Please download the PTB-XL dataset from PhysioNet and place it in the 'data/ptb-xl/' directory.")
        return None

    metadata_path = os.path.join(data_dir, 'ptbxl_database.csv')
    try:
        df = pd.read_csv(metadata_path, index_col='ecg_id')
        # Parse SCP codes dictionary
        df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
        if max_records and max_records > 0:
            df = df.head(max_records)
        return df
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None

def load_ecg_data(df, sampling_rate=100, data_dir='data/ptb-xl/'):
    """
    Loads ECG signals using WFDB for the records in the metadata dataframe.
    
    Parameters:
    - df: pd.DataFrame, metadata dataframe
    - sampling_rate: int, sampling rate (100 or 500) to determine which files to load
    - data_dir: str, path to the PTB-XL dataset directory
    
    Returns:
    - np.ndarray: array of ECG signals
    - pd.DataFrame: dataframe of valid records (excluding corrupted/missing ones)
    """
    if df is None or df.empty:
        return np.array([]), df
    
    data = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        if sampling_rate == 100:
            file_name = row['filename_lr']
        else:
            file_name = row['filename_hr']
        
        file_path = os.path.join(data_dir, file_name)
        
        try:
            # wfdb.rdsamp expects the path without the extension
            signal, meta = wfdb.rdsamp(file_path)
            data.append(signal)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Warning: Failed to load record {idx} at {file_path}. Skipping. Error: {e}")
            
    df_valid = df.loc[valid_indices]
    return np.array(data), df_valid
