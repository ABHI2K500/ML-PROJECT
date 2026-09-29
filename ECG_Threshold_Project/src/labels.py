import pandas as pd
import os
import numpy as np

def aggregate_diagnostic(y_dic, agg_df):
    """
    Aggregates specific SCP codes into diagnostic superclasses.
    """
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index and agg_df.loc[key].diagnostic_class is not np.nan:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

def create_binary_labels(df, data_dir='data/ptb-xl/'):
    """
    Creates binary classification targets based on PTB-XL metadata.
    
    Target definition:
    0 = Normal ECG pattern
    1 = Abnormal ECG pattern
    
    This function uses scp_statements.csv to map scp_codes to diagnostic superclasses.
    A record is assigned to class 0 if its only diagnostic superclass is 'NORM'.
    It is assigned to class 1 if it has any other diagnostic superclass (e.g., MI, STTC, CD, HYP).
    Records with ambiguous or empty diagnostic superclasses are removed.
    
    Parameters:
    - df: pd.DataFrame, metadata dataframe containing 'scp_codes'
    - data_dir: str, path to the PTB-XL dataset directory
    
    Returns:
    - pd.DataFrame: dataframe with an additional 'label' column (0 or 1), 
                    ambiguous records dropped.
    """
    scp_path = os.path.join(data_dir, 'scp_statements.csv')
    if not os.path.exists(scp_path):
        print(f"Error: {scp_path} not found. Cannot create labels.")
        return df
        
    agg_df = pd.read_csv(scp_path, index_col=0)
    
    # Apply aggregation to get diagnostic superclasses
    df['diagnostic_superclass'] = df.scp_codes.apply(lambda x: aggregate_diagnostic(x, agg_df))
    
    def assign_label(superclasses):
        if len(superclasses) == 0:
            return -1 # Ambiguous/No diagnostic class
        if 'NORM' in superclasses and len(superclasses) == 1:
            return 0 # Normal
        if 'NORM' not in superclasses:
            return 1 # Abnormal
        # If it has both NORM and others, we treat it as Abnormal to be safe,
        # but one can also drop these. Let's treat them as Abnormal (1).
        return 1
        
    df['label'] = df['diagnostic_superclass'].apply(assign_label)
    
    # Drop ambiguous records
    initial_len = len(df)
    df = df[df['label'] != -1].copy()
    dropped = initial_len - len(df)
    
    # Print class distribution
    num_normal = (df['label'] == 0).sum()
    num_abnormal = (df['label'] == 1).sum()
    total = len(df)
    
    print(f"Label Creation Summary:")
    print(f"Dropped {dropped} ambiguous/unlabeled records.")
    print(f"Number of normal samples: {num_normal} ({(num_normal/total)*100:.2f}%)")
    print(f"Number of abnormal samples: {num_abnormal} ({(num_abnormal/total)*100:.2f}%)")
    
    return df
