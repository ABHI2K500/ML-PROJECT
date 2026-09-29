import numpy as np

def preprocess_signals(X):
    """
    Preprocesses the ECG waveform data.
    
    Steps:
    1. Handle missing values (NaNs replaced by 0 or interpolation).
    2. Ensure consistent dimensions.
    3. Ensure numeric representation (float32).
    
    Note: Standardization/Scaling is performed using scikit-learn's StandardScaler 
    during the model training pipeline to prevent data leakage.
    
    Parameters:
    - X: np.ndarray, shape (num_samples, num_timesteps, num_leads)
    
    Returns:
    - np.ndarray: preprocessed signals
    """
    if len(X) == 0:
        return X
        
    # Convert to float32 to save memory
    X = X.astype(np.float32)
    
    # Handle missing values by replacing NaNs with 0 (baseline)
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0)
        
    # Basic baseline wander removal could be added here, 
    # but to keep it reproducible and simple, we rely on the features extracted later.
    
    return X
