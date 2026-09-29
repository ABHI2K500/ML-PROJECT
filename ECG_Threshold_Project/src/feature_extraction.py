import numpy as np

def extract_features(X):
    """
    Extracts meaningful statistical features from ECG signals.
    
    For each of the 12 leads, computes:
    - Mean
    - Standard deviation
    - Minimum
    - Maximum
    - RMS (Root Mean Square)
    - Signal range (Max - Min)
    
    Parameters:
    - X: np.ndarray, shape (num_samples, num_timesteps, num_leads)
    
    Returns:
    - np.ndarray: Feature matrix of shape (num_samples, num_leads * 6)
    """
    if len(X) == 0:
        return np.array([])
        
    num_samples, num_timesteps, num_leads = X.shape
    num_features_per_lead = 6
    
    features = np.zeros((num_samples, num_leads * num_features_per_lead))
    
    for i in range(num_samples):
        sample = X[i] # Shape: (num_timesteps, num_leads)
        
        # Calculate features across time (axis=0) for each lead
        mean_vals = np.mean(sample, axis=0)
        std_vals = np.std(sample, axis=0)
        min_vals = np.min(sample, axis=0)
        max_vals = np.max(sample, axis=0)
        rms_vals = np.sqrt(np.mean(sample**2, axis=0))
        range_vals = max_vals - min_vals
        
        # Concatenate features for this sample
        sample_features = np.concatenate([
            mean_vals,
            std_vals,
            min_vals,
            max_vals,
            rms_vals,
            range_vals
        ])
        
        features[i] = sample_features
        
    return features
