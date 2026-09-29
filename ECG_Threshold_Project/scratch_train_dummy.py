import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def train_dummy():
    print("Training dummy model for backend testing...")
    
    # 72 features (12 leads * 6 features)
    X_dummy = np.random.rand(100, 72)
    y_dummy = np.random.randint(0, 2, 100)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dummy)
    
    model = LogisticRegression()
    model.fit(X_scaled, y_dummy)
    
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/ecg_model.pkl')
    joblib.dump(scaler, 'model/ecg_scaler.pkl')
    print("Dummy model and scaler saved successfully to model/ directory.")

if __name__ == "__main__":
    train_dummy()
