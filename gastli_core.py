import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# --- Configuration ---
# Features must match the exact order used during training (from t.py/r.py)
INPUT_FEATURES = ['mass_MEarth', 'CMF', 'Zenv', 'Zwater_core', 'Tsurf_K', 'Psurf_bar']
OUTPUT_LABELS = ['radius_Rearth', 'entropy_SI', 'signed_log10_f_s_SI']

class GastliModel:
    def __init__(self, model_dir='models'):
        # Updated filenames to match t.py/r.py output
        self.model_path = os.path.join(model_dir, 'final_model.keras')
        self.x_scaler_path = os.path.join(model_dir, 'x_scaler.joblib')
        self.y_scaler_path = os.path.join(model_dir, 'y_scaler.joblib')
        self.model = None
        self.X_scaler = None
        self.Y_scaler = None
        
    def load(self):
        """Loads the trained ResNet model and data scalers."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Missing model file: {self.model_path}")
            
        # 1. Load Scalers
        self.X_scaler = joblib.load(self.x_scaler_path)
        self.Y_scaler = joblib.load(self.y_scaler_path)
        
        # 2. Load Model
        # compile=False prevents loading optimizer states/losses, 
        # which is safer and faster for inference.
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        return True

    def predict(self, inputs_dict):
        """
        Runs the prediction pipeline.
        Returns a dictionary with physical values.
        """
        # 1. Format Input
        input_df = pd.DataFrame([inputs_dict])
        
        # Ensure column order matches training exactly
        missing_cols = [c for c in INPUT_FEATURES if c not in input_df.columns]
        if missing_cols:
            raise ValueError(f"Input missing columns: {missing_cols}")
        input_df = input_df[INPUT_FEATURES] 
        
        # 2. Scale Input (RobustScaler)
        input_scaled = self.X_scaler.transform(input_df)
        
        # 3. Predict 
        # The ResNet outputs the 3 values directly (Radius, Entropy, signed_fs)
        raw_pred_scaled = self.model.predict(input_scaled, verbose=0)
        
        # 4. Inverse Scale (StandardScaler)
        # Result is shape (1, 3), we take the first row
        preds_original = self.Y_scaler.inverse_transform(raw_pred_scaled)[0]
        
        # 5. Extract raw values
        radius = preds_original[0]
        entropy = preds_original[1]
        fs_log = preds_original[2]
        
        # 6. Physical Transformation for f_s
        # The model predicts: sign(x) * log10(1 + |x|)
        # We invert it: sign(y) * (10^|y| - 1)
        fs_physical = np.sign(fs_log) * (10**np.abs(fs_log) - 1)
        
        return {
            'radius_Rearth': float(radius),
            'entropy_SI': float(entropy),
            'f_s_SI': float(fs_physical)
        }
