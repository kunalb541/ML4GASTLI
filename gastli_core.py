import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# --- Configuration ---
# These must match the scalar training columns exactly
TRAINING_FEATURES = ['log_mass_MEarth', 'CMF', 'Zenv', 'Zwater_core', 'Tsurf_K', 'log_Psurf_bar']

class GastliModel:
    def __init__(self, model_dir='models'):
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
        # compile=False prevents loading optimizer states, safer for inference
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        return True

    def predict(self, inputs_dict):
        """
        Runs the prediction pipeline.
        
        Expected keys in inputs_dict:
        'mass_MEarth', 'CMF', 'Zenv', 'Zwater_core', 'Tsurf_K', 'Psurf_bar'
        """
        # --- 1. Preprocessing (Must match train.py) ---
        # We must create a dict that matches the TRAINING_FEATURES list
        processed_input = {
            # Log transform Mass and Pressure (adding epsilon not strictly necessary for inference if inputs > 0)
            'log_mass_MEarth': np.log10(inputs_dict['mass_MEarth']),
            'CMF': inputs_dict['CMF'],
            'Zenv': inputs_dict['Zenv'],
            'Zwater_core': inputs_dict['Zwater_core'],
            'Tsurf_K': inputs_dict['Tsurf_K'],
            'log_Psurf_bar': np.log10(inputs_dict['Psurf_bar'])
        }

        # Convert to DataFrame with correct column order
        input_df = pd.DataFrame([processed_input])
        input_df = input_df[TRAINING_FEATURES]
        
        # --- 2. Scale Input ---
        # Uses RobustScaler loaded from training
        input_scaled = self.X_scaler.transform(input_df)
        
        # --- 3. Predict ---
        # The model returns a list of 3 tensors: [radius_out, entropy_out, fs_out]
        # We predict with verbose=0 to silence logs
        raw_pred_list = self.model.predict(input_scaled, verbose=0)
        
        # Stack them to shape (1, 3) for the scaler
        # raw_pred_list is [array([[r]]), array([[s]]), array([[f]])]
        pred_stacked = np.column_stack(raw_pred_list)

        # --- 4. Inverse Scale Outputs ---
        # Uses QuantileTransformer loaded from training
        preds_original_scale = self.Y_scaler.inverse_transform(pred_stacked)[0]
        
        # Extract values (Note: Radius is still in log10 space here)
        log_radius = preds_original_scale[0]
        entropy = preds_original_scale[1]
        fs_log_signed = preds_original_scale[2]
        
        # --- 5. Physical Transformations ---
        
        # A. Radius: Model predicted log10(R), convert to R
        radius_physical = 10**log_radius
        
        # B. f_s: Model predicted signed_log10(fs), convert to physical
        # Inverse of: sign(x) * log10(1 + |x|)
        fs_physical = np.sign(fs_log_signed) * (10**np.abs(fs_log_signed) - 1)
        
        return {
            'radius_Rearth': float(radius_physical),
            'entropy_SI': float(entropy),
            'f_s_SI': float(fs_physical)
        }
