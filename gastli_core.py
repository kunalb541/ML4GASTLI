import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# --- Configuration ---
# Features must match the exact order used during training
INPUT_FEATURES = ['mass_MEarth', 'CMF', 'Zenv', 'Zwater_core', 'Tsurf_K', 'Psurf_bar']
OUTPUT_LABELS = ['radius_Rearth', 'entropy_SI', 'signed_log10_f_s_SI']

# --- Dummy Loss Functions ---
# Keras needs these to load the model architecture, 
# even though we are only using it for prediction.
def heteroskedastic_loss(y_true, y_pred): return 0.0
def mean_prediction_loss(y_true, y_pred): return 0.0
def variance_prediction_loss(y_true, y_pred): return 0.0

class GastliModel:
    def __init__(self, model_dir='models'):
        self.model_path = os.path.join(model_dir, 'final_heteroskedastic_model.h5')
        self.x_scaler_path = os.path.join(model_dir, 'X_scaler.joblib')
        self.y_scaler_path = os.path.join(model_dir, 'Y_scaler.joblib')
        self.model = None
        self.X_scaler = None
        self.Y_scaler = None
        
    def load(self):
        """Loads the trained model and data scalers."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Missing model file: {self.model_path}")
            
        # Load Scalers
        self.X_scaler = joblib.load(self.x_scaler_path)
        self.Y_scaler = joblib.load(self.y_scaler_path)
        
        # Load Model (with custom loss dictionary)
        self.model = tf.keras.models.load_model(
            self.model_path, 
            custom_objects={
                'heteroskedastic_loss': heteroskedastic_loss,
                'mean_prediction_loss': mean_prediction_loss,
                'variance_prediction_loss': variance_prediction_loss
            },compile=False 
        )
        return True

    def predict(self, inputs_dict):
        """
        Runs the prediction pipeline.
        Returns a dictionary with physical values.
        """
        # 1. Format Input
        input_df = pd.DataFrame([inputs_dict])
        input_df = input_df[INPUT_FEATURES] # Ensure column order
        
        # 2. Scale Input (RobustScaler)
        input_scaled = self.X_scaler.transform(input_df)
        
        # 3. Predict (Returns [Means..., LogVars...])
        raw_pred = self.model.predict(input_scaled, verbose=0)
        
        # 4. Extract only Means (First 3 columns)
        # We ignore columns 3-5 which are variance/uncertainty
        means_scaled = raw_pred[:, :3]
        
        # 5. Inverse Scale (StandardScaler)
        means_original = self.Y_scaler.inverse_transform(means_scaled)[0]
        
        # 6. Extract raw values
        radius = means_original[0]
        entropy = means_original[1]
        fs_log = means_original[2]
        
        # 7. Physical Transformation for f_s
        # The model predicts: sign(x) * log10(1 + |x|)
        # We invert it: sign(y) * (10^|y| - 1)
        fs_physical = np.sign(fs_log) * (10**np.abs(fs_log) - 1)
        
        return {
            'radius_Rearth': radius,
            'entropy_SI': entropy,
            'f_s_SI': fs_physical
        }
