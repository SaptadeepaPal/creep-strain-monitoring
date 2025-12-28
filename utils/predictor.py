"""
predictor.py
============
Backend module for creep strain prediction
Stateless, reusable, Streamlit-compatible
"""

import joblib
import numpy as np
import pandas as pd
from typing import Tuple


class CreepPredictor:
    """
    Creep strain prediction engine using trained XGBoost model.
    
    This class handles:
    - Loading pre-trained model and scaler artifacts
    - Input preprocessing with proper feature naming
    - Strain prediction with inverse log transformation
    - Value clamping to valid range [0, 100]%
    """
    
    def __init__(self, model_path='xgb_model.pkl', scaler_path='scaler.pkl', 
                 config_path='config.pkl'):
        """
        Initialize predictor by loading trained artifacts.
        
        Args:
            model_path: Path to trained XGBoost model
            scaler_path: Path to fitted RobustScaler
            config_path: Path to configuration dictionary
        """
        self.model = self._load_artifact(model_path, "Model")
        self.scaler = self._load_artifact(scaler_path, "Scaler")
        self.config = self._load_artifact(config_path, "Config")
        
        # Extract configuration values
        self.warning_threshold = self.config.get('warning_threshold', 5.0)
        self.rmse = self.config.get('rmse', 0.5)
    
    def _load_artifact(self, path: str, artifact_name: str):
        """
        Load a pickled artifact with error handling.
        
        Args:
            path: File path to artifact
            artifact_name: Name of artifact (for error messages)
            
        Returns:
            Loaded artifact object
            
        Raises:
            FileNotFoundError: If artifact file is not found
            Exception: For other loading errors
        """
        try:
            artifact = joblib.load(path)
            return artifact
        except FileNotFoundError:
            raise FileNotFoundError(f"{artifact_name} file not found: {path}")
        except Exception as e:
            raise Exception(f"Error loading {artifact_name}: {str(e)}")
    
    def preprocess_input(self, duration: float, voltage: float) -> np.ndarray:
        """
        Preprocess raw input values using the fitted scaler.
        
        CRITICAL: Uses exact feature names from training to avoid sklearn errors:
                  ['Expose Duration (Hrs)', 'Output Voltage (mV)']
        
        Args:
            duration: Exposure duration in hours
            voltage: Output voltage in millivolts
            
        Returns:
            np.ndarray: Scaled feature array (shape: 1x2)
        """
        # Create DataFrame with exact column names from training
        input_df = pd.DataFrame({
            'Expose Duration (Hrs)': [duration],
            'Output Voltage (mV)': [voltage]
        })
        
        # Transform using pre-fitted scaler (never refit)
        scaled_input = self.scaler.transform(input_df)
        
        return scaled_input
    
    def predict(self, duration: float, voltage: float) -> float:
        """
        Predict creep strain percentage for given inputs.
        
        Process:
        1. Preprocess inputs (scale using RobustScaler)
        2. Make prediction in log-transformed space
        3. Inverse transform using expm1 (reverses log1p)
        4. Clamp to valid range [0%, 100%]
        
        Args:
            duration: Exposure duration in hours
            voltage: Output voltage in millivolts
            
        Returns:
            float: Predicted strain percentage (0-100)
        """
        # Step 1: Preprocess
        scaled_input = self.preprocess_input(duration, voltage)
        
        # Step 2: Predict in log-space
        strain_log = self.model.predict(scaled_input)[0]
        
        # Step 3: Inverse transform (expm1 reverses log1p)
        strain_percent = np.expm1(strain_log)
        
        # Step 4: Clamp to physically valid range
        strain_percent = np.clip(strain_percent, 0.0, 100.0)
        
        return strain_percent
    
    def get_config(self) -> dict:
        """
        Get configuration dictionary.
        
        Returns:
            dict: Configuration with warning_threshold and rmse
        """
        return {
            'warning_threshold': self.warning_threshold,
            'rmse': self.rmse
        }
    
    def validate_input(self, duration: float, voltage: float) -> Tuple[bool, str]:
        """
        Validate input values before prediction.
        
        Args:
            duration: Exposure duration in hours
            voltage: Output voltage in millivolts
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if duration < 0:
            return False, "Exposure Duration cannot be negative"
        
        if voltage < 0:
            return False, "Output Voltage cannot be negative"
        
        return True, ""