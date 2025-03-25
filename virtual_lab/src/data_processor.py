# src/data_processor.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import os


class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [
            'angle',
            'curl',
            'fiber_radius',
            'height',
            'helix_radius',
            'n_turns',
            'pitch',
            'total_fiber_length',
            'total_length'
        ]

    def load_and_process_data(self, csv_path, test_size=0.2, random_state=42):
        # Load data
        df = pd.read_csv(csv_path)

        # Split features and target
        X = df[self.feature_columns]
        y = df['g_factor']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Save scaler
        if not os.path.exists('../models'):
            os.makedirs('../models')
        with open('../models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        return X_train_scaled, X_test_scaled, y_train.values, y_test.values

    def process_input(self, input_data):
        """Process input data for inference"""
        # Load saved scaler
        with open('../models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # Convert input to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])

        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in input_data.columns:
                raise ValueError(f"Missing required feature: {col}")

        # Scale the input data
        scaled_data = scaler.transform(input_data[self.feature_columns])
        return scaled_data