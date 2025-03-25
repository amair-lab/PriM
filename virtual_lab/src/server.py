# src/inference.py
from flask import Flask, request, jsonify
import torch
from model import GFactorPredictor
from data_processor import DataProcessor
import pandas as pd

app = Flask(__name__)

# Load model and processor
model = GFactorPredictor()
model.load_state_dict(torch.load('../models/best_model.pth'))
model.eval()
processor = DataProcessor()


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.get_json()

        # Convert input to DataFrame
        input_df = pd.DataFrame([data])

        # Process input data
        processed_input = processor.process_input(input_df)

        # Convert to tensor and get prediction
        input_tensor = torch.FloatTensor(processed_input)
        with torch.no_grad():
            prediction = model(input_tensor)

        # Return prediction
        return jsonify({
            'status': 'success',
            'predicted_g_factor': float(prediction[0][0])
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=21500)