
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
import numpy as np # Import numpy for numerical operations

app = Flask(__name__)

# Define model, scaler, and columns filenames
model_filename = 'airbnb_rf_model.joblib'
scaler_filename = 'airbnb_scaler.joblib'
columns_filename = 'airbnb_training_columns.joblib' # New: Filename for training columns

# Load the trained model, scaler, and training columns
try:
    model = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
    TRAINING_COLUMNS = joblib.load(columns_filename) # New: Load training columns
    print(f"Model '{model_filename}', Scaler '{scaler_filename}', and Training Columns '{columns_filename}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: One or more files not found. Please ensure '{model_filename}', '{scaler_filename}', and '{columns_filename}' are in the same directory.")
    model = None # Set to None to handle errors later
    scaler = None
    TRAINING_COLUMNS = [] # Initialize as empty list to prevent NameError

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded. Check server logs.'}), 500

    try:
        data = request.get_json(force=True)

        # Convert input JSON data to DataFrame
        input_df = pd.DataFrame([data])

        # Preprocessing

        # 1. Feature Engineering
        if 'Neighbourhood ' in input_df.columns and 'Room Type' in input_df.columns:
            input_df['Neighbourhood_RoomType'] = input_df['Neighbourhood '] + '_' + input_df['Room Type']
        else:
            # This case should ideally not happen if the frontend sends all required inputs
            # For robustness, we can try to proceed without the interaction feature if inputs are truly missing,
            # but it's better to ensure the frontend sends all necessary data.
            # For now, let's raise an error to indicate missing input.
            return jsonify({'error': 'Missing Neighbourhood or Room Type for interaction feature.'}), 400

        # 2. One-hot encode
        # Ensure all expected categorical columns are present in input_df, even if with NaN, before one-hot encoding.
        # This is important if some categories might be missing in a single prediction request.
        categorical_features_for_ohe = ['Neighbourhood ', 'Property Type', 'Room Type', 'Zipcode', 'Neighbourhood_RoomType']

        for col in categorical_features_for_ohe:
            if col not in input_df.columns:
                input_df[col] = np.nan # Add missing categorical column with NaN to be handled by get_dummies
        
        input_df_encoded = pd.get_dummies(input_df, columns=categorical_features_for_ohe, drop_first=True)

        # Add missing columns (features that were in training data but not in this specific input)
        missing_cols = set(TRAINING_COLUMNS) - set(input_df_encoded.columns)
        for c in missing_cols:
            input_df_encoded[c] = 0

        # Drop any extra columns (features that were in this specific input but not in training data)
        extra_cols = set(input_df_encoded.columns) - set(TRAINING_COLUMNS)
        input_df_encoded = input_df_encoded.drop(columns=list(extra_cols))

        # Reorder columns to match the training set
        final_input = input_df_encoded[TRAINING_COLUMNS]

        # 3. Scale numerical features
        final_input_scaled = scaler.transform(final_input)

        # Make prediction
        prediction = model.predict(final_input_scaled)
        predicted_price = round(prediction[0], 2)

        # Calculate a plausible price range (e.g., +/- 10% of the predicted price)
        range_percentage = 0.10 # 10%
        lower_bound = round(predicted_price * (1 - range_percentage), 2)
        upper_bound = round(predicted_price * (1 + range_percentage), 2)

        return jsonify({
            'predicted_price': predicted_price,
            'price_range_lower': lower_bound,
            'price_range_upper': upper_bound
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
