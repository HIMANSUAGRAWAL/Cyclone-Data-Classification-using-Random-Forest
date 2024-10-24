# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
from project import classify_cyclone, cyclone_descriptions  # Import the classify function and descriptions

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    max_wind = float(data['max_wind'])
    min_pressure = float(data['min_pressure'])
    latitude = float(data['latitude'])
    year = int(data['year'])
    longitude = float(data['longitude'])

    # Call the classify function
    predicted_cyclone = classify_cyclone(max_wind, min_pressure, latitude, year, longitude)
    description = cyclone_descriptions.get(predicted_cyclone, "No description available for this cyclone type.")

    return jsonify({
        'prediction': predicted_cyclone,
        'description': description
    })

if __name__ == '__main__':
    app.run(debug=True)
