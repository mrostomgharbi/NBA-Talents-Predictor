from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import pandas as pd
from tensorflow.keras.models import model_from_json
import keras
import json
from sklearn.externals import joblib


MODELS_BASE_PATH="models"

# Get headers for payload
headers = ['GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3P Made', '3PA','3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV']


# Model reconstruction from JSON file
with open(MODELS_BASE_PATH + "/model.json", 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights(MODELS_BASE_PATH + "/model.h5")
scaler = joblib.load(MODELS_BASE_PATH + "/scaler.save")



app = Flask(__name__)
CORS(app)

def compute(values):
    input_variables = pd.DataFrame([values],
                                columns=headers,
                                dtype=float,
                                index=['input'])

    input_variables = scaler.transform(input_variables)
    # Get the model's prediction
    prediction_proba = model.predict_proba(input_variables)
    prediction = prediction_proba

    return float(prediction)

@app.route("/", methods=['GET'])
def hello():
    return "Welcome to NBA talents predictor tool"

@app.route("/api/nba-ml/player", methods=['POST'])
def predict():
    payload = request.json['data']
    values = [float(i) for i in payload]
    return jsonify({'prediction':compute(values)})

@app.route("/nba-ml/player", methods=['POST','GET'])
def predict_interface():
    if request.method == 'POST':
        values = []
        result = request.form
        for v in result.values():
            values.append(v)
        res = compute(values)

        return render_template('index.html',result='The player will succeed with probability {}%'.format( 100 * round(res , 2) ) )
    return render_template('index.html')



# running REST interface, port=5000 for direct test
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
