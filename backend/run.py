from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

model = joblib.load(r"backend\model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Sent data should be a JSON
    {
        "Temperature": 0,  -> int
        "Pressure": 0      -> int
    }
    """
    data = request.get_json()

    if not data:
        return jsonify({"m": "No data is transferred."}), 400
    
    # Convert to DataFrame
    X_new = pd.DataFrame([data])

    # Predict
    predicted = model.predict(X_new)
    print(predicted)

    # Compare first element
    if predicted[0] == 0:
        return jsonify({"prediction": "Fail"})
    elif predicted[0] == 1:
        return jsonify({"prediction": "Pass"})
    else:
        return jsonify({"prediction": "Something went wrong!"})

if __name__ == "__main__":
    app.run(debug=True)