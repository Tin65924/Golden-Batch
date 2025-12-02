from flask import jsonify, request
import joblib, pandas

model = joblib.load(r"backend\model.pkl")

def predict():
    """ Sent data should be a JSON
    {
    "Temperature": 0,       -> int
    "Pressure": 0.0        -> float/double
    }
    """

    data = request.json()

    if not data:
        return jsonify({"m": "No data is transferred."})
    
    X_new = pandas.DataFrame([data])

    predicted = model.predict(X_new)

    print(predicted)

    if predicted == 0:
        return "Failed"
    elif predicted == 1:
        return "Passed"
    else:
        return "Something wrong."
    
predict(jsonify({"Temperature": 5, "Pressure": 6}))