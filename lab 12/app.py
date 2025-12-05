from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained artifacts
model = pickle.load(open("svr_model.pkl", "rb"))
scaler_X = pickle.load(open("scaler_X.pkl", "rb"))
scaler_y = pickle.load(open("scaler_y.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

# Feature order must match training
feature_order = ['brand','model','fuel_type','engine','transmission',
                 'ext_col','int_col','accident','clean_title','milage']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    data = {col: request.form.get(col, "") for col in feature_order}
    
    # Encode categorical columns
    X_row = []
    for col in feature_order:
        val = data[col]
        if col in label_encoders:
            le = label_encoders[col]
            try:
                enc = le.transform([str(val)])[0]
            except:
                enc = 0  # unseen label fallback
            X_row.append(enc)
        else:
            try:
                X_row.append(float(val))
            except:
                X_row.append(0.0)
    
    X_arr = np.array(X_row).reshape(1, -1)
    X_scaled = scaler_X.transform(X_arr)
    pred_scaled = model.predict(X_scaled)
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1,1)).ravel()[0]

    return render_template("result.html", price=round(pred, 2))

if __name__ == "__main__":
    app.run(debug=True)
