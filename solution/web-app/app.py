import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained RandomForestClassifier model
model = pickle.load(open("../heart-failure_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Mapping dictionary for categorical variables
    chest_pain_map = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}
    resting_ecg_map = {'Normal': 0, 'ST': 1, 'LVH': 2}
    st_slope_map = {'Up': 0, 'Flat': 1, 'Down': 2}

    # Extracting and converting the input features from the form
    int_features = []
    for key, value in request.form.items():
        if key == 'ChestPainType':
            int_features.append(chest_pain_map[value])
        elif key == 'RestingECG':
            int_features.append(resting_ecg_map[value])
        elif key == 'ST_Slope':
            int_features.append(st_slope_map[value])
        else:
            int_features.append(float(value))  # Convert input to float
    
    final_features = [np.array(int_features)]
    
    # Predicting using the loaded model
    prediction = model.predict(final_features)
    output = int(prediction[0])
    
    # Render the result in the HTML template
    return render_template(
        "index.html",
        prediction_text=f"{'Positive' if output == 1 else 'Negativ'}"
    )


if __name__ == "__main__":
    app.run(debug=True)
