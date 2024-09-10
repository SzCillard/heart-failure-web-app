import pickle
import numpy as np
from flask import Flask, request, render_template
from PIL import Image
from keras.models import load_model
from keras.layers import TFSMLayer
import tensorflow as tf

app = Flask(__name__)

heart_model = pickle.load(open("../heart-failure_model.pkl", "rb"))
lungs_model = load_model('../lungs_class_model.keras')

patients_data = [
    {"name": "John Doe", "sex": "M", "place_of_birth": "Seattle", "date_of_birth": "1999-01-01"},
    {"name": "Jane Smith", "sex": "F", "place_of_birth": "Washington", "date_of_birth": "1998-01-28"},
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/patients")
def patients():
    return render_template("patients.html", patients=patients_data)

@app.route("/search_patient", methods=["GET"])
def search_patient():
    query = request.args.get('query', '').lower()
    filtered_patients = [p for p in patients_data if query in p['name'].lower()]
    return render_template("patients.html", patients=filtered_patients)

@app.route("/register_patient", methods=["POST"])
def register_patient():
    first_name = request.form.get("first_name")
    last_name = request.form.get("last_name")
    sex = request.form.get("sex")
    place_of_birth = request.form.get("place_of_birth")
    date_of_birth = request.form.get("date_of_birth")
    
    full_name = f"{first_name} {last_name}"
    
    new_patient = {
        "name": full_name,
        "sex": "M" if sex == "0" else "F",
        "place_of_birth": place_of_birth,
        "date_of_birth": date_of_birth,
    }
    patients_data.append(new_patient)
    
    return render_template("patients.html", patients=patients_data)

@app.route("/heart")
def heart():
    return render_template("heart.html")

@app.route("/predict_heart", methods=["POST"])
def predict_heart():
    chest_pain_map = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}
    resting_ecg_map = {'Normal': 0, 'ST': 1, 'LVH': 2}
    st_slope_map = {'Up': 0, 'Flat': 1, 'Down': 2}

    int_features = []
    for key, value in request.form.items():
        if key == 'ChestPainType':
            int_features.append(chest_pain_map[value])
        elif key == 'RestingECG':
            int_features.append(resting_ecg_map[value])
        elif key == 'ST_Slope':
            int_features.append(st_slope_map[value])
        else:
            int_features.append(float(value))
    
    final_features = [np.array(int_features)]
    prediction = heart_model.predict(final_features)
    output = int(prediction[0])
    
    return render_template(
        "heart.html",
        prediction_text=f"{'Positive' if output == 1 else 'Negative'}"
    )


@app.route("/covid")
def covid():
    return render_template("covid.html")


@app.route("/predict_lungs", methods=["POST"])
def predict_lungs():
    file = request.files['image_file']
    if file.filename == '':
        return render_template("covid.html", prediction_text="No selected file.")
    
    img = Image.open(file)
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = lungs_model.predict(img_array)
    result = np.argmax(prediction, axis=1)[0]
    if result==0:
        result="Covid"
    elif result==1:
        result="Normal"
    else:
        result="Viral Pneumonia"

    return render_template("covid.html", prediction_text=f"{result}")
    

if __name__ == "__main__":
    app.run(debug=True)
