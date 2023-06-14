from flask import Flask, render_template, request
from joblib import load
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model and label encoder
model = load('model/trained_model.joblib')
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(load('pkl/label_encoder.joblib'))

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    input_data = request.form.to_dict()

    # Preprocess the input data
    input_df = pd.DataFrame([input_data])
    input_encoded = input_df.apply(lambda x: label_encoder.transform(x.astype(str)))

    # Make the prediction
    prediction = model.predict(input_encoded)

    # Convert the prediction to human-readable label
    predicted_label = label_encoder.inverse_transform(prediction)

    # Render the prediction result
    return render_template('result.html', prediction=predicted_label)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
