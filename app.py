from flask import Flask, request, render_template
import numpy as np
import pickle

# Load trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__, template_folder='docs', static_folder='static')

@app.route('/')
def index():
    return render_template("index.html", show_form=False)

@app.route("/show_form")
def show_form():
    return render_template("index.html", show_form=True)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_features = [float(request.form[f'feature{i}']) for i in range(9)]
        np_features = np.asarray(input_features).reshape(1, -1)

        # Apply scaling before prediction
        np_features_scaled = scaler.transform(np_features)

        prediction = model.predict(np_features_scaled)
        output = "Cancerous (Malignant)" if prediction[0] == 1 else "Not Cancerous (Benign)"
        return render_template('index.html', message=output, show_form=True)
    except Exception as e:
        return render_template('index.html', message=f"⚠️ Error: {str(e)}", show_form=True)

if __name__ == '__main__':
    app.run(debug=True)