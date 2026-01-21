from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
with open("model/breast_cancer_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        # Get input values from form
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        result = model.predict(final_features)[0]
        prediction = "Malignant (Cancerous)" if result == 1 else "Benign (Non-Cancerous)"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

