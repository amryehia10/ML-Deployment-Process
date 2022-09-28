import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    pred = model.predict(features)

    return render_template("index.html", prediction_text="The flower species is {}".format(pred))

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8000)
