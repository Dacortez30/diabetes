from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model_diabetes.pkl")

@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    usia = float(data["usia"])
    berat = float(data["berat_badan"])
    olahraga = float(data["olahraga"])
    gula = float(data["gula_harian"])

    try:

        input_data = pd.DataFrame(
            [[usia, berat, olahraga, gula]]
        )

        hasil = model.predict(input_data)

        if hasil[0] == 1:

            return jsonify({
                "risiko":"tinggi",
                "keyakinan":92
            })

        else:

            return jsonify({
                "risiko":"rendah",
                "keyakinan":89
            })

    except Exception as e:

        return jsonify({
            "error":str(e)
        })


if __name__=="__main__":
    app.run(debug=True)