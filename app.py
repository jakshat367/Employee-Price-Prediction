from flask import Flask, render_template, request

import pandas as pd
import joblib
import os


app = Flask(__name__)


# Load ML Model
model_path = "model/salary_model.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(
        "Model not found. Run python train_model.py first."
    )

model = joblib.load(model_path)


@app.route("/")
def home():

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        experience = float(request.form["experience"])

        education = request.form["education"]

        job_role = request.form["job_role"]

        location = request.form["location"]

        company_size = request.form["company_size"]


        # Prepare Input Data
        input_data = pd.DataFrame(
            [[
                experience,
                education,
                job_role,
                location,
                company_size
            ]],

            columns=[
                "experience",
                "education",
                "job_role",
                "location",
                "company_size"
            ]
        )


        # Predict Salary
        prediction = model.predict(input_data)[0]

        monthly_salary = prediction / 12


        return render_template(
            "result.html",

            salary=round(prediction, 2),

            monthly=round(monthly_salary, 2),

            experience=experience,

            education=education,

            job_role=job_role,

            location=location
        )

    except (ValueError, KeyError):

        return "Invalid input. Please check your details.", 400


if __name__ == "__main__":

    app.run(debug=True)