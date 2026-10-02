from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib


# ============================================
# 1. CREATE FLASK APP
# ============================================

app = Flask(__name__)


# ============================================
# 2. LOAD ML MODEL
# ============================================

model = joblib.load(
    "bubt_waiver_prediction_model.pkl"
)


# ============================================
# 3. HOME PAGE
# ============================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================
# 4. PREDICTION API
# ============================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ------------------------------------
        # Get JSON data from frontend
        # ------------------------------------

        data = request.get_json()

        semester = int(
            data["semester"]
        )

        ssc_gpa = float(
            data.get("ssc_gpa", 0)
        )

        hsc_gpa = float(
            data.get("hsc_gpa", 0)
        )

        sgpa = float(
            data.get("sgpa", 0)
        )

        cgpa = float(
            data.get("cgpa", 0)
        )

        tuition_fee = float(
            data["tuition_fee"]
        )


        # ------------------------------------
        # Calculate Academic Score
        # ------------------------------------

        if semester == 1:

            academic_score = (
                ssc_gpa + hsc_gpa
            ) / 2

        else:

            academic_score = (
                sgpa + cgpa
            ) / 2


        # ------------------------------------
        # Create DataFrame
        # ------------------------------------

        input_data = pd.DataFrame({

            "Semester": [semester],

            "SSC_GPA": [ssc_gpa],

            "HSC_GPA": [hsc_gpa],

            "SGPA": [sgpa],

            "CGPA": [cgpa],

            "Academic_Score": [
                academic_score
            ]

        })


        # ------------------------------------
        # ML MODEL PREDICTION
        # ------------------------------------

        predicted_waiver = model.predict(
            input_data
        )[0]


        # ------------------------------------
        # Keep prediction between 0-100
        # ------------------------------------

        predicted_waiver = max(
            0,
            min(
                100,
                predicted_waiver
            )
        )


        # ------------------------------------
        # Calculate waiver amount
        # ------------------------------------

        waiver_amount = (
            tuition_fee
            * predicted_waiver
            / 100
        )


        # ------------------------------------
        # Calculate remaining fee
        # ------------------------------------

        remaining_fee = (
            tuition_fee
            - waiver_amount
        )


        # ------------------------------------
        # Send result to frontend
        # ------------------------------------

        return jsonify({

            "status": "success",

            "semester": semester,

            "predicted_waiver_percent":
                round(
                    predicted_waiver,
                    2
                ),

            "tuition_fee":
                round(
                    tuition_fee,
                    2
                ),

            "estimated_waiver_amount":
                round(
                    waiver_amount,
                    2
                ),

            "estimated_remaining_fee":
                round(
                    remaining_fee,
                    2
                )

        })


    # ========================================
    # ERROR HANDLING
    # ========================================

    except Exception as e:

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 400


# ============================================
# 5. RUN FLASK SERVER
# ============================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
