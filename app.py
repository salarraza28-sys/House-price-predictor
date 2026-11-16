import os
import sys
import pickle
import logging
from flask import Flask, render_template, request, jsonify

from source_code import (
    HousePricePipeline,
    train_pipeline
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

CITIES_LIST = [
    'Algona', 'Auburn', 'Beaux Arts Village', 'Bellevue', 'Black Diamond',
    'Bothell', 'Burien', 'Carnation', 'Clyde Hill', 'Covington',
    'Des Moines', 'Duvall', 'Enumclaw', 'Fall City', 'Federal Way',
    'Inglewood-Finn Hill', 'Issaquah', 'Kenmore', 'Kent', 'Kirkland',
    'Lake Forest Park', 'Maple Valley', 'Medina', 'Mercer Island',
    'Milton', 'Newcastle', 'Normandy Park', 'North Bend', 'Pacific',
    'Preston', 'Ravensdale', 'Redmond', 'Renton', 'Sammamish',
    'SeaTac', 'Seattle', 'Shoreline', 'Skykomish', 'Snoqualmie',
    'Snoqualmie Pass', 'Tukwila', 'Vashon', 'Woodinville', 'Yarrow Point'
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
pipeline = None


def load_or_train_pipeline():
    global pipeline
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                pipeline = pickle.load(f)
        else:
            data_file = os.path.join(os.path.dirname(__file__), "data.csv")
            pipeline = train_pipeline(data_path=data_file, output_pickle=MODEL_PATH)
    except Exception as e:
        logger.error(f"Error loading pipeline: {e}")
        pipeline = None


load_or_train_pipeline()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    error_text = None
    breakdown = {}
    form_data = {}

    if request.method == "POST":
        try:
            if pipeline is None:
                raise RuntimeError("Prediction model is not loaded.")

            form_data = {
                "city": request.form.get("city", "Seattle").strip(),
                "bedrooms": request.form.get("bedrooms", "3").strip(),
                "bathrooms": request.form.get("bathrooms", "2.0").strip(),
                "sqft_living": request.form.get("sqft_living", "1800").strip(),
                "sqft_lot": request.form.get("sqft_lot", "6000").strip(),
                "floors": request.form.get("floors", "1.0").strip(),
                "sqft_above": request.form.get("sqft_above", "1800").strip(),
                "sqft_basement": request.form.get("sqft_basement", "0").strip(),
                "waterfront": request.form.get("waterfront", "0").strip(),
                "view": request.form.get("view", "0").strip(),
                "condition": request.form.get("condition", "3").strip(),
                "yr_built": request.form.get("yr_built", "1990").strip(),
                "yr_renovated": request.form.get("yr_renovated", "0").strip(),
                "date": request.form.get("date", "2014-06-01").strip()
            }

            try:
                beds = float(form_data["bedrooms"])
                baths = float(form_data["bathrooms"])
                sqft_liv = float(form_data["sqft_living"])
                sqft_l = float(form_data["sqft_lot"])
                flrs = float(form_data["floors"])
                sqft_ab = float(form_data["sqft_above"])
                sqft_b = float(form_data["sqft_basement"])
                yr_b = int(float(form_data["yr_built"]))
                yr_r = int(float(form_data["yr_renovated"]))
            except (ValueError, TypeError) as val_err:
                raise ValueError(f"Invalid numeric input detected: {val_err}")

            if sqft_liv <= 0 or sqft_l <= 0:
                raise ValueError("Square footage values must be greater than 0.")

            if beds < 0 or baths < 0 or flrs <= 0:
                raise ValueError("Bedrooms, bathrooms, and floors cannot be negative.")

            if yr_b < 1850 or yr_b > 2030:
                raise ValueError("Year Built must be between 1850 and 2030.")

            predicted_value = pipeline.predict(form_data)
            prediction_text = f"${predicted_value:,.2f}"

            sale_year = int(form_data["date"][:4]) if len(form_data["date"]) >= 4 else 2014
            house_age = max(0, sale_year - yr_b)
            price_per_sqft = round(predicted_value / max(1.0, sqft_liv), 2)
            reno_status = f"Renovated ({yr_r})" if yr_r > 0 else "Original Condition"
            total_effective_sqft = sqft_liv + sqft_b

            breakdown = {
                "price_per_sqft": f"{price_per_sqft:,.2f}",
                "house_age": house_age,
                "reno_status": reno_status,
                "total_sqft": f"{int(total_effective_sqft):,}"
            }

        except Exception as e:
            logger.error(f"Error handling prediction request: {e}")
            error_text = str(e)

    return render_template(
        "index.html",
        cities=CITIES_LIST,
        prediction_text=prediction_text,
        error_text=error_text,
        breakdown=breakdown,
        form_data=form_data
    )


@app.route("/analytics", methods=["GET"])
def analytics():
    plots = [
        {
            "id": "actual_vs_predicted",
            "title": "Actual vs. Predicted Valuation",
            "file": "actual_vs_predicted.png",
            "desc": "Evaluates predicted house prices against actual recorded sales values on the holdout test set with regression trend line."
        },
        {
            "id": "residuals_distribution",
            "title": "Residuals & Prediction Error Distribution",
            "file": "residuals_distribution.png",
            "desc": "Histogram and KDE density estimation validating zero-mean residual normality."
        },
        {
            "id": "feature_coefficients",
            "title": "Standardized Regression Coefficients (Feature Weights)",
            "file": "feature_coefficients.png",
            "desc": "Relative impact of each standardized engineered feature on property pricing."
        },
        {
            "id": "correlation_heatmap",
            "title": "Correlation Heatmap (Top Predictive Features)",
            "file": "correlation_heatmap.png",
            "desc": "Pearson correlation matrix identifying multicollinearity and target relationship."
        },
        {
            "id": "model_comparison",
            "title": "Model Comparison: MLR vs. Polynomial Regression (Degree 2)",
            "file": "model_comparison.png",
            "desc": "Cross-model benchmark on Test R², RMSE, and MAE showing generalization capability."
        }
    ]
    return render_template("analytics.html", plots=plots)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        if pipeline is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json(force=True)
        prediction = pipeline.predict(data)
        return jsonify({
            "status": "success",
            "predicted_price": prediction,
            "formatted_price": f"${prediction:,.2f}"
        })
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy" if pipeline is not None else "degraded"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True)
