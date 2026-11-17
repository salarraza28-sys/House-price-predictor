# Real Estate Valuation AI System

**By A.M Salar Raza | Vihara Tech**

A complete end-to-end Machine Learning web application predicting house prices using the King County real estate dataset. The project leverages **Scikit-Learn** for feature preprocessing and regression modeling, **Matplotlib** and **Seaborn** for exploratory data analysis and visual diagnostics, and **Flask** for an interactive web interface.

---

## 📁 Project Architecture

```
MiniProject/
│
├── data.csv                # King County housing sales dataset (4,600 rows)
├── source_code.py          # Scikit-Learn ML pipeline, evaluation & Seaborn/Matplotlib plotting
├── model.pkl               # Serialized trained Scikit-Learn model & scaler
├── app.py                  # Flask web backend with real-time prediction and analytics routes
├── templates/
│   ├── index.html          # Valuation calculator UI with Sapphire & Cyan theme
│   └── analytics.html      # Visual diagnostics dashboard displaying Seaborn & Matplotlib plots
├── static/plots/           # Generated high-resolution diagnostic charts
│   ├── actual_vs_predicted.png
│   ├── residuals_distribution.png
│   ├── feature_coefficients.png
│   ├── correlation_heatmap.png
│   └── model_comparison.png
└── requirements.txt        # Python dependencies
```

---

## 🔬 Machine Learning Pipeline (Powered by Scikit-Learn)

1. **Robust Data Ingestion & Preprocessing:**
   - Handled with **Pandas** and **Scikit-Learn**.
   - Price outlier filtering ($50,000 to $3,500,000) removes non-representative transaction anomalies.

2. **Feature Engineering (19 Predictive Dimensions):**
   - Temporal Extraction: Sale Year parsed from `date`
   - `house_age = sale_year - yr_built`
   - `is_renovated = 1 if yr_renovated > 0 else 0`
   - `years_since_reno = sale_year - yr_renovated` (or `house_age` if unrenovated)
   - `has_basement = 1 if sqft_basement > 0 else 0`
   - `living_to_lot_ratio = sqft_living / (sqft_lot + 1.0)`
   - `bed_to_bath_ratio = bedrooms / (bathrooms + 0.1)`
   - `sqft_per_room = sqft_living / (bedrooms + bathrooms + 0.1)`
   - `total_sqft = sqft_living + sqft_basement`
   - `city_encoded = City Target Encoding` calculated strictly on the training set to prevent data leakage.

3. **Feature Scaling & Regression:**
   - **`sklearn.preprocessing.StandardScaler`** standardizes feature distributions to zero mean and unit variance.
   - **`sklearn.linear_model.Ridge` / `LinearRegression`** fits the multiple linear regression weights.
   - Evaluated with **`sklearn.metrics`** (`r2_score`, `mean_squared_error`, `mean_absolute_error`).

4. **Train / Test Split (80% / 20%):**
   - Conducted with `sklearn.model_selection.train_test_split(..., test_size=0.2, random_state=42)`.
   - 3,634 Training Samples (80%)
   - 909 Testing Holdout Samples (20%)

5. **Diagnostic Comparison (MLR vs Polynomial Regression):**

| Metric | Multiple Linear Regression (MLR) | Polynomial Regression (Degree 2) | Winner |
|---|---|---|---|
| **Train $R^2$** | `0.7044` | `0.7617` | Polynomial Reg |
| **Test $R^2$** | `0.6524` | `0.6771` | Polynomial Reg |
| **Test RMSE** | `$198,393.22` | `$191,226.34` | Polynomial Reg |
| **Test MAE** | `$123,222.69` | `$113,506.83` | Polynomial Reg |

---

## 📊 Visual Diagnostics (Matplotlib & Seaborn)

The training pipeline generates 5 publication-ready diagnostic visualizations:

1. **Actual vs. Predicted Prices (`actual_vs_predicted.png`)**: Seaborn scatter plot with $y = x$ reference diagonal and linear regression fit.
2. **Residuals Distribution (`residuals_distribution.png`)**: Seaborn histogram + KDE kernel density plot verifying zero-mean normality of prediction errors.
3. **Standardized Feature Coefficients (`feature_coefficients.png`)**: Diverging Seaborn barplot ranking the positive and negative valuation drivers.
4. **Correlation Heatmap (`correlation_heatmap.png`)**: Seaborn blue-scale correlation matrix between primary square-footage metrics, property features, and price.
5. **Model Comparison Benchmarks (`model_comparison.png`)**: Grouped comparison charts comparing $R^2$, RMSE, and MAE.

---

## 🚀 How to Run

### 1. Train and Generate Diagnostic Visuals
```bash
python source_code.py
```

### 2. Launch the Flask Web Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```
To view the visual analytics dashboard, visit:
```
http://127.0.0.1:5000/analytics
```

### 3. API Usage
Send a POST request with property parameters:
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{
       "city": "Seattle",
       "bedrooms": 4,
       "bathrooms": 2.5,
       "sqft_living": 2400,
       "sqft_lot": 7000,
       "floors": 2.0,
       "sqft_above": 2000,
       "sqft_basement": 400,
       "waterfront": "0",
       "view": 1,
       "condition": 4,
       "yr_built": 1995,
       "yr_renovated": 2010,
       "date": "2014-06-15"
     }'
```
