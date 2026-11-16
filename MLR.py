import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

if __name__ == "__main__":
    sys.modules["source_code"] = sys.modules[__name__]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Aesthetic styling for Seaborn & Matplotlib plots
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "figure.autolayout": True,
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 11
})


class HousePricePipeline:
    """
    End-to-End House Price Valuation Pipeline powered by Scikit-Learn.
    Handles feature engineering, target encoding, standard scaling,
    and multiple linear regression modeling.
    """
    __module__ = "source_code"

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.city_mean = {}
        self.global_mean = 550000.0
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)
        self.feature_names = [
            "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
            "waterfront", "view", "condition", "sqft_above", "sqft_basement",
            "house_age", "is_renovated", "years_since_reno", "has_basement",
            "living_to_lot_ratio", "bed_to_bath_ratio", "sqft_per_room",
            "total_sqft", "city_encoded"
        ]

    def engineer_features_single(self, params):
        """
        Engineers 19 domain-specific features from a single dictionary of inputs.
        """
        try:
            raw_date = str(params.get("date", "2014")).strip()
            try:
                sale_year = int(raw_date[:4])
            except (ValueError, IndexError):
                sale_year = 2014

            bedrooms = float(params.get("bedrooms", 3))
            bathrooms = float(params.get("bathrooms", 2.0))
            sqft_living = float(params.get("sqft_living", 1800))
            sqft_lot = float(params.get("sqft_lot", 5000))
            floors = float(params.get("floors", 1.0))
            waterfront = 1.0 if str(params.get("waterfront", "0")).lower() in ["1", "true", "yes"] else 0.0
            view = float(params.get("view", 0))
            condition = float(params.get("condition", 3))
            sqft_above = float(params.get("sqft_above", sqft_living))
            sqft_basement = float(params.get("sqft_basement", 0))
            yr_built = float(params.get("yr_built", 1980))
            yr_renovated = float(params.get("yr_renovated", 0))
            city = str(params.get("city", "Seattle")).strip()

            house_age = max(0.0, sale_year - yr_built)
            is_renovated = 1.0 if yr_renovated > 0 else 0.0
            years_since_reno = max(0.0, sale_year - yr_renovated) if yr_renovated > 0 else house_age
            has_basement = 1.0 if sqft_basement > 0 else 0.0
            living_to_lot_ratio = sqft_living / (sqft_lot + 1.0)
            bed_to_bath_ratio = bedrooms / (bathrooms + 0.1)
            sqft_per_room = sqft_living / (bedrooms + bathrooms + 0.1)
            total_sqft = sqft_living + sqft_basement
            city_val = self.city_mean.get(city, self.global_mean) / 100000.0

            return [
                bedrooms, bathrooms, sqft_living, sqft_lot, floors,
                waterfront, view, condition, sqft_above, sqft_basement,
                house_age, is_renovated, years_since_reno, has_basement,
                living_to_lot_ratio, bed_to_bath_ratio, sqft_per_room,
                total_sqft, city_val
            ]
        except Exception as e:
            logger.error(f"Error in single feature engineering: {e}")
            raise

    def engineer_features_df(self, df):
        """
        Vectorized feature engineering on a Pandas DataFrame for training and batch evaluation.
        """
        try:
            sale_year = pd.to_datetime(df["date"], errors="coerce").dt.year.fillna(2014)
            bedrooms = df["bedrooms"].astype(float)
            bathrooms = df["bathrooms"].astype(float)
            sqft_living = df["sqft_living"].astype(float)
            sqft_lot = df["sqft_lot"].astype(float)
            floors = df["floors"].astype(float)
            waterfront = df["waterfront"].astype(float)
            view = df["view"].astype(float)
            condition = df["condition"].astype(float)
            sqft_above = df["sqft_above"].astype(float)
            sqft_basement = df["sqft_basement"].astype(float)
            yr_built = df["yr_built"].astype(float)
            yr_renovated = df["yr_renovated"].astype(float)

            house_age = np.maximum(0.0, sale_year - yr_built)
            is_renovated = (yr_renovated > 0).astype(float)
            years_since_reno = np.where(yr_renovated > 0, np.maximum(0.0, sale_year - yr_renovated), house_age)
            has_basement = (sqft_basement > 0).astype(float)
            living_to_lot_ratio = sqft_living / (sqft_lot + 1.0)
            bed_to_bath_ratio = bedrooms / (bathrooms + 0.1)
            sqft_per_room = sqft_living / (bedrooms + bathrooms + 0.1)
            total_sqft = sqft_living + sqft_basement
            city_encoded = df["city"].map(self.city_mean).fillna(self.global_mean) / 100000.0

            feats = pd.DataFrame({
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "sqft_living": sqft_living,
                "sqft_lot": sqft_lot,
                "floors": floors,
                "waterfront": waterfront,
                "view": view,
                "condition": condition,
                "sqft_above": sqft_above,
                "sqft_basement": sqft_basement,
                "house_age": house_age,
                "is_renovated": is_renovated,
                "years_since_reno": years_since_reno,
                "has_basement": has_basement,
                "living_to_lot_ratio": living_to_lot_ratio,
                "bed_to_bath_ratio": bed_to_bath_ratio,
                "sqft_per_room": sqft_per_room,
                "total_sqft": total_sqft,
                "city_encoded": city_encoded
            })
            return feats[self.feature_names]
        except Exception as e:
            logger.error(f"Error in batch DataFrame feature engineering: {e}")
            raise

    def fit(self, train_df):
        """
        Fits the Scikit-Learn pipeline (StandardScaler + Ridge Regression) using training data.
        """
        try:
            self.city_mean = train_df.groupby("city")["price"].mean().to_dict()
            self.global_mean = float(train_df["price"].mean())

            X_train = self.engineer_features_df(train_df)
            y_train = train_df["price"].values

            X_scaled = self.scaler.fit_transform(X_train)
            self.model.fit(X_scaled, y_train)
            return self
        except Exception as e:
            logger.error(f"Error fitting HousePricePipeline: {e}")
            raise

    def predict(self, raw_params):
        """
        Predicts house price for user input dictionary using fitted StandardScaler and Ridge model.
        """
        try:
            feat_vec = self.engineer_features_single(raw_params)
            feat_df = pd.DataFrame([feat_vec], columns=self.feature_names)
            scaled_vec = self.scaler.transform(feat_df)
            raw_pred = self.model.predict(scaled_vec)[0]
            predicted_price = max(50000.0, float(raw_pred))
            return round(predicted_price, 2)
        except Exception as e:
            logger.error(f"Prediction failed in pipeline: {e}")
            raise


# Backwards compatibility aliases
StandardDataScaler = StandardScaler
RegressionModel = Ridge


def load_dataset(file_path="data.csv"):
    """
    Loads and filters dataset using Pandas, filtering out non-representative outliers.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at: {file_path}")

        df = pd.read_csv(file_path)
        logger.info(f"Raw dataset shape: {df.shape}")

        df = df[(df["price"] >= 50000) & (df["price"] <= 3500000)].copy()
        df = df.dropna().reset_index(drop=True)
        logger.info(f"Cleaned dataset shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def generate_visualizations(pipeline, X_train_df, X_test_df, y_train, y_test, comparison_results, plot_dirs):
    """
    Generates publication-grade analytical and diagnostic charts using Matplotlib and Seaborn.
    """
    try:
        for pdir in plot_dirs:
            os.makedirs(pdir, exist_ok=True)

        X_test_scaled = pipeline.scaler.transform(X_test_df)
        y_pred = pipeline.model.predict(X_test_scaled)
        residuals = y_test - y_pred

        # -------------------------------------------------------------
        # 1. Actual vs Predicted Scatter Plot with Seaborn
        # -------------------------------------------------------------
        plt.figure(figsize=(9, 6))
        sns.scatterplot(
            x=y_test / 1000.0,
            y=y_pred / 1000.0,
            alpha=0.45,
            color="#2563eb",
            edgecolor=None,
            s=40
        )
        min_val = min(y_test.min(), y_pred.min()) / 1000.0
        max_val = max(y_test.max(), y_pred.max()) / 1000.0
        plt.plot([min_val, max_val], [min_val, max_val], color="#ef4444", linestyle="--", linewidth=2, label="Ideal Line (y = x)")

        sns.regplot(
            x=y_test / 1000.0,
            y=y_pred / 1000.0,
            scatter=False,
            color="#0f172a",
            line_kws={"linewidth": 1.5, "linestyle": "-"},
            label="Regression Trend"
        )

        r2_val = r2_score(y_test, y_pred)
        mae_val = mean_absolute_error(y_test, y_pred)
        plt.title(f"Actual vs. Predicted House Prices (Test Set) | $R^2 = {r2_val:.4f}$, MAE = ${mae_val:,.0f}", fontweight="bold", pad=12)
        plt.xlabel("Actual Price ($ in Thousands)")
        plt.ylabel("Predicted Price ($ in Thousands)")
        plt.legend(frameon=True, facecolor="white", framealpha=0.9)
        plt.tight_layout()

        for pdir in plot_dirs:
            plt.savefig(os.path.join(pdir, "actual_vs_predicted.png"), dpi=300)
        plt.close()

        # -------------------------------------------------------------
        # 2. Residuals Distribution with Seaborn KDE & Histogram
        # -------------------------------------------------------------
        plt.figure(figsize=(9, 6))
        sns.histplot(
            residuals / 1000.0,
            kde=True,
            bins=50,
            color="#06b6d4",
            edgecolor="#0891b2",
            line_kws={"linewidth": 2}
        )
        plt.axvline(0, color="#ef4444", linestyle="--", linewidth=2, label="Zero Error Line")
        plt.title(f"Residuals Error Distribution (Test Set) | Mean = ${residuals.mean():,.0f}, Std = ${residuals.std():,.0f}", fontweight="bold", pad=12)
        plt.xlabel("Residuals (Actual - Predicted) in $ Thousands")
        plt.ylabel("Frequency")
        plt.legend(frameon=True, facecolor="white", framealpha=0.9)
        plt.tight_layout()

        for pdir in plot_dirs:
            plt.savefig(os.path.join(pdir, "residuals_distribution.png"), dpi=300)
        plt.close()

        # -------------------------------------------------------------
        # 3. Model Coefficients / Feature Importances with Seaborn
        # -------------------------------------------------------------
        coef_df = pd.DataFrame({
            "Feature": pipeline.feature_names,
            "Coefficient": pipeline.model.coef_
        }).sort_values(by="Coefficient", ascending=False)

        plt.figure(figsize=(10, 7))
        palette = ["#10b981" if c > 0 else "#ef4444" for c in coef_df["Coefficient"]]
        sns.barplot(
            data=coef_df,
            x="Coefficient",
            y="Feature",
            palette=palette,
            hue="Feature",
            legend=False
        )
        plt.axvline(0, color="#64748b", linewidth=1.2)
        plt.title("Multiple Linear Regression Standardized Coefficients (Feature Weights)", fontweight="bold", pad=12)
        plt.xlabel("Standardized Coefficient Value ($ impact per standard deviation)")
        plt.ylabel("Engineered Feature")
        plt.tight_layout()

        for pdir in plot_dirs:
            plt.savefig(os.path.join(pdir, "feature_coefficients.png"), dpi=300)
        plt.close()

        # -------------------------------------------------------------
        # 4. Correlation Heatmap of Engineered Features with Seaborn
        # -------------------------------------------------------------
        corr_data = X_train_df.copy()
        corr_data["price"] = y_train
        top_corr_features = corr_data.corr()["price"].abs().sort_values(ascending=False).head(10).index
        sub_corr = corr_data[top_corr_features].corr()

        plt.figure(figsize=(9, 8))
        sns.heatmap(
            sub_corr,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=True,
            square=True,
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": 9}
        )
        plt.title("Correlation Heatmap: Top Predictive Features vs House Price", fontweight="bold", pad=14)
        plt.tight_layout()

        for pdir in plot_dirs:
            plt.savefig(os.path.join(pdir, "correlation_heatmap.png"), dpi=300)
        plt.close()

        # -------------------------------------------------------------
        # 5. Model Comparison: MLR vs Polynomial Regression (Degree 2)
        # -------------------------------------------------------------
        comp_df = pd.DataFrame(comparison_results)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # R2 Comparison
        sns.barplot(
            ax=axes[0],
            data=comp_df,
            x="Model",
            y="Test R2",
            hue="Model",
            palette=["#2563eb", "#f59e0b"],
            legend=False
        )
        axes[0].set_title("Test R² Score (Higher is Better)", fontweight="bold")
        axes[0].set_ylim([min(0.0, comp_df["Test R2"].min() - 0.1), 1.0])
        for p in axes[0].patches:
            axes[0].annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', fontsize=10, xytext=(0, 4), textcoords='offset points')

        # RMSE Comparison
        sns.barplot(
            ax=axes[1],
            data=comp_df,
            x="Model",
            y="Test RMSE ($)",
            hue="Model",
            palette=["#2563eb", "#f59e0b"],
            legend=False
        )
        axes[1].set_title("Test RMSE (Lower is Better)", fontweight="bold")
        for p in axes[1].patches:
            axes[1].annotate(f"${p.get_height():,.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', fontsize=10, xytext=(0, 4), textcoords='offset points')

        # MAE Comparison
        sns.barplot(
            ax=axes[2],
            data=comp_df,
            x="Model",
            y="Test MAE ($)",
            hue="Model",
            palette=["#2563eb", "#f59e0b"],
            legend=False
        )
        axes[2].set_title("Test MAE (Lower is Better)", fontweight="bold")
        for p in axes[2].patches:
            axes[2].annotate(f"${p.get_height():,.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', fontsize=10, xytext=(0, 4), textcoords='offset points')

        plt.suptitle("Model Evaluation & Comparison: Multiple Linear Regression vs Polynomial (Degree 2)", fontsize=14, fontweight="bold")
        plt.tight_layout()

        for pdir in plot_dirs:
            plt.savefig(os.path.join(pdir, "model_comparison.png"), dpi=300)
        plt.close()

        logger.info(f"Generated 5 diagnostic visualization charts in {plot_dirs}")
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        raise


def train_pipeline(data_path="data.csv", output_pickle="model.pkl"):
    """
    Executes end-to-end ML pipeline with Scikit-Learn:
    1. Loads dataset using Pandas
    2. Splits 80/20 using Scikit-Learn's train_test_split
    3. Fits HousePricePipeline with StandardScaler and Ridge/LinearRegression
    4. Evaluates and benchmarks against Polynomial Regression (Degree 2)
    5. Produces charts using Matplotlib & Seaborn
    6. Serializes model to model.pkl
    """
    try:
        df = load_dataset(data_path)

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

        pipeline = HousePricePipeline(alpha=1.0)
        pipeline.fit(train_df)

        X_train_df = pipeline.engineer_features_df(train_df)
        X_test_df = pipeline.engineer_features_df(test_df)
        y_train = train_df["price"].values
        y_test = test_df["price"].values

        X_train_scaled = pipeline.scaler.transform(X_train_df)
        X_test_scaled = pipeline.scaler.transform(X_test_df)

        mlr_train_preds = pipeline.model.predict(X_train_scaled)
        mlr_test_preds = pipeline.model.predict(X_test_scaled)

        mlr_train_r2 = r2_score(y_train, mlr_train_preds)
        mlr_test_r2 = r2_score(y_test, mlr_test_preds)
        mlr_test_rmse = np.sqrt(mean_squared_error(y_test, mlr_test_preds))
        mlr_test_mae = mean_absolute_error(y_test, mlr_test_preds)

        logger.info("--- Multiple Linear Regression (MLR) Performance ---")
        logger.info(f"Train R²: {mlr_train_r2:.4f}")
        logger.info(f"Test R²:  {mlr_test_r2:.4f}")
        logger.info(f"Test RMSE: ${mlr_test_rmse:,.2f}")
        logger.info(f"Test MAE:  ${mlr_test_mae:,.2f}")

        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)

        poly_model = Ridge(alpha=100.0)
        poly_model.fit(X_train_poly, y_train)

        poly_train_preds = poly_model.predict(X_train_poly)
        poly_test_preds = poly_model.predict(X_test_poly)

        poly_train_r2 = r2_score(y_train, poly_train_preds)
        poly_test_r2 = r2_score(y_test, poly_test_preds)
        poly_test_rmse = np.sqrt(mean_squared_error(y_test, poly_test_preds))
        poly_test_mae = mean_absolute_error(y_test, poly_test_preds)

        logger.info("--- Polynomial Regression (Degree 2) Performance ---")
        logger.info(f"Train R²: {poly_train_r2:.4f}")
        logger.info(f"Test R²:  {poly_test_r2:.4f}")
        logger.info(f"Test RMSE: ${poly_test_rmse:,.2f}")
        logger.info(f"Test MAE:  ${poly_test_mae:,.2f}")

        comparison_results = [
            {
                "Model": "Multiple Linear Reg",
                "Train R2": mlr_train_r2,
                "Test R2": mlr_test_r2,
                "Test RMSE ($)": mlr_test_rmse,
                "Test MAE ($)": mlr_test_mae
            },
            {
                "Model": "Polynomial Reg (Deg 2)",
                "Train R2": poly_train_r2,
                "Test R2": poly_test_r2,
                "Test RMSE ($)": poly_test_rmse,
                "Test MAE ($)": poly_test_mae
            }
        ]

        plot_directories = [
            os.path.join(os.path.dirname(__file__), "plots"),
            os.path.join(os.path.dirname(__file__), "static", "plots")
        ]
        generate_visualizations(
            pipeline, X_train_df, X_test_df, y_train, y_test, comparison_results, plot_directories
        )

        with open(output_pickle, "wb") as f:
            pickle.dump(pipeline, f)
        logger.info(f"Successfully serialized trained pipeline to {output_pickle}")

        sample_test_input = {
            "bedrooms": 4,
            "bathrooms": 2.5,
            "sqft_living": 2400,
            "sqft_lot": 7500,
            "floors": 2.0,
            "waterfront": "0",
            "view": 1,
            "condition": 4,
            "sqft_above": 2000,
            "sqft_basement": 400,
            "yr_built": 1995,
            "yr_renovated": 2010,
            "city": "Seattle",
            "date": "2014-06-15"
        }
        test_pred = pipeline.predict(sample_test_input)
        logger.info(f"Sample test prediction: ${test_pred:,.2f}")

        return pipeline
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise


if __name__ == "__main__":
    data_file = os.path.join(os.path.dirname(__file__), "data.csv")
    model_file = os.path.join(os.path.dirname(__file__), "model.pkl")
    train_pipeline(data_path=data_file, output_pickle=model_file)
