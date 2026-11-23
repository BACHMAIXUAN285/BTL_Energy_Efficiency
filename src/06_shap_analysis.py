"""Generate SHAP summary plots for model interpretation and save to reports."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "app" / "best_model.pkl"
REPORTS_DIR = ROOT / "reports" / "shap"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    model_data = joblib.load(MODEL_PATH)
    pipeline = model_data["pipeline"]
    feature_names = model_data["feature_names"]
    target_names = model_data["target_names"]

    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]

    # Load raw data for background distribution
    raw_data_path = ROOT / "data" / "raw" / "ENB2012_data.csv"
    raw_data = pd.read_csv(raw_data_path)
    X_raw = raw_data[[f"X{i}" for i in range(1, 9)]]

    # Transform background data
    X_bg = preprocessor.transform(X_raw.sample(n=200, random_state=42))
    X_bg_df = pd.DataFrame(X_bg, columns=feature_names)

    def model_fn(X):
        return regressor.predict(X)

    explainer = shap.Explainer(model_fn, X_bg_df, feature_names=feature_names)

    # Compute SHAP values for all background samples
    shap_values = explainer(X_bg_df)

    # Summary plots
    for target_idx, target_name in enumerate(target_names):
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values[:, :, target_idx],
            X_bg_df,
            plot_type="bar",
            show=False,
        )
        plt.title(f"SHAP Feature Importance - {target_name}")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / f"shap_summary_bar_{target_name}.png", dpi=150)
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values[:, :, target_idx],
            X_bg_df,
            plot_type="dot",
            show=False,
        )
        plt.title(f"SHAP Summary - {target_name}")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / f"shap_summary_dot_{target_name}.png", dpi=150)
        plt.close()

    # Waterfall plots for a few example predictions
    X_sample = preprocessor.transform(X_raw.sample(n=5, random_state=42))
    X_sample_df = pd.DataFrame(X_sample, columns=feature_names)
    shap_values_sample = explainer(X_sample_df)

    for target_idx, target_name in enumerate(target_names):
        for i in range(min(3, len(X_sample_df))):
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values_sample[i, :, target_idx], max_display=14, show=False)
            plt.title(f"{target_name} - Example {i+1} Waterfall")
            plt.tight_layout()
            plt.savefig(REPORTS_DIR / f"waterfall_{target_name}_ex{i+1}.png", dpi=150)
            plt.close()

    # Save SHAP values for reference
    shap_df = pd.DataFrame()
    for target_idx, target_name in enumerate(target_names):
        for feat_idx, feat_name in enumerate(feature_names):
            shap_df[f"{target_name}_{feat_name}"] = shap_values.values[:, feat_idx, target_idx]

    shap_df.to_csv(REPORTS_DIR / "shap_values.csv", index=False)

    # Feature importance from Ridge coefficients
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Heating_Load_Coef": regressor.coef_[0],
        "Cooling_Load_Coef": regressor.coef_[1],
    })
    coef_df.to_csv(REPORTS_DIR / "ridge_coefficients.csv", index=False)

    print(f"SHAP plots saved to: {REPORTS_DIR}")
    print("Generated files:")
    for f in REPORTS_DIR.glob("*"):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()