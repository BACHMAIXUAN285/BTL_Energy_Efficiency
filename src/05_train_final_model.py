"""Train final Ridge model on full dataset and save for Streamlit app."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "ENB2012_data.csv"
MODEL_PATH = ROOT / "app" / "best_model.pkl"

FEATURE_COLUMNS = [f"X{i}" for i in range(1, 9)]
TARGET_COLUMNS = ["Y1", "Y2"]
NUMERIC_FEATURES = ["X1", "X2", "X3", "X4", "X5", "X7"]
CATEGORICAL_FEATURES = ["X6", "X8"]


def load_raw_data(data_path: Path) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    return data[FEATURE_COLUMNS + TARGET_COLUMNS].copy()


def make_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", MinMaxScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )


def get_feature_names_after_preprocessing(pipeline: Pipeline, original_features: list[str]) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    cat_encoder = preprocessor.named_transformers_["categorical"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    numeric_names = NUMERIC_FEATURES
    return numeric_names + cat_feature_names


def main() -> None:
    data = load_raw_data(DATA_PATH)
    features = data[FEATURE_COLUMNS]
    targets = data[TARGET_COLUMNS]

    pipeline = make_pipeline()
    pipeline.fit(features, targets)

    feature_names = get_feature_names_after_preprocessing(pipeline, FEATURE_COLUMNS)
    target_names = ["Heating_Load", "Cooling_Load"]

    # Set feature names on the regressor to avoid sklearn warnings
    regressor = pipeline.named_steps["regressor"]
    regressor.feature_names_in_ = np.array(feature_names)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_names": feature_names,
            "target_names": target_names,
            "original_feature_columns": FEATURE_COLUMNS,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        },
        MODEL_PATH,
    )

    print(f"Model trained on {len(features)} samples")
    print(f"Feature names after preprocessing: {feature_names}")
    print(f"Target names: {target_names}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()