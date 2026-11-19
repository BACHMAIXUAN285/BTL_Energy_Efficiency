"""Train and evaluate a baseline energy-efficiency regression model."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed" / "cleaned_energy_data.csv"
MODEL_PATH = ROOT_DIR / "app" / "best_model.pkl"
TARGET_COLUMNS = ["Heating_Load", "Cooling_Load"]


def load_data(data_path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed data and separate features from prediction targets."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu đã xử lý: {data_path}. "
        )

    data = pd.read_csv(data_path)
    missing_targets = [column for column in TARGET_COLUMNS if column not in data]
    if missing_targets:
        raise ValueError(f"Dữ liệu thiếu cột mục tiêu: {missing_targets}")

    features = data.drop(columns=TARGET_COLUMNS)
    targets = data[TARGET_COLUMNS]
    if features.empty or len(data) < 2:
        raise ValueError("Dữ liệu phải có ít nhất một đặc trưng và hai dòng.")
    return features, targets


def train_and_evaluate(
    features: pd.DataFrame, targets: pd.DataFrame
) -> tuple[LinearRegression, dict[str, dict[str, float]]]:
    """Train a linear regression model and return test-set metrics."""
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {}
    for index, target in enumerate(TARGET_COLUMNS):
        metrics[target] = {
            "rmse": float(mean_squared_error(y_test.iloc[:, index], predictions[:, index]) ** 0.5),
            "r2": float(r2_score(y_test.iloc[:, index], predictions[:, index])),
        }
    return model, metrics


def main() -> None:
    """Train the baseline model, print metrics, and persist the model artifact."""
    features, targets = load_data()
    model, metrics = train_and_evaluate(features, targets)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": list(features.columns),
            "target_names": TARGET_COLUMNS,
        },
        MODEL_PATH,
    )

    print(f"Số mẫu: {len(features)} | Đặc trưng: {features.shape[1]}")
    for target, scores in metrics.items():
        print(f"{target}: RMSE={scores['rmse']:.4f}, R2={scores['r2']:.4f}")
    print(f"Mô hình được lưu tại: {MODEL_PATH}")


if __name__ == "__main__":
    main()
