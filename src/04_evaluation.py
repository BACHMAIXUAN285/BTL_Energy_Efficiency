"""Cross-validate energy-load regressors without preprocessing leakage."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = [f"X{i}" for i in range(1, 9)]
TARGET_COLUMNS = ["Y1", "Y2"]
REQUIRED_COLUMNS = FEATURE_COLUMNS + TARGET_COLUMNS
NUMERIC_FEATURES = ["X1", "X2", "X3", "X4", "X5", "X7"]
CATEGORICAL_FEATURES = ["X6", "X8"]
GROUP_FEATURES = ["X1", "X2", "X3", "X4", "X5"]
METRIC_NAMES = ["rmse", "mae", "r2", "mape_percent"]


def load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy tệp dữ liệu: {data_path}. "
            "Hãy truyền đường dẫn CSV hoặc Excel đúng bằng tham số --data."
        )

    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        data = pd.read_csv(data_path)
    elif suffix in {".xls", ".xlsx", ".xlsm"}:
        try:
            data = pd.read_excel(data_path)
        except ImportError as exc:
            raise ImportError(
                "Không thể đọc Excel; hãy cài openpyxl bằng "
                "'python -m pip install openpyxl'."
            ) from exc
    else:
        raise ValueError(
            f"Định dạng tệp không được hỗ trợ: {suffix or '(không có phần mở rộng)'}. "
            "Hãy dùng .csv, .xls, .xlsx hoặc .xlsm."
        )

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(
            f"Tệp {data_path} thiếu các cột bắt buộc: {', '.join(missing_columns)}."
        )

    data = data[REQUIRED_COLUMNS].copy()
    missing_counts = data.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        details = ", ".join(
            f"{column}: {count}" for column, count in missing_counts.items()
        )
        raise ValueError(f"Dữ liệu có giá trị trống ({details}).")

    for column in REQUIRED_COLUMNS:
        converted = pd.to_numeric(data[column], errors="coerce")
        invalid_mask = converted.isna()
        if invalid_mask.any():
            invalid_rows = (np.flatnonzero(invalid_mask.to_numpy())[:5] + 2).tolist()
            raise ValueError(
                f"Cột {column} có {int(invalid_mask.sum())} giá trị không hợp lệ "
                f"(không phải số); dòng dữ liệu ví dụ: {invalid_rows}."
            )
        if not np.isfinite(converted.to_numpy(dtype=float)).all():
            raise ValueError(f"Cột {column} chứa giá trị không hữu hạn.")
        data[column] = converted

    zero_targets = {
        target: int(data[target].eq(0).sum())
        for target in TARGET_COLUMNS
        if data[target].eq(0).any()
    }
    if zero_targets:
        details = ", ".join(f"{target}: {count}" for target, count in zero_targets.items())
        raise ValueError(
            f"MAPE không xác định khi target bằng 0; phát hiện giá trị 0 ({details})."
        )

    return data


def make_model_factories(
    baseline_only: bool,
) -> list[tuple[str, Callable[[], object]]]:
    models: list[tuple[str, Callable[[], object]]] = [
        ("Linear Regression", LinearRegression),
    ]
    if baseline_only:
        return models

    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError(
            "Chế độ đầy đủ cần LightGBM. Cài bằng "
            "'python -m pip install lightgbm' hoặc 'python -m pip install -r requirements.txt'."
        ) from exc

    models.extend(
        [
            ("Ridge", lambda: Ridge(alpha=1.0)),
            ("Lasso", lambda: Lasso(alpha=0.01, max_iter=20000)),
            (
                "Random Forest",
                lambda: RandomForestRegressor(n_estimators=250, random_state=42),
            ),
            (
                "Gradient Boosting",
                lambda: GradientBoostingRegressor(
                    n_estimators=350,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42,
                ),
            ),
            (
                "LightGBM",
                lambda: LGBMRegressor(
                    n_estimators=350,
                    learning_rate=0.05,
                    num_leaves=24,
                    min_child_samples=8,
                    random_state=42,
                    verbosity=-1,
                ),
            ),
        ]
    )
    return models


def make_pipeline(estimator: object) -> Pipeline:
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
            ("regressor", estimator),
        ]
    )


def build_splits(
    features: pd.DataFrame,
) -> list[tuple[str, list[tuple[np.ndarray, np.ndarray]]]]:
    group_values = np.empty(len(features), dtype=object)
    for index, values in enumerate(
        features[GROUP_FEATURES].itertuples(index=False, name=None)
    ):
        group_values[index] = values
    splitters = [
        ("KFold", KFold(n_splits=5, shuffle=True, random_state=42), None),
        ("GroupKFold", GroupKFold(n_splits=5), group_values),
    ]
    results = []
    for name, splitter, groups in splitters:
        folds = list(splitter.split(features, groups=groups))
        validation_counts = np.zeros(len(features), dtype=int)
        validation_group_fold: dict[tuple[float, ...], int] = {}
        for fold_index, (train_indices, validation_indices) in enumerate(folds):
            validation_counts[validation_indices] += 1
            if groups is not None:
                train_groups = {groups[index] for index in train_indices}
                validation_groups = {groups[index] for index in validation_indices}
                if train_groups & validation_groups:
                    raise RuntimeError(
                        f"{name} fold {fold_index + 1} trùng nhóm hình học giữa train/validation."
                    )
                for group in validation_groups:
                    previous_fold = validation_group_fold.setdefault(group, fold_index)
                    if previous_fold != fold_index:
                        raise RuntimeError(
                            f"{name} đưa cùng nhóm hình học vào nhiều fold validation."
                        )

        if not np.all(validation_counts == 1):
            raise RuntimeError(
                f"{name} không dùng mỗi mẫu đúng một lần làm validation."
            )
        results.append((name, folds))
    return results


def evaluate(
    data: pd.DataFrame,
    models: list[tuple[str, Callable[[], object]]],
) -> pd.DataFrame:
    features = data[FEATURE_COLUMNS]
    splits = build_splits(features)
    fold_metrics: list[dict[str, object]] = []

    for split_name, folds in splits:
        print(f"Đang đánh giá {split_name} ({len(folds)} folds)...")
        for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
            X_train = features.iloc[train_indices]
            X_validation = features.iloc[validation_indices]
            for target in TARGET_COLUMNS:
                y_train = data[target].iloc[train_indices]
                y_validation = data[target].iloc[validation_indices]
                for model_name, estimator_factory in models:
                    pipeline = make_pipeline(estimator_factory())
                    pipeline.fit(X_train, y_train)
                    predictions = pipeline.predict(X_validation)
                    metrics = {
                        "rmse": float(np.sqrt(mean_squared_error(y_validation, predictions))),
                        "mae": float(mean_absolute_error(y_validation, predictions)),
                        "r2": float(r2_score(y_validation, predictions)),
                        "mape_percent": float(
                            mean_absolute_percentage_error(y_validation, predictions) * 100
                        ),
                    }
                    if not np.isfinite(list(metrics.values())).all():
                        raise RuntimeError(
                            f"Chỉ số không hữu hạn: {split_name}, fold {fold_number}, "
                            f"{model_name}, {target}."
                        )
                    fold_metrics.append(
                        {
                            "split": split_name,
                            "fold": fold_number,
                            "model": model_name,
                            "target": target,
                            "validation_samples": len(validation_indices),
                            **metrics,
                        }
                    )

    return pd.DataFrame(fold_metrics)


def summarize(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    summary = fold_metrics.groupby(["split", "target", "model"])[METRIC_NAMES].agg(
        ["mean", "std"]
    )
    summary.columns = [
        f"{metric}_{statistic}" for metric, statistic in summary.columns
    ]
    summary = summary.reset_index()
    return summary.sort_values(
        ["split", "target", "rmse_mean"], kind="stable"
    ).reset_index(drop=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="backslashreplace")

    parser = argparse.ArgumentParser(
        description="Đánh giá mô hình dự báo Heating_Load và Cooling_Load bằng 5-fold CV."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "raw" / "ENB2012_data.csv",
        help=(
            "Đường dẫn tệp CSV hoặc Excel ENB2012 "
            "(mặc định: data/raw/ENB2012_data.csv)."
        ),
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Chỉ chạy Linear Regression, không cần LightGBM.",
    )
    args = parser.parse_args()

    data = load_dataset(args.data)
    print(f"Đã đọc {len(data)} dòng từ {args.data}.")
    models = make_model_factories(args.baseline_only)
    metrics = evaluate(data, models)
    summary = summarize(metrics)

    output_dir = ROOT / "reports" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "fold_metrics.csv"
    summary_path = output_dir / "cv_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Đã lưu {len(metrics)} dòng fold metrics: {metrics_path}")
    print(f"Đã lưu {len(summary)} dòng tổng hợp: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
