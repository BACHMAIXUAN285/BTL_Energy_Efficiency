"""Streamlit app for Energy Efficiency Prediction with SHAP explanations."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "app" / "best_model.pkl"
REPORTS_SHAP_DIR = ROOT / "reports" / "shap"

FEATURE_INFO = {
    "X1": {"label": "Độ nén tương đối (Relative Compactness)", "min": 0.6, "max": 1.0, "step": 0.01, "default": 0.9},
    "X2": {"label": "Diện tích bề mặt (Surface Area)", "min": 500.0, "max": 900.0, "step": 1.0, "default": 600.0},
    "X3": {"label": "Diện tích tường (Wall Area)", "min": 200.0, "max": 500.0, "step": 1.0, "default": 300.0},
    "X4": {"label": "Diện tích mái (Roof Area)", "min": 100.0, "max": 300.0, "step": 1.0, "default": 150.0},
    "X5": {"label": "Chiều cao tổng thể (Overall Height)", "min": 3.5, "max": 7.0, "step": 0.5, "default": 7.0},
    "X6": {"label": "Hướng tòa nhà (Orientation)", "options": [2, 3, 4, 5], "default": 2, "help": "2=Bắc, 3=Đông, 4=Nam, 5=Tây"},
    "X7": {"label": "Diện tích kính (Glazing Area)", "min": 0.0, "max": 0.4, "step": 0.05, "default": 0.0},
    "X8": {"label": "Phân bố diện tích kính (Glazing Area Distribution)", "options": [0, 1, 2, 3, 4, 5], "default": 0, "help": "0=Đồng đều, 1=Bắc, 2=Đông, 3=Nam, 4=Tây, 5=Không có kính"},
}

ORIGINAL_FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]


@st.cache_resource
def load_model():
    model_data = joblib.load(MODEL_PATH)
    return model_data


@st.cache_resource
def create_explainer(_pipeline, _feature_names):
    preprocessor = _pipeline.named_steps["preprocessor"]
    regressor = _pipeline.named_steps["regressor"]

    X_bg = pd.DataFrame(
        np.random.rand(100, len(_feature_names)),
        columns=_feature_names
    )

    def model_fn(X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=_feature_names)
        return regressor.predict(X)

    explainer = shap.Explainer(model_fn, X_bg, feature_names=_feature_names)
    return explainer, preprocessor


@st.cache_data
def load_shap_summary_plots():
    """Load pre-generated SHAP summary plot images."""
    plots = {}
    for target in ["Heating_Load", "Cooling_Load"]:
        plots[target] = {
            "bar": REPORTS_SHAP_DIR / f"shap_summary_bar_{target}.png",
            "dot": REPORTS_SHAP_DIR / f"shap_summary_dot_{target}.png",
        }
    return plots


def prepare_input_data(inputs: dict) -> pd.DataFrame:
    df = pd.DataFrame([inputs])
    return df[ORIGINAL_FEATURES]


def main():
    st.set_page_config(
        page_title="Dự Báo Năng Lượng Tòa Nhà",
        page_icon="🏢",
        layout="wide",
    )

    st.title("🏢 Dự Báo Tải Năng Lượng Tòa Nhà")
    st.markdown(
        "Công cụ dự báo **tải sưởi ấm (Heating Load)** và **tải làm mát (Cooling Load)** "
        "dựa trên các thông số thiết kế kiến trúc. Sử dụng mô hình **Ridge Regression** "
        "được huấn luyện trên dataset Energy Efficiency (UCI)."
    )

    # CSS căn giữa hình ảnh khi bấm zoom / fullscreen
    st.markdown(
        """
        <style>
        /* Căn giữa container hình ảnh trong chế độ fullscreen */
        div[data-testid="stImageContainer"] {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            width: 100% !important;
        }
        div[data-testid="stImageContainer"] img {
            display: block !important;
            margin: 0 auto !important;
        }
        /* Căn giữa khung fullscreen */
        div[data-testid="stFullScreenFrame"] {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }
        /* Căn giữa toàn bộ block ảnh */
        div[data-testid="stImage"] {
            display: flex !important;
            justify-content: center !important;
            width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    model_data = load_model()
    pipeline = model_data["pipeline"]
    feature_names = model_data["feature_names"]
    target_names = model_data["target_names"]

    with st.sidebar:
        st.header("⚙️ Thông Số Thiết Kế")

        inputs = {}
        for feat, info in FEATURE_INFO.items():
            if "options" in info:
                inputs[feat] = st.selectbox(
                    info["label"],
                    options=info["options"],
                    index=info["options"].index(info["default"]),
                    help=info.get("help", ""),
                )
            else:
                inputs[feat] = st.number_input(
                    info["label"],
                    min_value=info["min"],
                    max_value=info["max"],
                    value=info["default"],
                    step=info["step"],
                )

        predict_button = st.button("🔮 Dự Báo", type="primary", use_container_width=True)

    if predict_button:
        input_df = prepare_input_data(inputs)

        with st.spinner("Đang dự báo..."):
            # Transform with feature names preserved for Ridge
            preprocessor = pipeline.named_steps["preprocessor"]
            regressor = pipeline.named_steps["regressor"]
            X_processed = preprocessor.transform(input_df)
            X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
            predictions = regressor.predict(X_processed_df)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="🔥 Tải Sưởi Ấm (Heating Load)",
                value=f"{predictions[0, 0]:.2f}",
                help="Đơn vị: kWh/m²"
            )
        with col2:
            st.metric(
                label="❄️ Tải Làm Mát (Cooling Load)",
                value=f"{predictions[0, 1]:.2f}",
                help="Đơn vị: kWh/m²"
            )

        st.divider()

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 EDA Khám Phá Dữ Liệu",
            "📈 SHAP Waterfall (Dự báo hiện tại)",
            "📉 SHAP Summary (Toàn cục)",
            "📋 Độ Quan Trọng Đặc Trưng (Ridge Coef)",
            "📝 Chi Tiết Đầu Vào"
        ])

        with tab1:
            st.subheader("Khám Phá Dữ Liệu (EDA)")
            st.markdown(
                "Các biểu đồ dưới đây được tạo từ **Bước 1 (EDA)** của quy trình KDD/CRISP-DM, "
                "giúp hiểu phân phối, tương đồng và phát hiện điểm bất thường."
            )

            eda_dir = ROOT / "reports"

            st.markdown("#### 📊 Ma Trận Tương Quan")
            if (eda_dir / "correlation_heatmap.png").exists():
                st.image(str(eda_dir / "correlation_heatmap.png"), caption="Correlation Heatmap", use_column_width=True)
            else:
                st.warning("Không tìm thấy file: correlation_heatmap.png")

            st.markdown("#### 📈 Phân Bố Biến Mục Tiêu")
            if (eda_dir / "target_distribution.png").exists():
                st.image(str(eda_dir / "target_distribution.png"), caption="Target Distribution", use_column_width=True)
            else:
                st.warning("Không tìm thấy file: target_distribution.png")

            st.markdown("#### 📦 Phát Hiện Điểm Bất Thường")
            if (eda_dir / "outliers_boxplot.png").exists():
                st.image(str(eda_dir / "outliers_boxplot.png"), caption="Outliers Boxplot", use_column_width=True)
            else:
                st.warning("Không tìm thấy file: outliers_boxplot.png")

            st.info(
                "Lưu ý: Tập dữ liệu ENB2012 được giữ nguyên các điểm biên vì chúng "
                "đại diện cho các hình thái kiến trúc hợp lệ, không phải nhiễm lois."
            )

        with tab2:
            st.subheader("Giải Thích Dự Báo Cụ Thể (Waterfall Plot)")
            st.markdown(
                "Biểu đồ dưới đây cho thấy mức độ đóng góp của từng đặc trưng kiến trúc "
                "đối với kết quả dự báo **của mẫu hiện tại** so với giá trị trung bình của tập huấn luyện."
            )

            with st.spinner("Đang tính toán giá trị SHAP..."):
                try:
                    explainer, preprocessor = create_explainer(pipeline, feature_names)
                    X_processed = preprocessor.transform(input_df)
                    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
                    shap_values = explainer(X_processed_df)

                    col_h, col_c = st.columns(2)
                    with col_h:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        shap.plots.waterfall(shap_values[0, :, 0], max_display=14, show=False)
                        plt.title("Heating Load - Waterfall Plot")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                    with col_c:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        shap.plots.waterfall(shap_values[0, :, 1], max_display=14, show=False)
                        plt.title("Cooling Load - Waterfall Plot")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                except Exception as e:
                    st.error(f"Không thể tính toán SHAP: {e}")
                    st.info("Thử lại hoặc kiểm tra phiên bản SHAP.")

        with tab3:
            st.subheader("Phân Tích SHAP Toàn Cục (Global Feature Importance)")
            st.markdown(
                "Các biểu đồ dưới đây được tính từ **200 mẫu ngẫu nhiên** của tập dữ liệu, "
                "cho thấy mức độ quan trọng trung bình của từng đặc trưng trên toàn bộ dữ liệu."
            )

            summary_plots = load_shap_summary_plots()

            for target_name in ["Heating_Load", "Cooling_Load"]:
                st.markdown(f"#### {target_name.replace('_', ' ')}")

                st.markdown("##### 📊 SHAP Feature Importance (Bar)")
                if summary_plots[target_name]["bar"].exists():
                    st.image(str(summary_plots[target_name]["bar"]), caption=f"{target_name} - Bar", use_column_width=True)
                else:
                    st.warning(f"Không tìm thấy file: {summary_plots[target_name]['bar'].name}")

                st.markdown("##### 📉 SHAP Summary (Dot)")
                if summary_plots[target_name]["dot"].exists():
                    st.image(str(summary_plots[target_name]["dot"]), caption=f"{target_name} - Dot", use_column_width=True)
                else:
                    st.warning(f"Không tìm thấy file: {summary_plots[target_name]['dot'].name}")

                st.divider()

        with tab4:
            st.subheader("Độ Quan Trọng Đặc Trưng - Hệ Số Ridge Regression")
            st.markdown(
                "Hệ số hồi quy Ridge (đã chuẩn hóa MinMax) cho thấy tác động tương đối của từng đặc trưng. "
                "**Giá trị dương = tăng tải năng lượng**, **Giá trị âm = giảm tải năng lượng**."
            )

            regressor = pipeline.named_steps["regressor"]
            coef_heating = regressor.coef_[0]
            coef_cooling = regressor.coef_[1]

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Heating_Load_Coef": coef_heating,
                "Cooling_Load_Coef": coef_cooling,
            })
            importance_df["Abs_Mean"] = importance_df[["Heating_Load_Coef", "Cooling_Load_Coef"]].abs().mean(axis=1)
            importance_df = importance_df.sort_values("Abs_Mean", ascending=True)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

            colors_heating = ["red" if c > 0 else "blue" for c in importance_df["Heating_Load_Coef"]]
            ax1.barh(importance_df["Feature"], importance_df["Heating_Load_Coef"], color=colors_heating)
            ax1.set_title("Heating Load - Ridge Coefficients")
            ax1.set_xlabel("Coefficient Value")
            ax1.axvline(x=0, color="black", linewidth=0.5)

            colors_cooling = ["red" if c > 0 else "blue" for c in importance_df["Cooling_Load_Coef"]]
            ax2.barh(importance_df["Feature"], importance_df["Cooling_Load_Coef"], color=colors_cooling)
            ax2.set_title("Cooling Load - Ridge Coefficients")
            ax2.set_xlabel("Coefficient Value")
            ax2.axvline(x=0, color="black", linewidth=0.5)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("#### Bảng hệ số chi tiết")
            st.dataframe(
                importance_df[["Feature", "Heating_Load_Coef", "Cooling_Load_Coef"]].style.format({
                    "Heating_Load_Coef": "{:.4f}",
                    "Cooling_Load_Coef": "{:.4f}",
                }),
                use_container_width=True,
            )

        with tab5:
            st.subheader("Chi Tiết Thông Số Đầu Vào")
            input_display = pd.DataFrame([inputs]).T
            input_display.columns = ["Giá Trị"]
            input_display.index.name = "Tham Số"
            st.dataframe(input_display, use_container_width=True)

            st.download_button(
                label="📥 Tải Kết Quả (JSON)",
                data=json.dumps({
                    "inputs": inputs,
                    "predictions": {
                        "Heating_Load": float(predictions[0, 0]),
                        "Cooling_Load": float(predictions[0, 1]),
                    },
                }, indent=2, ensure_ascii=False),
                file_name="energy_prediction.json",
                mime="application/json",
            )

    else:
        st.info("👈 Vui lòng điều chỉnh thông số ở thanh bên và nhấn **Dự Báo** để xem kết quả.")

        st.divider()
        st.subheader("📚 Hướng Dẫn Sử Dụng")
        st.markdown("""
        **Các thông số đầu vào:**
        - **X1 - Độ nén tương đối**: Tỷ lệ thể tích/bề mặt bao bọc. Giá trị cao = hình dáng compact hơn.
        - **X2 - Diện tích bề mặt**: Tổng diện tích bao bọc tòa nhà (m²).
        - **X3 - Diện tích tường**: Diện tích tường ngoài (m²).
        - **X4 - Diện tích mái**: Diện tích mái nhà (m²).
        - **X5 - Chiều cao tổng thể**: Chiều cao tòa nhà (m).
        - **X6 - Hướng tòa nhà**: 2=Bắc, 3=Đông, 4=Nam, 5=Tây.
        - **X7 - Diện tích kính**: Tỷ lệ diện tích kính so với diện tích sàn (0-0.4).
        - **X8 - Phân bố kính**: 0=Đồng đều, 1=Bắc, 2=Đông, 3=Nam, 4=Tây, 5=Không có kính.

        **Mô hình:** Ridge Regression (α=1.0) với tiền xử lý:
        - MinMaxScaler cho đặc trưng số
        - One-Hot Encoding (drop_first) cho đặc trưng phân loại X6, X8

        **Độ đo:** RMSE, MAE, R², MAPE trên cross-validation GroupKFold (5 folds).
        """)


if __name__ == "__main__":
    main()