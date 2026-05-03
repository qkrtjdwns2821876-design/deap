from io import BytesIO
from pathlib import Path
import pickle

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "ammonia_25_clean_deploy_model.pkl"
METRICS_PATH = BASE_DIR / "ammonia_25_clean_deploy_metrics.csv"
IMPORTANCE_PATH = BASE_DIR / "ammonia_25_clean_shap_importance.csv"
DATA_PATH = BASE_DIR / "ammonia_data_25.xlsx"
SHAP_SUMMARY_PATH = BASE_DIR / "ammonia_25_clean_shap_summary.png"
SHAP_BAR_PATH = BASE_DIR / "ammonia_25_clean_shap_bar.png"

PREDICTION_COLUMN = "Predicted_NH3_emission_g/kg_Initial_TS"

DISPLAY_LABELS = {
    "Day": "Composting day",
    "Aeration_rate(L/min)": "Aeration rate (L/min)",
    "C_N_Ratio_Initial": "Initial C/N ratio",
    "Maximum_temperature_C": "Maximum temperature (C)",
    "Composting_cycle_length_day": "Composting cycle length (day)",
    "Temperature_C": "Temperature (C)",
    "pH": "pH",
    "EC_(mS cm-1)_Initial": "Initial EC (mS/cm)",
    "MC_(%)_Initial": "Initial moisture content (%)",
}


st.set_page_config(
    page_title="DAEP",
    page_icon="NH3",
    layout="wide",
)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    return pd.DataFrame()


@st.cache_data
def load_feature_importance():
    if IMPORTANCE_PATH.exists():
        return pd.read_csv(IMPORTANCE_PATH)
    return pd.DataFrame()


@st.cache_data
def load_default_data():
    if DATA_PATH.exists():
        return pd.read_excel(DATA_PATH)
    return pd.DataFrame()


def predict_dataframe(df, bundle):
    features = bundle["features"]
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        return None, missing

    result = df.copy()
    pred = np.expm1(bundle["model"].predict(result[features]))
    result[PREDICTION_COLUMN] = np.clip(pred, 0, None)
    return result, []


def estimate_prediction_uncertainty(df, bundle):
    model = bundle["model"]
    transformed = model.named_steps["preprocessor"].transform(df[bundle["features"]])
    trees = getattr(model.named_steps["regressor"], "estimators_", [])
    if not trees:
        return None
    tree_preds = np.array([np.expm1(tree.predict(transformed)) for tree in trees])
    tree_preds = np.clip(tree_preds, 0, None)
    return {
        "lower": np.percentile(tree_preds, 10, axis=0),
        "upper": np.percentile(tree_preds, 90, axis=0),
        "std": np.std(tree_preds, axis=0),
    }


def compute_scores(df, target):
    valid = df[[target, PREDICTION_COLUMN]].dropna()
    if valid.empty:
        return None
    y_true = valid[target].astype(float)
    y_pred = valid[PREDICTION_COLUMN].astype(float)
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "n": len(valid),
    }


def to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
    return output.getvalue()


def format_metric(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.3f}"


def validation_metric(split, model, metric):
    if metrics.empty:
        return "-"
    selected = metrics[(metrics["split"] == split) & (metrics["model"] == model)]
    if selected.empty:
        return "-"
    return format_metric(selected[metric].iloc[0])


bundle = load_model()
features = bundle["features"]
target = bundle["target"]
metrics = load_metrics()
importance = load_feature_importance()
default_data = load_default_data()

st.title("DAEP")
st.caption("Daily Ammonia Emission Prediction for Composting")

summary_cols = st.columns(4)
summary_cols[0].metric("Training rows", f"{bundle.get('training_rows', 0):,}")
if not default_data.empty and "Source" in default_data.columns:
    summary_cols[1].metric("Total sources", f"{default_data['Source'].nunique():,}")
else:
    summary_cols[1].metric("Training sources", f"{bundle.get('source_count', 0):,}")

if not metrics.empty:
    best_random = metrics[
        (metrics["split"] == "random_row_split") & (metrics["model"] == "ExtraTrees")
    ]
    best_source = metrics[
        (metrics["split"] == "source_group_split") & (metrics["model"] == "ExtraTrees")
    ]
    if not best_random.empty:
        summary_cols[2].metric("Random split R2", format_metric(best_random["R2"].iloc[0]))
    if not best_source.empty:
        summary_cols[3].metric("Source split R2", format_metric(best_source["R2"].iloc[0]))

tab_single, tab_file, tab_model = st.tabs(
    ["Prediction calculator", "Batch prediction", "Model results"]
)

with tab_single:
    input_panel, output_panel = st.columns([0.42, 0.58], gap="large")
    values = {}
    defaults = {
        "Day": 5.0,
        "Aeration_rate(L/min)": 2.8,
        "C_N_Ratio_Initial": 18.0,
        "Maximum_temperature_C": 60.0,
        "Composting_cycle_length_day": 30.0,
        "Temperature_C": 55.0,
        "pH": 8.5,
        "EC_(mS cm-1)_Initial": 3.0,
        "MC_(%)_Initial": 65.0,
    }

    with input_panel:
        st.subheader("Input variables")
        wet_weight_kg = st.number_input(
            "Initial compost wet weight (kg)",
            min_value=0.0,
            max_value=1_000_000.0,
            value=100.0,
            step=1.0,
        )
        values["Day"] = st.number_input("Day", min_value=0.0, max_value=200.0, value=defaults["Day"], step=1.0)
        values["Temperature_C"] = st.number_input(
            DISPLAY_LABELS["Temperature_C"],
            min_value=0.0,
            max_value=100.0,
            value=defaults["Temperature_C"],
            step=0.1,
            help="Use the daily average or representative compost temperature for the selected day.",
        )
        values["pH"] = st.number_input("pH", min_value=0.0, max_value=14.0, value=defaults["pH"], step=0.1)
        values["MC_(%)_Initial"] = st.number_input(DISPLAY_LABELS["MC_(%)_Initial"], min_value=0.0, max_value=100.0, value=defaults["MC_(%)_Initial"], step=0.1)
        values["C_N_Ratio_Initial"] = st.number_input(DISPLAY_LABELS["C_N_Ratio_Initial"], min_value=0.0, max_value=100.0, value=defaults["C_N_Ratio_Initial"], step=0.1)
        values["Aeration_rate(L/min)"] = st.number_input(DISPLAY_LABELS["Aeration_rate(L/min)"], min_value=0.0, max_value=100.0, value=defaults["Aeration_rate(L/min)"], step=0.1)
        values["Maximum_temperature_C"] = st.number_input(DISPLAY_LABELS["Maximum_temperature_C"], min_value=0.0, max_value=100.0, value=defaults["Maximum_temperature_C"], step=0.1)
        values["Composting_cycle_length_day"] = st.number_input(DISPLAY_LABELS["Composting_cycle_length_day"], min_value=1.0, max_value=365.0, value=defaults["Composting_cycle_length_day"], step=1.0)
        values["EC_(mS cm-1)_Initial"] = st.number_input(DISPLAY_LABELS["EC_(mS cm-1)_Initial"], min_value=0.0, max_value=50.0, value=defaults["EC_(mS cm-1)_Initial"], step=0.1)

    one_row = pd.DataFrame([{feature: values[feature] for feature in features}])
    one_pred, _ = predict_dataframe(one_row, bundle)
    prediction = one_pred[PREDICTION_COLUMN].iloc[0]
    uncertainty = estimate_prediction_uncertainty(one_row, bundle)
    if uncertainty is None:
        lower_bound = upper_bound = np.nan
    else:
        lower_bound = float(uncertainty["lower"][0])
        upper_bound = float(uncertainty["upper"][0])
    initial_ts_kg = wet_weight_kg * max(0.0, 1 - values["MC_(%)_Initial"] / 100)
    total_nh3_g = prediction * initial_ts_kg
    total_nh3_kg = total_nh3_g / 1000
    total_lower_g = lower_bound * initial_ts_kg if not pd.isna(lower_bound) else np.nan
    total_upper_g = upper_bound * initial_ts_kg if not pd.isna(upper_bound) else np.nan

    with output_panel:
        st.subheader("Prediction")
        result_cols = st.columns(3)
        result_cols[0].metric(
            "Daily NH3 emission",
            f"{prediction:.4f} g/kg Initial TS",
        )
        result_cols[1].metric("Initial TS", f"{initial_ts_kg:,.2f} kg")
        result_cols[2].metric("Total NH3", f"{total_nh3_g:,.2f} g")
        if not pd.isna(lower_bound):
            st.caption(
                f"Estimated range: {lower_bound:.4f} - {upper_bound:.4f} g/kg Initial TS "
                f"({total_lower_g:,.2f} - {total_upper_g:,.2f} g/day). "
                "This range is estimated from variation among ensemble trees, not a probability of exact correctness."
            )
        st.caption(
            f"Calculation: {wet_weight_kg:,.2f} kg wet compost x "
            f"(1 - {values['MC_(%)_Initial']:.1f}/100) = {initial_ts_kg:,.2f} kg Initial TS; "
            f"{prediction:.4f} g/kg Initial TS x {initial_ts_kg:,.2f} kg = {total_nh3_g:,.2f} g NH3/day "
            f"({total_nh3_kg:,.4f} kg/day)."
        )

        cycle_length = int(max(1, values["Composting_cycle_length_day"]))
        day_curve = pd.DataFrame(
            [
                {**{feature: values[feature] for feature in features}, "Day": day}
                for day in range(0, cycle_length + 1)
            ]
        )
        day_curve_pred, _ = predict_dataframe(day_curve, bundle)
        day_curve_pred["Initial_TS_kg"] = initial_ts_kg
        day_curve_pred["Total_NH3_g"] = (
            day_curve_pred[PREDICTION_COLUMN] * initial_ts_kg
        )
        day_chart = (
            alt.Chart(day_curve_pred)
            .mark_line(point=True, color="#2563eb")
            .encode(
                x=alt.X("Day:Q", title="Day"),
                y=alt.Y(f"{PREDICTION_COLUMN}:Q", title="Predicted NH3 emission"),
                tooltip=[
                    alt.Tooltip("Day:Q", format=".0f"),
                    alt.Tooltip(f"{PREDICTION_COLUMN}:Q", format=".4f"),
                ],
            )
            .properties(height=360)
        )
        marker = (
            alt.Chart(one_pred)
            .mark_circle(size=120, color="#dc2626")
            .encode(
                x=alt.X("Day:Q"),
                y=alt.Y(f"{PREDICTION_COLUMN}:Q"),
                tooltip=[
                    alt.Tooltip("Day:Q", format=".0f"),
                    alt.Tooltip(f"{PREDICTION_COLUMN}:Q", format=".4f"),
                ],
            )
        )
        st.altair_chart(day_chart + marker, width="stretch")
        one_pred["Initial_compost_wet_weight_kg"] = wet_weight_kg
        one_pred["Initial_TS_kg"] = initial_ts_kg
        one_pred["Total_NH3_g"] = total_nh3_g
        one_pred["Total_NH3_kg"] = total_nh3_kg
        one_pred["Estimated_lower_g/kg_TS"] = lower_bound
        one_pred["Estimated_upper_g/kg_TS"] = upper_bound
        st.dataframe(one_pred, width="stretch", hide_index=True)

with tab_file:
    left, right = st.columns([0.78, 0.22], gap="large")
    with right:
        source_mode = st.radio(
            "Dataset",
            ["Built-in 25-paper data", "Upload Excel or CSV"],
            index=0,
        )
        uploaded = None
        if source_mode == "Upload Excel or CSV":
            uploaded = st.file_uploader(
                "File",
                type=["xlsx", "xls", "csv"],
                label_visibility="collapsed",
            )

    if source_mode == "Built-in 25-paper data":
        input_df = default_data
    elif uploaded is not None and uploaded.name.lower().endswith(".csv"):
        input_df = pd.read_csv(uploaded)
    elif uploaded is not None:
        input_df = pd.read_excel(uploaded)
    else:
        input_df = pd.DataFrame()

    with left:
        if input_df.empty:
            st.info("Upload a file to run predictions.")
        else:
            predicted_df, missing_features = predict_dataframe(input_df, bundle)
            if missing_features:
                st.error("Missing required columns: " + ", ".join(missing_features))
                template = pd.DataFrame(columns=features + [target])
                st.download_button(
                    "Download template",
                    data=to_excel_bytes(template),
                    file_name="nh3_prediction_template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                scored = target in predicted_df.columns
                scores = compute_scores(predicted_df, target) if scored else None

                metric_cols = st.columns(4)
                metric_cols[0].metric("Rows", f"{len(predicted_df):,}")
                if scores and source_mode != "Built-in 25-paper data":
                    metric_cols[1].metric("Uploaded comparison R2", format_metric(scores["R2"]))
                    metric_cols[2].metric("MAE", format_metric(scores["MAE"]))
                    metric_cols[3].metric("RMSE", format_metric(scores["RMSE"]))
                elif scores:
                    metric_cols[1].metric(
                        "Random split R2",
                        validation_metric("random_row_split", "ExtraTrees", "R2"),
                    )
                    metric_cols[2].metric(
                        "Source split R2",
                        validation_metric("source_group_split", "ExtraTrees", "R2"),
                    )
                    metric_cols[3].metric(
                        "Source split RMSE",
                        validation_metric("source_group_split", "ExtraTrees", "RMSE"),
                    )
                    st.info(
                        "The built-in dataset was used to train the final model, so these cards show held-out validation metrics instead of in-sample fit."
                    )
                else:
                    metric_cols[1].metric("R2", "-")
                    metric_cols[2].metric("MAE", "-")
                    metric_cols[3].metric("RMSE", "-")

                if scored:
                    plot_df = predicted_df[[target, PREDICTION_COLUMN]].dropna()
                    scatter = (
                        alt.Chart(plot_df)
                        .mark_circle(size=42, opacity=0.55, color="#2563eb")
                        .encode(
                            x=alt.X(f"{target}:Q", title="Actual NH3"),
                            y=alt.Y(f"{PREDICTION_COLUMN}:Q", title="Predicted NH3"),
                            tooltip=[
                                alt.Tooltip(f"{target}:Q", format=".4f"),
                                alt.Tooltip(f"{PREDICTION_COLUMN}:Q", format=".4f"),
                            ],
                        )
                        .properties(height=330)
                    )
                    max_value = float(
                        plot_df[[target, PREDICTION_COLUMN]].max().max()
                    )
                    line_df = pd.DataFrame({"x": [0, max_value], "y": [0, max_value]})
                    identity = (
                        alt.Chart(line_df)
                        .mark_line(color="#111827", strokeDash=[5, 5])
                        .encode(x="x:Q", y="y:Q")
                    )
                    st.altair_chart(scatter + identity, width="stretch")

                st.dataframe(predicted_df, width="stretch", height=320)
                st.download_button(
                    "Download predictions",
                    data=to_excel_bytes(predicted_df),
                    file_name="nh3_dashboard_predictions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

with tab_model:
    left, right = st.columns([0.45, 0.55], gap="large")
    with left:
        if not metrics.empty:
            st.dataframe(metrics, width="stretch", hide_index=True)
    with right:
        if not importance.empty:
            importance_chart = (
                alt.Chart(importance)
                .mark_bar(color="#0f766e", cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
                .encode(
                    x=alt.X("importance:Q", title="Importance"),
                    y=alt.Y("feature:N", title=None, sort="-x"),
                    tooltip=[
                        "feature",
                        alt.Tooltip("importance:Q", format=".3f"),
                    ],
                )
                .properties(height=420)
            )
            st.altair_chart(importance_chart, width="stretch")
    if SHAP_SUMMARY_PATH.exists() and SHAP_BAR_PATH.exists():
        st.subheader("SHAP analysis")
        shap_left, shap_right = st.columns(2, gap="large")
        shap_left.image(str(SHAP_BAR_PATH), caption="Mean absolute SHAP values")
        shap_right.image(str(SHAP_SUMMARY_PATH), caption="SHAP summary plot")
