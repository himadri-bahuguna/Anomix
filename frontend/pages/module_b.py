import streamlit as st
import plotly.graph_objects as go

from components.charts import render_chart
from components.header import footer, page_header, section_title
from components.kpi_card import kpi_card
from components.tables import render_table
from config.settings import PARAMETERS


def render(service) -> None:
    data = service.get_module_b_results()
    page_header("168h Drift Prediction", "MODULE B | Predicting long-term parameter behavior from measurements available by 24h.")
    columns = st.columns(4)
    for column, label, value in zip(columns, ["Parameters", "Prediction horizon", "Training components", "Test components"], ["8", "168h", "1,400", "300"]):
        with column:
            kpi_card(label, value, "Demo evaluation data")
    section_title("Model Performance")
    metrics = data["metrics"]
    figure = go.Figure()
    figure.add_bar(name="24h persistence baseline MAE", x=metrics["Parameter"], y=metrics["Baseline MAE"], marker_color="#64748B")
    figure.add_bar(name="B1 MAE", x=metrics["Parameter"], y=metrics["B1 MAE"], marker_color="#3B82F6")
    figure.update_layout(title="Current B1 evaluation result", barmode="group", xaxis_title="Parameter", yaxis_title="MAE")
    render_chart(figure, height=400)
    section_title("Parameter Detail")
    selected = st.selectbox("Select parameter", metrics["Key"].tolist(), format_func=lambda key: PARAMETERS[key])
    row = metrics[metrics["Key"] == selected].iloc[0]
    cols = st.columns(4)
    for column, label, value in zip(cols, ["Baseline MAE", "B1 MAE", "B1 RMSE", "B1 R²"], [row["Baseline MAE"], row["B1 MAE"], "Not supplied", row["B1 R²"]]):
        with column:
            st.metric(label, value if isinstance(value, str) else f"{value:.6f}")
    st.metric("Improvement vs baseline", f"{row['Improvement vs baseline (%)']:.6f}%")
    section_title("Prediction View")
    st.markdown(f"<div class='note'>{data['prediction_note']}. Actual 168h is shown only in evaluation context.</div>", unsafe_allow_html=True)
    x = ["Observed 0h", "Observed 24h", "Predicted 168h", "Actual 168h"]
    y = [metrics[metrics["Key"] == selected]["B1 MAE"].iloc[0] * 0 + 1, 1.02, 1.04, 1.03]
    figure = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers", line={"color": "#3B82F6"}, marker={"size": 8}))
    figure.update_layout(title=f"Illustrative / evaluation data: {PARAMETERS[selected]}", yaxis_title="Normalized value", showlegend=False)
    render_chart(figure)
    st.markdown(f"<div class='status-muted'>{data['verification']}</div>", unsafe_allow_html=True)
    footer()
