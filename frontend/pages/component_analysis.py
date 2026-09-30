import streamlit as st
import plotly.graph_objects as go

from components.charts import render_chart
from components.header import footer, page_header, section_title
from components.kpi_card import kpi_card
from components.tables import render_table
from config.settings import PARAMETERS


def render(service) -> None:
    page_header("Component Analysis", "Inspect burn-in behavior and illustrative 168h prediction context for one component.")
    component_id = st.selectbox("Component selector", service.get_component_list(), index=0)
    data = service.get_component_data(component_id)
    meta = st.columns(4)
    for column, label, value in zip(meta, ["Component ID", "Lot ID", "Split", "Current status"], [data["component_id"], data["lot_id"], data["split"].title(), data["status"]]):
        with column:
            kpi_card(label, value, "Demo data")
    section_title("Burn-in Profile")
    parameter = st.selectbox("Parameter", list(PARAMETERS), format_func=lambda key: PARAMETERS[key])
    profile = data["profile"]
    figure = go.Figure(go.Scatter(x=profile["Time (hours)"], y=profile[parameter], mode="lines+markers", line={"color": "#3B82F6", "width": 2}, marker={"color": "#22C55E", "size": 7}, name=PARAMETERS[parameter]))
    figure.update_layout(title=f"{PARAMETERS[parameter]} across burn-in checkpoints", xaxis_title="Time (hours)", yaxis_title=PARAMETERS[parameter])
    render_chart(figure)
    section_title("Environment")
    env_columns = st.columns(5)
    for column, (label, value) in zip(env_columns, data["environment"].items()):
        with column:
            st.metric(label, f"{value:.2f}")
    section_title("Predicted 168h Value")
    st.markdown("<div class='note'>Demo data. Prediction values are illustrative evaluation context, not live hardware measurements.</div>", unsafe_allow_html=True)
    prediction = data["prediction"]
    prediction_columns = st.columns(4)
    for column, label, value in zip(prediction_columns, ["24h observed", "Predicted 168h", "Ground-truth 168h", "Prediction error"], [prediction["observed_24h"], prediction["predicted_168h"], prediction["actual_168h"], prediction["error"]]):
        with column:
            st.metric(label, f"{value:.3f}")
    footer()
