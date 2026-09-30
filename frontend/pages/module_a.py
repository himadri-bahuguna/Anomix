import streamlit as st
import plotly.graph_objects as go

from components.charts import render_chart
from components.header import footer, page_header, section_title
from components.kpi_card import kpi_card
from components.tables import render_table
from styles.theme import COLORS


def render(service) -> None:
    data = service.get_module_a_results()
    page_header("Burn-in Anomaly Detection", "MODULE A | Component-level screening across burn-in measurements.")
    st.markdown("<div class='note'><b>DEMO / BACKEND PENDING</b>&nbsp;&nbsp; Mock anomaly scores are shown until the Module A backend is integrated.</div>", unsafe_allow_html=True)
    columns = st.columns(4)
    for column, label, value in zip(columns, ["Components monitored", "Anomalies detected", "Normal", "Flagged"], ["2,000", "108", "1,510", "490"]):
        with column:
            kpi_card(label, value, "Demo evaluation data")
    section_title("Anomaly Distribution")
    left, right = st.columns(2)
    with left:
        distribution = data["distribution"]
        figure = go.Figure(go.Bar(x=distribution.index, y=distribution.values, marker_color=[COLORS["green"], COLORS["amber"], COLORS["red"]], text=distribution.values, textposition="outside"))
        figure.update_layout(title="Component status", xaxis_title="Status", yaxis_title="Components", showlegend=False)
        render_chart(figure)
    with right:
        figure = go.Figure(go.Histogram(x=data["scores"], nbinsx=18, marker_color=COLORS["blue"]))
        figure.update_layout(title="Anomaly score distribution", xaxis_title="Anomaly score", yaxis_title="Components", showlegend=False)
        render_chart(figure)
    section_title("Component Flags")
    selected = st.selectbox("Status filter", ["All", "Normal", "Warning", "Anomaly"])
    flags = data["flags"] if selected == "All" else data["flags"][data["flags"]["Status"] == selected]
    render_table(flags, height=350)
    footer()
