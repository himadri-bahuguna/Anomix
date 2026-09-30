import streamlit as st
import plotly.graph_objects as go

from components.charts import render_chart
from components.header import footer, page_header, section_title
from components.kpi_card import kpi_card
from components.status_badge import status_badge
from components.tables import render_table
from config.settings import PARAMETERS


def render(service) -> None:
    data = service.get_overview()
    page_header("Reliability & Predictive Monitoring", "Engineering health view across the converter population.")
    st.markdown("<div class='note'><b>SYSTEM STATUS</b>&nbsp;&nbsp; Operational / Demo data active. Backend integration pending.</div>", unsafe_allow_html=True)
    columns = st.columns(4)
    for column, label, value, detail in zip(columns, ["Components", "Parameters", "Burn-in checkpoints", "ML modules"], ["2,000", "8", "4", "2"], ["Demo population", "Monitored signals", "0h / 24h / 96h / 168h", "Module A + Module B"]):
        with column:
            kpi_card(label, value, detail)
    section_title("Health Overview")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("<div class='card'><div class='kpi-label'>Overall monitoring status</div><div class='kpi-value' style='color:#22C55E'>NORMAL</div><div class='kpi-detail'>No live hardware connection. Summary is illustrative demo data.</div></div>", unsafe_allow_html=True)
    with right:
        for name, detail, status in [("Module A", "Anomaly Detection", "Ready"), ("Module B", "168h Drift Prediction", "Ready"), ("Backend", "Demo Data", "Connected")]:
            st.markdown(f"<div style='padding:.48rem 0;border-bottom:1px solid #263244'><b>{name}</b><br><span class='status-muted'>{detail}</span></div>", unsafe_allow_html=True)
            status_badge(status)
    section_title("Parameter Snapshot")
    render_table(data["parameter_snapshot"], height=300)
    section_title("Recent / Model Status")
    status_columns = st.columns(3)
    for column, title, detail in zip(status_columns, ["Module A", "Module B", "Data"], ["Baseline anomaly detection", "168h drift prediction", "0h / 24h / 96h / 168h checkpoints"]):
        with column:
            st.markdown(f"<div class='card'><div class='kpi-label'>{title}</div><div style='margin-top:.6rem'>{detail}</div><div class='kpi-detail'>Demo status</div></div>", unsafe_allow_html=True)
    footer()
