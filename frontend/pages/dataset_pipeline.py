import streamlit as st

from components.header import footer, page_header, section_title
from components.kpi_card import kpi_card
from components.status_badge import status_badge


def render(service) -> None:
    data = service.get_dataset_status()
    page_header("Dataset & Pipeline", "Lineage, split structure, and integrity checks for the monitoring dataset.")
    columns = st.columns(4)
    for column, label, value in zip(columns, ["Dataset size", "Measurement rows", "Parameters", "Module B rows"], ["2,000 components", "8,000", "8", "16,000"]):
        with column:
            kpi_card(label, value, "Current project structure")
    section_title("Dataset Structure")
    st.markdown(f"<div class='card'>Checkpoints: <b>{' / '.join(str(point) + 'h' for point in data['checkpoints'])}</b><br>Train: <b>{data['train']:,}</b> components &nbsp; Validation: <b>{data['validation']:,}</b> &nbsp; Test: <b>{data['test']:,}</b></div>", unsafe_allow_html=True)
    section_title("Processing Pipeline")
    steps = ["Raw Measurements", "Quality Processing", "Component Split", "Module A", "Module B", "ML Evaluation", "Dashboard"]
    for index, step in enumerate(steps):
        st.markdown(f"<div class='pipeline-step'>{step}</div>", unsafe_allow_html=True)
        if index < len(steps) - 1:
            st.markdown("<div class='pipeline-arrow'>↓</div>", unsafe_allow_html=True)
    section_title("Data Integrity Checks")
    for check, result in data["integrity"].items():
        left, right = st.columns([3, 1])
        with left:
            st.markdown(f"<div style='padding:.5rem 0;border-bottom:1px solid #263244'>{check}</div>", unsafe_allow_html=True)
        with right:
            status_badge("pass" if result == "PASS" else "ready", result)
    footer()
