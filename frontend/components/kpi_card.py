import streamlit as st


def kpi_card(label: str, value: str, detail: str = "") -> None:
    st.markdown(
        f"<div class='card'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div><div class='kpi-detail'>{detail}</div></div>",
        unsafe_allow_html=True,
    )
