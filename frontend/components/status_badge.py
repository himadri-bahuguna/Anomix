import streamlit as st

COLORS = {"normal": "green", "ready": "blue", "warning": "amber", "anomaly": "red", "pass": "green"}


def status_badge(status: str, detail: str | None = None) -> None:
    tone = COLORS.get(status.lower(), "blue")
    detail_html = f"<span class='status-muted'>{detail}</span>" if detail else ""
    st.markdown(
        f"<span class='status-dot status-{tone}'></span><span class='status-text'>{status}</span> {detail_html}",
        unsafe_allow_html=True,
    )
