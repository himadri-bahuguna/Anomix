import streamlit as st


def page_header(title: str, subtitle: str, eyebrow: str = "HMC 30W DC-DC CONVERTER") -> None:
    st.markdown(f"<div class='eyebrow'>{eyebrow}</div>", unsafe_allow_html=True)
    st.title(title)
    st.markdown(f"<div class='page-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def section_title(title: str, kicker: str | None = None) -> None:
    if kicker:
        st.markdown(f"<div class='section-kicker'>{kicker}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)


def footer() -> None:
    st.markdown(
        "<div style='border-top:1px solid #263244;margin-top:2.5rem;padding-top:1rem;color:#64748B;font-size:.75rem;'>HMC 30W DC-DC Converter | Reliability & Predictive Monitoring</div>",
        unsafe_allow_html=True,
    )
