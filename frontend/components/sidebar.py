import streamlit as st
from config.settings import VERSION

PAGES = ["Overview", "Component Analysis", "Module A", "Module B", "ML Performance", "Dataset & Pipeline"]


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("<div class='brand-mark'>HMC 30W</div><div class='brand-sub'>Reliability Monitor</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-rule'></div>", unsafe_allow_html=True)
        current = st.session_state.get("page", "Overview")
        for page in PAGES:
            if st.button(page, key=f"nav_{page}", width="stretch", type="primary" if page == current else "secondary"):
                st.session_state.page = page
                st.rerun()
        st.markdown("<div class='sidebar-footer'>SYSTEM STATUS<br><span class='status-dot status-green'></span><span class='status-text'>Demo Mode</span><br><br>HMC 30W<br>Reliability Monitoring<br>Version: " + VERSION + "</div>", unsafe_allow_html=True)
    return st.session_state.get("page", "Overview")
