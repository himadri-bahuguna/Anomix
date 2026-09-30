import streamlit as st

from components.sidebar import render_sidebar
from pages import component_analysis, dataset_pipeline, ml_performance, module_a, module_b, overview
from services.data_service import DataService
from styles.theme import inject_global_css

st.set_page_config(page_title="HMC 30W | Reliability Monitor", page_icon="H", layout="wide", initial_sidebar_state="expanded")
inject_global_css()
service = DataService()
page = render_sidebar()

routes = {
    "Overview": overview.render,
    "Component Analysis": component_analysis.render,
    "Module A": module_a.render,
    "Module B": module_b.render,
    "ML Performance": ml_performance.render,
    "Dataset & Pipeline": dataset_pipeline.render,
}
routes.get(page, overview.render)(service)
