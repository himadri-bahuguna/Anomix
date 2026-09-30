import streamlit as st
import plotly.graph_objects as go

from components.charts import render_chart
from components.header import footer, page_header, section_title
from components.tables import render_table
from config.settings import PARAMETERS


def render(service) -> None:
    data = service.get_ml_metrics()
    metrics = data["metrics"]
    page_header("ML Performance", "Evaluation of predictive performance across monitored parameters.")
    section_title("B0 vs B1 Metric Comparison")
    display = metrics[["Parameter", "Baseline MAE", "B1 MAE", "B1 R²", "Improvement vs baseline (%)"]].copy()
    display["B1 RMSE"] = "Not supplied"
    display = display[["Parameter", "Baseline MAE", "B1 MAE", "B1 RMSE", "B1 R²", "Improvement vs baseline (%)"]]
    render_table(display, height=330)
    left, right = st.columns(2)
    with left:
        figure = go.Figure()
        figure.add_bar(name="Baseline MAE", x=metrics["Parameter"], y=metrics["Baseline MAE"], marker_color="#64748B")
        figure.add_bar(name="B1 MAE", x=metrics["Parameter"], y=metrics["B1 MAE"], marker_color="#3B82F6")
        figure.update_layout(title="MAE comparison", barmode="group", xaxis_title="Parameter", yaxis_title="MAE")
        render_chart(figure, height=390)
    with right:
        figure = go.Figure(go.Bar(x=metrics["Parameter"], y=metrics["B1 R²"], marker_color="#22C55E"))
        figure.update_layout(title="B1 R² by parameter", xaxis_title="Parameter", yaxis_title="R²", showlegend=False)
        render_chart(figure, height=390)
    section_title("Parameter Detail")
    selected = st.selectbox("Parameter selector", metrics["Key"].tolist(), format_func=lambda key: PARAMETERS[key])
    row = metrics[metrics["Key"] == selected].iloc[0]
    st.write(f"**{PARAMETERS[selected]}** | Test-set result")
    st.json({"MAE": round(float(row["B1 MAE"]), 6), "RMSE": "Not supplied", "R²": round(float(row["B1 R²"]), 6), "Improvement vs baseline (%)": round(float(row["Improvement vs baseline (%)"]), 6)})
    section_title("Methodology")
    st.markdown(f"<div class='card'>Training: <b>{data['training']:,} components</b><br>Validation: <b>{data['validation']:,} components</b><br>Test: <b>{data['test']:,} components</b><br>Split: <b>{data['split']}</b><br>Prediction horizon: <b>{data['horizon']}</b><br>Information available to model: <b>{data['features']}</b><br><br><span class='status-muted'>Test data was not used for model selection.</span></div>", unsafe_allow_html=True)
    footer()
