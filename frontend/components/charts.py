import plotly.graph_objects as go
import streamlit as st
from styles.theme import COLORS, plotly_layout


def render_chart(figure: go.Figure, height: int = 330) -> None:
    figure.update_layout(**plotly_layout(), height=height)
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def add_status_colors(figure: go.Figure) -> None:
    figure.update_traces(marker_color=[COLORS["green"], COLORS["amber"], COLORS["red"]])
