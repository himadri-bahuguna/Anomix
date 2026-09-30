import pandas as pd
import streamlit as st


def render_table(data: pd.DataFrame, height: int = 260) -> None:
    st.dataframe(data, width="stretch", hide_index=True, height=height)
