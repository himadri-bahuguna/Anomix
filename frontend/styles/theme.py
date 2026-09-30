import streamlit as st

COLORS = {
    "background": "#0B1220",
    "surface": "#111827",
    "card": "#151F2E",
    "border": "#263244",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "blue": "#3B82F6",
    "green": "#22C55E",
    "amber": "#F59E0B",
    "red": "#EF4444",
}


def inject_global_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Roboto+Mono:wght@500;600&display=swap');
        :root {{
            --bg: {COLORS['background']}; --surface: {COLORS['surface']}; --card: {COLORS['card']};
            --border: {COLORS['border']}; --text: {COLORS['text']}; --muted: {COLORS['muted']};
            --blue: {COLORS['blue']}; --green: {COLORS['green']}; --amber: {COLORS['amber']}; --red: {COLORS['red']};
        }}
        .stApp {{ background: var(--bg); color: var(--text); font-family: 'IBM Plex Sans', sans-serif; }}
        [data-testid='stHeader'] {{ background: rgba(11,18,32,.92); }}
        [data-testid='stSidebar'] {{ background: var(--surface); border-right: 1px solid var(--border); }}
        [data-testid='stSidebarNav'] {{ display: none; }}
        [data-testid='stSidebar'] > div:first-child {{ padding: 1.5rem 1rem; }}
        .block-container {{ max-width: 1440px; padding: 2.25rem 2.5rem 3rem; }}
        h1, h2, h3, h4, p {{ color: var(--text); }}
        h1 {{ letter-spacing: 0; }}
        .eyebrow {{ color: var(--blue); font-size: .72rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; margin-bottom: .35rem; }}
        .page-subtitle {{ color: var(--muted); font-size: .95rem; margin-top: -.35rem; margin-bottom: 1.75rem; }}
        .section-title {{ color: var(--text); font-size: 1.18rem; font-weight: 600; margin: 1.55rem 0 .7rem; }}
        .section-kicker {{ color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .1em; }}
        .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 6px; padding: 1.1rem 1.2rem; min-height: 100%; }}
        .kpi-label {{ color: var(--muted); font-size: .77rem; text-transform: uppercase; letter-spacing: .09em; }}
        .kpi-value {{ color: var(--text); font: 700 1.75rem 'Roboto Mono', monospace; margin-top: .5rem; }}
        .kpi-detail {{ color: var(--muted); font-size: .78rem; margin-top: .3rem; }}
        .status-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 7px; }}
        .status-green {{ background: var(--green); }} .status-blue {{ background: var(--blue); }}
        .status-amber {{ background: var(--amber); }} .status-red {{ background: var(--red); }}
        .status-text {{ color: var(--text); font-size: .88rem; font-weight: 600; }}
        .status-muted {{ color: var(--muted); font-size: .78rem; }}
        .brand-mark {{ color: var(--text); font: 700 1.25rem 'Roboto Mono', monospace; letter-spacing: .03em; }}
        .brand-sub {{ color: var(--muted); font-size: .78rem; margin-top: .2rem; }}
        .sidebar-rule {{ border-top: 1px solid var(--border); margin: 1.4rem 0; }}
        .pipeline-step {{ background: var(--card); border: 1px solid var(--border); padding: .75rem 1rem; text-align: center; color: var(--text); font-weight: 600; border-radius: 4px; }}
        .pipeline-arrow {{ color: var(--blue); text-align: center; font-size: 1.2rem; padding: .2rem 0; }}
        .note {{ border-left: 3px solid var(--blue); background: rgba(59,130,246,.08); color: var(--muted); padding: .75rem 1rem; font-size: .84rem; }}
        .sidebar-footer {{ color: var(--muted); font-size: .76rem; line-height: 1.6; padding-top: 2rem; }}
        div[data-testid='stMetric'] {{ background: var(--card); border: 1px solid var(--border); padding: .75rem 1rem; border-radius: 6px; }}
        div[data-testid='stMetricLabel'] p {{ color: var(--muted) !important; }}
        .stSelectbox label, .stMultiSelect label {{ color: var(--muted) !important; font-size: .8rem !important; }}
        .stDataFrame {{ border: 1px solid var(--border); }}
        footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def plotly_layout(title: str | None = None) -> dict:
    layout = {
        "paper_bgcolor": COLORS["card"], "plot_bgcolor": COLORS["card"],
        "font": {"family": "IBM Plex Sans, sans-serif", "color": COLORS["muted"]},
        "margin": {"l": 48, "r": 20, "t": 42 if title else 18, "b": 42},
        "xaxis": {"showgrid": False, "zeroline": False},
        "yaxis": {"showgrid": True, "gridcolor": "#263244", "zeroline": False},
        "legend": {"orientation": "h", "y": 1.1, "x": 0},
        "hoverlabel": {"bgcolor": COLORS["surface"]},
    }
    if title:
        layout["title"] = {"text": title, "font": {"size": 14, "color": COLORS["text"]}, "x": 0.02}
    return layout
