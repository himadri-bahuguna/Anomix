# HMC 30W DC-DC Converter Dashboard

A modular Streamlit dashboard for converter reliability monitoring, burn-in analysis, anomaly detection, and 168h prediction review.

## Features

- Dark industrial engineering dashboard with custom sidebar navigation
- Overview, component analysis, Module A, Module B, ML performance, and dataset pipeline views
- Plotly charts and compact engineering tables
- Real backend connectivity when the existing FastAPI service is running
- Graceful fallback to the current demo dataset if the backend is unavailable
- Existing backend result files and ML screening outputs remain the source of truth

## Install

From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Start the backend API:

```bash
uvicorn backend.backend.app:app --host 127.0.0.1 --port 8000
```

Then launch the dashboard:

```bash
streamlit run frontend/app.py
```

The dashboard starts on the Overview page and uses the existing backend endpoints (summary, component, trajectory, and prediction data) when available.

## Structure

- `frontend/app.py` - page configuration and routing only
- `frontend/services/data_service.py` - UI facade that calls the backend API or falls back to local demo data
- `frontend/data/mock_data.py` - legacy demo dataset retained as a fallback
- `frontend/components/` - reusable cards, charts, tables, header, and sidebar
- `frontend/pages/` - individual dashboard views
- `backend/backend/app.py` - FastAPI API exposing the existing project results and burn-in data
- `backend/results/` - generated ML screening and summary files used by the backend

## Current integration status

The frontend is wired to the real backend where supported, especially for overview data, component detail, review statuses, and burn-in trajectories. The Module B evaluation table remains clearly labeled as existing evaluation context and is not treated as live hardware telemetry.
