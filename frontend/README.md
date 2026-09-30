# HMC 30W DC-DC Converter Dashboard

A modular Streamlit MVP for converter reliability monitoring, burn-in analysis, anomaly detection, 168h drift prediction, and ML evaluation.

## Features

- Dark industrial engineering dashboard with custom sidebar navigation
- Overview, component analysis, Module A, Module B, ML performance, and dataset pipeline views
- Plotly charts and compact engineering tables
- Deterministic mock/demo data with visible Demo Mode labeling
- Backend-ready `DataService` facade so API adapters can replace mock data without rewriting UI pages
- Exact supplied B0/B1 evaluation metrics, including the verified matched-prediction note

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application starts on the Overview page and does not require a backend, login, or uploaded files.

## Structure

- `app.py` - page configuration and routing only
- `data/mock_data.py` - all demo data and evaluation values
- `services/data_service.py` - UI-facing data contract for future API integration
- `components/` - reusable cards, charts, tables, header, and sidebar
- `pages/` - individual dashboard views
- `styles/theme.py` - centralized CSS and Plotly theme

## Demo mode and backend integration

The sidebar visibly reports Demo Mode. Mock values are illustrative and are not live hardware measurements. Replace methods in `services/data_service.py` with adapters for endpoints such as `/api/overview`, `/api/components/{component_id}`, and `/api/ml/metrics`; page modules should not need to change.

The Module B view preserves the current evaluation context: B0 and B1 contain 4,800 matched test predictions with zero target mismatches and zero prediction differences. The dashboard does not infer why that occurred.
