# EVolvAI — Module: Geospatial Dashboard + REST API

## What this module does
- Maps all 33 IEEE 33-bus nodes to real GPS coordinates in Hyderabad
- REST API serving node data, scenario adjustments, and Gini scores
- Interactive Streamlit dashboard with Folium map

## How to run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Start the API (Terminal 1)
uvicorn api:app --reload

### 3. Start the Dashboard (Terminal 2)
streamlit run dashboard.py

### 4. Open in browser
- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/nodes | All 33 nodes (baseline) |
| GET | /api/nodes/{scenario} | Nodes for scenario |
| GET | /api/gini | Gini score (baseline) |
| GET | /api/gini/{scenario} | Gini score for scenario |
| GET | /api/scenarios | Available scenarios |

## Scenarios
- baseline — current state
- winter_storm — 1.8x demand surge
- fleet_2x — 2.5x electrification multiplier