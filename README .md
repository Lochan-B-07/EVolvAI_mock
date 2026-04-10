# EVolvAI — Geospatial Dashboard Module
**Krishna's Module | IEEE IES GenAI Hackathon**

---

## What This Module Does

This module is the **Geospatial UI & Spatial Equity Layer** of the EVolvAI system. It:

- Maps all 33 IEEE 33-bus system nodes to real GPS coordinates in Hyderabad, India
- Calculates the **Gini-based Accessibility Index** to quantify spatial inequity in EV charger distribution
- Serves a **REST API** (FastAPI) that delivers node data, overload status, and Gini scores per scenario
- Renders an **interactive Streamlit dashboard** with a Folium map showing counterfactual scenario switching

---

## Files in This Module

```
geospatial_dashboard/
├── mock_data.json        ← 33 IEEE bus nodes with GPS coords, charger counts, Gini scores
├── gini.py               ← Gini coefficient calculator
├── api.py                ← FastAPI REST backend (3 endpoints)
├── dashboard.py          ← Streamlit frontend with Folium map
├── requirements.txt      ← Python dependencies
└── README.md             ← This file
```

---

## How to Run

### Step 1 — Activate virtual environment
```bash
source ~/vscode/.venv/bin/activate
```

### Step 2 — Install dependencies (first time only)
```bash
pip install fastapi uvicorn streamlit streamlit-folium folium requests numpy pandas
```

### Step 3 — Navigate to this folder
```bash
cd ~/vscode/EVolvAI/geospatial_dashboard
```

### Step 4 — Start the API (Terminal 1)
```bash
uvicorn api:app --reload
```
API runs at: `http://localhost:8000`
Auto docs at: `http://localhost:8000/docs`

### Step 5 — Start the Dashboard (Terminal 2, new terminal)
```bash
source ~/vscode/.venv/bin/activate
cd ~/vscode/EVolvAI/geospatial_dashboard
streamlit run dashboard.py
```
Dashboard runs at: `http://localhost:8501`

---

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/nodes` | All 33 nodes — baseline scenario |
| GET | `/api/nodes/{scenario}` | Nodes adjusted for a scenario |
| GET | `/api/gini` | Gini index — baseline |
| GET | `/api/gini/{scenario}` | Gini index for a scenario |
| GET | `/api/scenarios` | List of all available scenarios |

### Available Scenarios

| Key | Label | Demand Multiplier |
|-----|-------|-------------------|
| `baseline` | Baseline (Current) | 1.0x |
| `winter_storm` | Winter Storm | 1.8x |
| `fleet_2x` | Fleet Electrification | 2.5x |

---

## What the Dashboard Shows

- **Gini Accessibility Index** — 0 = equal charger access, 1 = all chargers in one area
- **Overloaded Transformers** — nodes where grid capacity is exceeded under the selected scenario
- **Interactive Folium Map** — green dots = normal nodes, red dots = overloaded transformers, dot size = charger count
- **Node Status Table** — filterable table of all 33 nodes
- **Gini Score Bar Chart** — distribution of accessibility inequality across all nodes

---

## How This Connects to the Team

| Teammate | Module | Handoff to This Module |
|----------|--------|------------------------|
| Sujay | GCD-VAE demand model | Outputs `[24, 33]` demand tensor → replaces mock Gini scores |
| Akshay | Physics penalty engine | Outputs overload status per node → replaces `transformer_overload` in API |
| Lochan | CVaR optimizer | Outputs optimal charger placements → replaces `charger_count` in API |

### How to swap in real data (async handoff)
When teammates finish their modules, only `api.py` needs to change — specifically the `MOCK_NODES` list and the `apply_scenario()` function. `dashboard.py` never needs to change.

---

## Gini Index — How It Works

The Gini coefficient used here measures **inequality in EV charger accessibility**:

- Score of **0.0** = every zone has equal charger access
- Score of **1.0** = all chargers are in one zone, everyone else has nothing
- Current baseline score: **0.482** (moderate inequality)

Formula applied per node:
```
G = (2 * sum(i * x_i) - (n+1) * sum(x_i)) / (n * sum(x_i))
```
Where `x_i` is the charger count at node `i`, sorted in ascending order.

---

## Author
Krishna — Geospatial Data, UI & Maps  

