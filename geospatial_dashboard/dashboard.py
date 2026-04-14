import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
import numpy as np
import pandas as pd

# ─── Config ──────────────────────────────────────────────────
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="EVolvAI Dashboard",
    page_icon="⚡",
    layout="wide"
)

# ─── Data Fetching ───────────────────────────────────────────
@st.cache_data(ttl=5)
def fetch_api(endpoint):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# ─── Header ──────────────────────────────────────────────────
st.title("⚡ EVolvAI — EV Infrastructure Dashboard")
st.caption("IEEE IES Hackathon | Physics-Constrained Generative AI for Equitable EV Planning")
st.divider()

# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Scenario Control")
    scenario = st.selectbox(
        "Select Counterfactual Scenario",
        options=["baseline", "winter_storm", "fleet_2x"],
        format_func=lambda x: {
            "baseline": "Baseline (Current)",
            "winter_storm": "Winter Storm (+1.8x demand)",
            "fleet_2x": "Fleet Electrification (2.5x demand)"
        }[x]
    )
    
    show_real = st.checkbox("Show real EV chargers (OpenChargeMap)", value=True)
    
    st.divider()
    st.markdown("**Color Legend**")
    st.markdown("🟢 Green = Node operating normally")
    st.markdown("🔴 Red = Transformer overloaded")
    st.markdown("⚪ Circle size = Charger count")
    st.markdown("🔵 Blue = Real world EV charger")
    
    st.divider()
    st.markdown("**About**")
    st.caption("This dashboard visualizes the IEEE 33-bus system mapped to Hyderabad, India.")

# ─── Get Data ────────────────────────────────────────────────
node_data = fetch_api(f"/api/nodes/{scenario}")
gini_data = fetch_api(f"/api/gini/{scenario}")

if "error" in node_data:
    st.error(f"Cannot connect to API at {API_URL}. Is it running? (uvicorn api:app)")
    st.stop()

nodes = node_data["nodes"]

# Fetch real chargers if enabled
real_chargers = []
if show_real:
    real_data = fetch_api("/api/real_chargers")
    if "chargers" in real_data:
        real_chargers = real_data["chargers"]
        st.sidebar.success(f"Showing {len(real_chargers)} real chargers")
    else:
        st.sidebar.warning(f"Could not load real chargers: {real_data.get('error', 'Unknown error')}")

# ─── Top Metrics Row ─────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    gini_val = gini_data.get("gini_index", "N/A")
    st.metric("Gini Accessibility Index", f"{gini_val:.3f}" if isinstance(gini_val, float) else gini_val)
with col2:
    overloaded = sum(1 for n in nodes if n["transformer_overload"])
    st.metric("Overloaded Transformers", f"{overloaded} / {len(nodes)}", f"{overloaded} at risk", delta_color="inverse")
with col3:
    total_chargers = sum(n["charger_count"] for n in nodes)
    st.metric("Total Chargers (Grid)", total_chargers)
with col4:
    zero_nodes = sum(1 for n in nodes if n["charger_count"] == 0)
    st.metric("Nodes w/o Chargers", zero_nodes, f"{zero_nodes} underserved", delta_color="inverse")

st.divider()

# ─── Map + Table Layout ──────────────────────────────────────
map_col, table_col = st.columns([2, 1])

with map_col:
    st.subheader("IEEE 33-Bus System — Hyderabad Grid Map")
    
    m = folium.Map(
        location=[np.mean([n["lat"] for n in nodes]), np.mean([n["lng"] for n in nodes])],
        zoom_start=12,
        tiles="CartoDB positron"
    )
    
    # 1. Add IEEE Nodes
    for node in nodes:
        color = "red" if node["transformer_overload"] else "green"
        folium.CircleMarker(
            location=[node["lat"], node["lng"]],
            radius=max(5, node["charger_count"] * 2.5),
            color=color,
            fill=True,
            fill_opacity=0.7,
            tooltip=f"Node {node['node_id']} | Chargers: {node['charger_count']}",
            popup=f"Node {node['node_id']}<br>Zone: {node['zone']}<br>Chargers: {node['charger_count']}"
        ).add_to(m)
        
        folium.Marker(
            location=[node["lat"], node["lng"]],
            icon=folium.DivIcon(html=f'<div style="font-size:9px;font-weight:bold">{node["node_id"]}</div>')
        ).add_to(m)

    # 2. Add Real Chargers (Blue)
    for charger in real_chargers:
        if charger["lat"] and charger["lng"]:
            folium.CircleMarker(
                location=[charger["lat"], charger["lng"]],
                radius=6,
                color="blue",
                fill=True,
                fill_opacity=0.8,
                tooltip=f"REAL CHARGER: {charger['name']}",
                popup=f"<b>{charger['name']}</b><br>{charger['address']}"
            ).add_to(m)
    
    # 3. RENDER MAP ONCE
    st_folium(m, width=700, height=500)

with table_col:
    st.subheader("Node Status Table")
    filter_opt = st.radio("Show", ["All nodes", "Overloaded only", "No chargers"])
    
    if filter_opt == "Overloaded only":
        display_nodes = [n for n in nodes if n["transformer_overload"]]
    elif filter_opt == "No chargers":
        display_nodes = [n for n in nodes if n["charger_count"] == 0]
    else:
        display_nodes = nodes
    
    rows = [{"Node": n["node_id"], "Zone": n["zone"], "Chargers": n["charger_count"], "Status": "🔴 Overloaded" if n["transformer_overload"] else "🟢 Normal"} for n in display_nodes]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, height=400)

# ─── Gini Chart ──────────────────────────────────────────────
st.divider()
st.subheader("Gini Score Distribution Across Nodes")
gini_df = pd.DataFrame([{"Node": f"N{n['node_id']}", "Gini Score": n["gini_score"]} for n in sorted(nodes, key=lambda x: x["gini_score"])])
st.bar_chart(gini_df.set_index("Node")["Gini Score"])

# ─── Footer ──────────────────────────────────────────────────
st.divider()
st.caption("EVolvAI | IEEE IES GenAI Hackathon | Dashboard by Krishna")
