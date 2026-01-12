
import os
import streamlit as st
import pandas as pd
from optimization import (
    load_graph,
    compute_candidate_routes,
    recommend_route,
    summarize_routes,
)
from map_utils import make_map

st.set_page_config(page_title="Road Route Optimization (India)", layout="wide")
st.title("🚗 Road Route Optimization (India) — Distance · Time · Cost · Emissions")
st.caption("Not using Google Maps. Uses OpenStreetMap via OSMnx when online; includes an offline demo graph.")

# --- Static origin/destination choices (lat, lon) ---
STATIC_POINTS = {
    "Kolkata": (22.5726, 88.3639),
    "Bhubaneswar": (20.2961, 85.8245),
    "Ranchi": (23.3441, 85.3096),
    "Patna": (25.5941, 85.1376),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Surat": (21.1702, 72.8311),
    "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319),
    "Nagpur": (21.1458, 79.0882),
    "Indore": (22.7196, 75.8577),
    "Bhopal": (23.2599, 77.4126),
    "Coimbatore": (11.0168, 76.9558),
    "Kochi": (9.9312, 76.2673),
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Visakhapatnam": (17.6868, 83.2185),
    "Vijayawada": (16.5062, 80.6480),
    "Guwahati": (26.1158, 91.7086),
    "Siliguri": (26.7271, 88.3953),
    "Durgapur": (23.5204, 87.3119),
    "Asansol": (23.6838, 86.9536),
    "Jamshedpur": (22.8046, 86.2029),
    "Dhanbad": (23.7957, 86.4300),
    "Cuttack": (20.4625, 85.8828),
    "Puri": (19.8135, 85.8312),
    "Gaya": (24.7925, 85.0078),
    "Varanasi": (25.3176, 82.9739),
}

# Sidebar controls
st.sidebar.header("Route Inputs")
city_names = sorted(list(STATIC_POINTS.keys()))
origin_default = city_names.index("Kolkata") if "Kolkata" in city_names else 0
dest_default = city_names.index("Bhubaneswar") if "Bhubaneswar" in city_names else min(1, len(city_names)-1)
origin_name = st.sidebar.selectbox("From (Origin)", city_names, index=origin_default)
dest_name = st.sidebar.selectbox("To (Destination)", city_names, index=dest_default)
origin = STATIC_POINTS[origin_name]
dest = STATIC_POINTS[dest_name]
if origin == dest:
    st.sidebar.warning("Origin and destination are the same. Select different points.")

st.sidebar.header("Optimization Preferences")
objective = st.sidebar.selectbox(
    "Primary objective",
    ["balanced", "min_distance", "min_time", "min_cost", "min_emissions", "pareto"],
    index=0,
    help="Choose how to rank routes. 'pareto' shows non-dominated routes across all metrics."
)
k_routes = st.sidebar.slider("Number of route alternatives (k)", min_value=3, max_value=25, value=10)
use_offline_demo = st.sidebar.checkbox(
    "Use Offline Demo Graph (no internet)", value=False,
    help="If enabled, uses a small synthetic road graph. For real road names and more details, disable and ensure internet access to fetch OpenStreetMap data."
)

st.sidebar.header("Vehicle & Environment")
fuel_price_inr_per_l = st.sidebar.number_input("Fuel price (INR/L)", value=105.0, min_value=50.0, max_value=200.0)
fuel_consumption_l_per_100km = st.sidebar.number_input("Fuel consumption (L/100 km)", value=6.5, min_value=3.0, max_value=20.0)
emission_factor_g_co2_per_km = st.sidebar.number_input("Emission factor (g CO₂/km)", value=170.0, min_value=0.0, max_value=400.0)
prefer_highways = st.sidebar.checkbox("Prefer highways", value=True)
avoid_bridges_flyovers = st.sidebar.checkbox("Avoid bridges / flyovers", value=False)

st.sidebar.header("ML Speed Model (Optional)")
speed_model_path = st.sidebar.text_input(
    "Speed model file (models/speed_model.pkl)", value="models/speed_model.pkl",
    help="If present, a ML model adjusts segment speeds. Otherwise default speed per road type is used."
)

# Run optimization
with st.spinner("Loading road network and computing candidate routes..."):
    G = load_graph(origin, dest, use_offline_demo=use_offline_demo)
    routes = compute_candidate_routes(
        G,
        origin,
        dest,
        k=k_routes,
        fuel_price_inr_per_l=fuel_price_inr_per_l,
        fuel_consumption_l_per_100km=fuel_consumption_l_per_100km,
        emission_factor_g_co2_per_km=emission_factor_g_co2_per_km,
        prefer_highways=prefer_highways,
        avoid_bridges_flyovers=avoid_bridges_flyovers,
        speed_model_path=speed_model_path,
    )
    df_routes = summarize_routes(routes)
    recommended_idx, decision_tag = recommend_route(df_routes, objective)

# Show filters and outputs
col1, col2 = st.columns([0.55, 0.45])
with col1:
    st.subheader("🗺️ Map: Routes from {} to {}".format(origin_name, dest_name))
    fmap = make_map(routes, recommended_idx, origin, dest)
    st.components.v1.html(fmap.get_root().render(), height=650)

with col2:
    st.subheader("📋 Route Alternatives")
    st.write("**Decision tag:**", decision_tag)
    if df_routes is not None and len(df_routes) > 0:
        st.dataframe(
            df_routes[[
                "route_id",
                "optimized",
                "total_distance_km",
                "total_time_min",
                "total_cost_inr",
                "total_emissions_kg",
                "num_segments",
                "highway_share_pct",
                "has_flyover",
                "dominance_rank",
            ]].style.highlight_max(subset=["total_distance_km", "total_time_min", "total_cost_inr", "total_emissions_kg"], color="#ffe0e0")
        )

        csv_data = df_routes.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download routes CSV", csv_data, file_name="routes_{}_to_{}.csv".format(origin_name, dest_name), mime="text/csv")
    else:
        st.info("No routes found for the selected inputs. Try increasing k or changing preferences.")

# Show detailed segment table for the recommended route
st.subheader("🔎 Recommended Route Details")
if routes and len(routes) > 0:
    rec_route = routes[recommended_idx]
    seg_df = pd.DataFrame(rec_route["segments"])  # list of dict per edge
    if not seg_df.empty:
        st.dataframe(seg_df[["name", "highway", "length_m", "maxspeed_kph", "is_flyover", "travel_time_min"]])
    else:
        st.info("No segment details available.")
else:
    st.info("No recommended route.")

# Footer info
with st.expander("ℹ️ Notes"):
    st.markdown("""
    - **Data source**: When online, roads & names are fetched from OpenStreetMap via OSMnx. In offline demo, a small synthetic graph is used for illustration.
    - **Optimization**: Multi-criteria (distance, time, cost, emissions). Recommended route is chosen by selected objective, or balanced by normalized scores.
    - **ML**: Optional scikit-learn model can adjust segment speeds using road attributes if you provide `models/speed_model.pkl`.
    - **Flyover detection**: Based on OSM tags like `bridge=yes` or segment name containing 'Flyover'.
    """)
