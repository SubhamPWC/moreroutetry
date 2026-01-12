
# app.py
import os
import math
import streamlit as st
import pandas as pd
from optimization import load_graph, compute_candidate_routes, recommend_route, summarize_routes
from map_utils import make_map
from multimodal import load_flights, load_trains, summarize_mode, recommend

st.set_page_config(page_title="Multimodal Route Optimization (India)", layout="wide")
st.title("🧭 Multimodal Route Optimization — Road · Train · Air")

STATIC_POINTS = {
    "Delhi": (28.6139, 77.2090), "Mumbai": (19.0760, 72.8777), "Kolkata": (22.5726, 88.3639),
    "Chennai": (13.0827, 80.2707), "Bengaluru": (12.9716, 77.5946), "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567), "Ahmedabad": (23.0225, 72.5714), "Jaipur": (26.9124, 75.7873),
    "Surat": (21.1702, 72.8311), "Lucknow": (26.8467, 80.9462), "Kanpur": (26.4499, 80.3319),
    "Nagpur": (21.1458, 79.0882), "Indore": (22.7196, 75.8577), "Bhopal": (23.2599, 77.4126),
    "Coimbatore": (11.0168, 76.9558), "Kochi": (9.9312, 76.2673), "Thiruvananthapuram": (8.5241, 76.9366),
    "Visakhapatnam": (17.6868, 83.2185), "Vijayawada": (16.5062, 80.6480), "Guwahati": (26.1158, 91.7086),
    "Siliguri": (26.7271, 88.3953), "Durgapur": (23.5204, 87.3119), "Asansol": (23.6838, 86.9536),
    "Jamshedpur": (22.8046, 86.2029), "Dhanbad": (23.7957, 86.4300), "Cuttack": (20.4625, 85.8828),
    "Bhubaneswar": (20.2961, 85.8245), "Puri": (19.8135, 85.8312), "Gaya": (24.7925, 85.0078),
    "Varanasi": (25.3176, 82.9739), "Noida": (28.5355, 77.3910), "Gurugram": (28.4595, 77.0266),
    "Nashik": (19.9975, 73.7898), "Vadodara": (22.3072, 73.1812), "Rajkot": (22.3039, 70.8022),
    "Chandigarh": (30.7333, 76.7794), "Amritsar": (31.6340, 74.8723), "Ludhiana": (30.9010, 75.8573),
    "Agra": (27.1767, 78.0081), "Prayagraj": (25.4358, 81.8463), "Aurangabad": (19.8762, 75.3433),
    "Panaji": (15.4909, 73.8278), "Mysuru": (12.2958, 76.6394), "Mangaluru": (12.9141, 74.8560),
    "Madurai": (9.9252, 78.1198), "Tiruchirappalli": (10.7905, 78.7047),
}

def haversine_km(a_lat, a_lon, b_lat, b_lon):
    R = 6371.0
    dlat = math.radians(b_lat - a_lat); dlon = math.radians(b_lon - a_lon)
    sa = math.sin(dlat/2.0)**2 + math.cos(math.radians(a_lat))*math.cos(math.radians(b_lat))*math.sin(dlon/2.0)**2
    return 2.0 * R * math.asin(math.sqrt(sa))

st.sidebar.header("Route Inputs")
city_names = sorted(list(STATIC_POINTS.keys()))
origin_name = st.sidebar.selectbox("From (Origin)", city_names, index=city_names.index("Delhi"))
dest_name   = st.sidebar.selectbox("To (Destination)",   city_names, index=city_names.index("Mumbai"))
origin = STATIC_POINTS[origin_name]; dest = STATIC_POINTS[dest_name]
if origin == dest: st.sidebar.warning("Origin and destination are the same. Select different points.")

st.sidebar.header("Primary objective")
objective = st.sidebar.selectbox("Objective", ["balanced","min_time","min_cost","min_emissions"], index=0)

st.sidebar.header("Road Preferences")
k_routes = st.sidebar.slider("Road routes (k)", min_value=3, max_value=25, value=10)
use_offline_demo = st.sidebar.checkbox("Use Offline Demo Graph (no internet)", value=False)
prefer_highways = st.sidebar.checkbox("Prefer highways", value=True)
avoid_bridges_flyovers = st.sidebar.checkbox("Avoid bridges / flyovers", value=False)

st.sidebar.header("Vehicle & Environment")
fuel_price_inr_per_l = st.sidebar.number_input("Fuel price (INR/L)", value=105.0, min_value=50.0, max_value=200.0)
fuel_consumption_l_per_100km = st.sidebar.number_input("Fuel consumption (L/100 km)", value=6.5, min_value=3.0, max_value=20.0)
emission_factor_g_co2_per_km = st.sidebar.number_input("Emission factor (g CO₂/km)", value=170.0, min_value=0.0, max_value=400.0)

st.sidebar.header("Corridor (OSM Fetch)")
corridor_width_km = st.sidebar.slider("Corridor width (km)", min_value=10, max_value=150, value=80, step=5)

st.sidebar.header("ML Speed Model (Optional)")
speed_model_path = st.sidebar.text_input("Speed model file (models/speed_model.pkl)", value="models/speed_model.pkl")

road_tab, train_tab, air_tab, compare_tab = st.tabs(["🚗 Road","🚆 Train","✈️ Air","📊 Compare"])

with st.spinner("Fetching OSM corridor and computing road routes..."):
    G, source = load_graph(origin, dest, use_offline_demo=use_offline_demo, corridor_width_km=corridor_width_km)
    road_routes = compute_candidate_routes((G, source), origin, dest, k=k_routes,
                                           fuel_price_inr_per_l=fuel_price_inr_per_l,
                                           fuel_consumption_l_per_100km=fuel_consumption_l_per_100km,
                                           emission_factor_g_co2_per_km=emission_factor_g_co2_per_km,
                                           prefer_highways=prefer_highways,
                                           avoid_bridges_flyovers=avoid_bridges_flyovers,
                                           speed_model_path=speed_model_path)
    df_road = summarize_routes(road_routes)

with road_tab:
    st.subheader(f"🗺️ Map: Routes from {origin_name} to {dest_name}")
    if road_routes:
        rec_idx_road, tag_road = recommend_route(df_road, objective)
        st.write("**Data source:**", road_routes[0].get('data_source', source))
        fmap = make_map(road_routes, rec_idx_road, origin, dest)
        st.components.v1.html(fmap.get_root().render(), height=600)
        st.write(f"**Road routes (k={k_routes})**")
        st.dataframe(df_road[["route_id","optimized","total_distance_km","total_time_min","total_cost_inr","total_emissions_kg","highway_share_pct","has_flyover","dominance_rank"]])
    else:
        st.info("No road routes found. Try a wider corridor or increase k.")

with train_tab:
    st.subheader("🚆 Trains")
    df_tr = load_trains(origin_name, dest_name)
    if len(df_tr)==0:
        st.info(f"Add rows to data/trains.csv for {origin_name}→{dest_name} to see train line‑ups.")
    else:
        dist_km = haversine_km(origin[0], origin[1], dest[0], dest[1])
        df_tr_sum = summarize_mode(df_tr,'train', dist_km)
        rec_idx_tr = recommend(df_tr_sum, objective)
        st.dataframe(df_tr_sum[["label","from_station","depart_time","to_station","arrive_time","duration_h","price_3A_inr","price_2A_inr","price_1A_inr","days","total_time_min","total_cost_inr","emissions_kg"]])
        if rec_idx_tr>=0:
            st.success(f"Recommended train ({objective}): {df_tr_sum.loc[rec_idx_tr,'label']}")

with air_tab:
    st.subheader("✈️ Flights")
    df_fl = load_flights(origin_name, dest_name)
    if len(df_fl)==0:
        st.info(f"Add rows to data/flights.csv for {origin_name}→{dest_name} to see flight line‑ups.")
    else:
        dist_km = haversine_km(origin[0], origin[1], dest[0], dest[1])
        df_fl_sum = summarize_mode(df_fl,'air', dist_km)
        rec_idx_fl = recommend(df_fl_sum, objective)
        st.dataframe(df_fl_sum[["label","depart_time","arrive_time","duration_h","typical_price_inr","total_time_min","total_cost_inr","emissions_kg","airline"]])
        if rec_idx_fl>=0:
            st.success(f"Recommended flight ({objective}): {df_fl_sum.loc[rec_idx_fl,'label']}")

with compare_tab:
    st.subheader("📊 Mode comparison")
    rows = []
    if road_routes and len(df_road)>0:
        rows.append({"mode":"Road","time_min":float(df_road['total_time_min'].min()),"cost_inr":float(df_road['total_cost_inr'].min()),"emissions_kg":float(df_road['total_emissions_kg'].min())})
    df_tr = load_trains(origin_name, dest_name)
    if len(df_tr)>0:
        dist_km = haversine_km(origin[0], origin[1], dest[0], dest[1])
        df_tr_sum = summarize_mode(df_tr,'train', dist_km)
        rows.append({"mode":"Train","time_min":float(df_tr_sum['total_time_min'].min()),"cost_inr":float(df_tr_sum['total_cost_inr'].min()),"emissions_kg":float(df_tr_sum['emissions_kg'].min())})
    df_fl = load_flights(origin_name, dest_name)
    if len(df_fl)>0:
        dist_km = haversine_km(origin[0], origin[1], dest[0], dest[1])
        df_fl_sum = summarize_mode(df_fl,'air', dist_km)
        rows.append({"mode":"Air","time_min":float(df_fl_sum['total_time_min'].min()),"cost_inr":float(df_fl_sum['total_cost_inr'].min()),"emissions_kg":float(df_fl_sum['emissions_kg'].min())})
    if rows:
        df_modes = pd.DataFrame(rows); st.dataframe(df_modes)
        X = df_modes[['time_min','cost_inr','emissions_kg']].values
        mins = X.min(axis=0); maxs = X.max(axis=0); denom = (maxs-mins); denom[denom==0]=1.0
        scores = ((X - mins)/denom).sum(axis=1); best_idx = scores.argmin()
        st.success(f"Overall recommended mode ({objective}): {df_modes.loc[best_idx,'mode']}")
    else:
        st.info("No modes to compare.")
