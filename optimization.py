
# optimization.py
import os
import sys
from typing import Dict, Tuple, List, Optional
import networkx as nx
import numpy as np
import pandas as pd

# Try OSMnx; keep app working if import fails
try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
    # Conservative, so we bail out quickly if Overpass/HTTP is slow
    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.timeout = 25  # seconds
except Exception:
    OSMNX_AVAILABLE = False

from shapely.geometry import LineString, Polygon
from shapely.affinity import scale as shp_scale

DEFAULT_SPEED_KPH_BY_HIGHWAY = {
    "motorway": 90, "trunk": 80, "primary": 60, "secondary": 50, "tertiary": 40,
    "unclassified": 35, "residential": 30, "service": 25,
    "motorway_link": 60, "trunk_link": 50, "primary_link": 45,
    "secondary_link": 35, "tertiary_link": 30,
}
EPS = 1e-6  # strictly positive lower bound for edge weights

# ---------- Corridor builder (correct lon/lat + metric-like buffer) ----------
def _build_corridor_polygon(
    origin: Tuple[float, float], dest: Tuple[float, float], width_km: float = 80.0
) -> Polygon:
    """
    Build a buffered corridor polygon around the straight line between origin and dest.
    Inputs are (lat, lon). Shapely expects (lon, lat). To approximate metric buffering,
    scale x by cos(mean_lat), buffer in degrees, then unscale.
    """
    lat1, lon1 = origin
    lat2, lon2 = dest
    line = LineString([(lon1, lat1), (lon2, lat2)])  # x=lon, y=lat
    mean_lat = (lat1 + lat2) / 2.0
    xfact = float(np.cos(np.radians(mean_lat)))
    line_scaled = shp_scale(line, xfact=xfact, yfact=1.0, origin=(0.0, 0.0))
    buf_deg = float(width_km) / 111.0
    corridor_scaled = line_scaled.buffer(buf_deg)
    corridor = shp_scale(corridor_scaled, xfact=1.0 / xfact, yfact=1.0, origin=(0.0, 0.0))
    return corridor

def _build_nh48_demo_graph(origin: Tuple[float, float], dest: Tuple[float, float]) -> nx.MultiDiGraph:
    """
    Offline synthetic graph approximating NH48 (Delhi→Jaipur→Ajmer→Udaipur→Ahmedabad→Surat→Mumbai)
    with a couple of alternatives (via Kota/Indore/Nashik). Segment names emulate real highway labels.
    """
    G = nx.MultiDiGraph()
    # Fix endpoints to chosen cities
    cities = [
        ("Delhi", origin),
        ("Jaipur", (26.9124, 75.7873)),
        ("Ajmer", (26.4499, 74.6390)),
        ("Udaipur", (24.5854, 73.7125)),
        ("Ahmedabad", (23.0225, 72.5714)),
        ("Vadodara", (22.3072, 73.1812)),
        ("Surat", (21.1702, 72.8311)),
        ("Mumbai", dest),
        # Alternates
        ("Kota", (25.2138, 75.8648)),
        ("Indore", (22.7196, 75.8577)),
        ("Nashik", (19.9975, 73.7898)),
    ]
    # add nodes
    for i, (name, (lat, lon)) in enumerate(cities, start=1):
        G.add_node(i, y=lat, x=lon, city=name)

    # helper to add named edge
    def add(u, v, km, name, highway="motorway", maxspeed=90, flyover=False):
        G.add_edge(
            u, v, key=len(G.edges()),
            length=km*1000, name=name, highway=highway, maxspeed=maxspeed,
            bridge='yes' if flyover else None
        )

    # Main NH48 chain (coastal west corridor)
    # Index mapping:
    idx = {nm:i for i,(nm,_) in enumerate(cities, start=1)}
    add(idx["Delhi"], idx["Jaipur"],   270, "NH48 Delhi–Jaipur", "motorway", 90)
    add(idx["Jaipur"], idx["Ajmer"],   140, "NH48 Jaipur–Ajmer", "motorway", 90)
    add(idx["Ajmer"], idx["Udaipur"],  260, "NH48 Ajmer–Udaipur", "motorway", 90)
    add(idx["Udaipur"], idx["Ahmedabad"], 260, "NH48 Udaipur–Ahmedabad", "motorway", 90)
    add(idx["Ahmedabad"], idx["Vadodara"], 110, "NH48 Ahmedabad–Vadodara", "motorway", 90)
    add(idx["Vadodara"], idx["Surat"], 150, "NH48 Vadodara–Surat", "motorway", 90)
    add(idx["Surat"], idx["Mumbai"],   280, "NH48 Surat–Mumbai", "motorway", 90)

    # Alternates via Kota (NH52), Indore (NH47/NH52), Nashik (NH60)
    add(idx["Jaipur"], idx["Kota"],    250, "NH52 Jaipur–Kota", "trunk", 80)
    add(idx["Kota"], idx["Indore"],    325, "NH52/NH47 Kota–Indore", "trunk", 80)
    add(idx["Indore"], idx["Nashik"],  410, "NH52/NH60 Indore–Nashik", "trunk", 80)
    add(idx["Nashik"], idx["Mumbai"],  170, "NH60 Nashik–Mumbai", "primary", 70)

    # Bidirectional edges
    edges_to_copy = list(G.edges(keys=True, data=True))
    for u, v, k, d in edges_to_copy:
        G.add_edge(v, u, key=len(G.edges()), **d)
    return G

def load_graph(
    origin: Tuple[float, float], dest: Tuple[float, float],
    use_offline_demo: bool=False, corridor_width_km: float = 80.0
) -> Tuple[nx.MultiDiGraph, str]:
    """
    Load road graph from OSM within a buffered corridor polygon between origin and dest.
    If OSMnx is unavailable or network times out, return an offline corridor demo graph.
    Returns (G, data_source).
    """
    # Force offline if env dictates or Python 3.13 (geo stack still evolving in some setups)
    force_offline = use_offline_demo or (os.environ.get("OSM_OFFLINE", "0") == "1") or (sys.version_info.minor >= 13)
    if force_offline or not OSMNX_AVAILABLE:
        # Provide NH48 demo for typical Delhi↔Mumbai; for other pairs, build a small straight demo
        return _build_nh48_demo_graph(origin, dest), "offline-demo"

    try:
        corridor = _build_corridor_polygon(origin, dest, width_km=corridor_width_km)
        G = ox.graph_from_polygon(corridor, network_type='drive')
        G = ox.add_edge_lengths(G)
        return G, "osm-corridor"
    except Exception:
        # Any Overpass/HTTP error → offline demo
        return _build_nh48_demo_graph(origin, dest), "offline-demo-fallback"

def _edge_speed_kph(data, speed_model=None):
    speed = float(data.get('maxspeed')) if isinstance(data.get('maxspeed'), (int, float)) else None
    if speed is None:
        hw = data.get('highway'); hw = hw[0] if isinstance(hw, list) else hw
        speed = DEFAULT_SPEED_KPH_BY_HIGHWAY.get(hw, 35)
    if speed_model is not None:
        hw = data.get('highway'); hw = hw[0] if isinstance(hw, list) else hw
        is_bridge = 1 if (data.get('bridge') in ['yes', True]) else 0
        length_m = float(data.get('length', 0.0))
        X = pd.DataFrame([{ 'length_m': length_m, 'is_bridge': is_bridge, 'highway': hw }])
        X['highway_code'] = X['highway'].map({
            'motorway': 6,'trunk': 5,'primary': 4,'secondary': 3,'tertiary': 2,
            'residential': 1,'service': 0,'unclassified': 1,
            'motorway_link': 4,'primary_link': 3,'secondary_link': 2,'tertiary_link': 2
        }).fillna(1)
        try:
            speed_pred = float(speed_model.predict(X[['length_m','is_bridge','highway_code']])[0])
            speed = 0.6*speed + 0.4*max(15.0, min(100.0, speed_pred))
        except Exception:
            pass
    return speed

def _edge_weight(data, fuel_price_inr_per_l, fuel_consumption_l_per_100km, emission_factor_g_co2_per_km,
                 prefer_highways, avoid_bridges_flyovers, speed_model):
    highway = data.get('highway'); highway = highway[0] if isinstance(highway, list) else highway
    is_bridge = (data.get('bridge') in ['yes', True])
    length_m = float(data.get('length', 0.0))
    speed_kph = _edge_speed_kph(data, speed_model)
    travel_time_min = (length_m/1000.0) / max(5.0, speed_kph) * 60.0
    cost_inr = (length_m/1000.0) * (fuel_consumption_l_per_100km/100.0) * fuel_price_inr_per_l
    emissions_kg = (length_m/1000.0) * (emission_factor_g_co2_per_km/1000.0)
    base_weight = travel_time_min + 0.001*length_m + 0.0002*cost_inr + 0.0002*emissions_kg
    if prefer_highways and highway in ["motorway", "trunk"]: base_weight *= 0.97
    if avoid_bridges_flyovers and is_bridge: base_weight *= 1.10
    return max(base_weight, EPS)

def _collapse_to_digraph(G_multi: nx.MultiDiGraph, weight_func) -> nx.DiGraph:
    DG = nx.DiGraph()
    for n, attrs in G_multi.nodes(data=True): DG.add_node(n, **attrs)
    for u, v in G_multi.edges():
        edges = G_multi.get_edge_data(u, v)
        if not edges: continue
        best_data = None; best_w = None
        for k, d in edges.items():
            w = weight_func(d)
            if (best_w is None) or (w < best_w): best_w, best_data = w, d
        if best_data is not None: DG.add_edge(u, v, **best_data)
    return DG

def compute_candidate_routes(G, origin, dest, k=10,
                             fuel_price_inr_per_l=105.0,
                             fuel_consumption_l_per_100km=6.5,
                             emission_factor_g_co2_per_km=170.0,
                             prefer_highways=True, avoid_bridges_flyovers=False,
                             speed_model_path=None) -> List[Dict]:
    data_source = None
    if isinstance(G, tuple): G, data_source = G

    speed_model = None
    if speed_model_path and os.path.exists(speed_model_path):
        try:
            import joblib
            speed_model = joblib.load(speed_model_path)
        except Exception:
            speed_model = None

    def weight_data(d):
        return _edge_weight(d, fuel_price_inr_per_l, fuel_consumption_l_per_100km,
                            emission_factor_g_co2_per_km, prefer_highways,
                            avoid_bridges_flyovers, speed_model)

    DG = _collapse_to_digraph(G, weight_data) if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)) else G
    for _, _, d in DG.edges(data=True): d["route_weight"] = weight_data(d)

    # Nearest nodes (lon,lat ordering is handled internally by OSMnx)
    try:
        origin_node = ox.nearest_nodes(DG, origin[1], origin[0]) if OSMNX_AVAILABLE else list(DG.nodes())[0]
        dest_node   = ox.nearest_nodes(DG, dest[1], dest[0])     if OSMNX_AVAILABLE else list(DG.nodes())[-1]
    except Exception:
        origin_node = list(DG.nodes())[0]; dest_node = list(DG.nodes())[-1]

    try:
        paths_gen = nx.shortest_simple_paths(DG, origin_node, dest_node, weight='route_weight')
    except nx.NetworkXNoPath:
        return []

    routes = []
    for i, path in enumerate(paths_gen):
        if i >= k: break
        segments = []
        total_length_m = total_time_min = total_cost_inr = total_emissions_kg = 0.0
        num_flyovers = 0; highway_len_m = 0.0

        for u, v in zip(path[:-1], path[1:]):
            data = DG.get_edge_data(u, v) if DG.has_edge(u, v) else {"length": 0.0, "name": None, "highway": None, "maxspeed": None}
            name = data.get('name') or 'Unnamed road'
            highway = data.get('highway'); highway = highway[0] if isinstance(highway, list) else highway
            length_m = float(data.get('length', 0.0))
            maxspeed_kph = _edge_speed_kph(data, speed_model)
            travel_time_min = (length_m/1000.0) / max(5.0, maxspeed_kph) * 60.0
            cost_inr = (length_m/1000.0) * (fuel_consumption_l_per_100km/100.0) * fuel_price_inr_per_l
            emissions_kg = (length_m/1000.0) * (emission_factor_g_co2_per_km/1000.0)
            is_flyover = (data.get('bridge') in ['yes', True]) or ('flyover' in str(name).lower())

            u_lat, u_lon = float(DG.nodes[u].get('y')), float(DG.nodes[u].get('x'))
            v_lat, v_lon = float(DG.nodes[v].get('y')), float(DG.nodes[v].get('x'))
            segments.append({
                "u": u, "v": v, "u_lat": u_lat, "u_lon": u_lon, "v_lat": v_lat, "v_lon": v_lon,
                "name": name, "highway": highway, "length_m": length_m, "maxspeed_kph": maxspeed_kph,
                "travel_time_min": travel_time_min, "cost_inr": cost_inr, "emissions_kg": emissions_kg,
                "is_flyover": is_flyover,
            })

            total_length_m += length_m; total_time_min += travel_time_min
            total_cost_inr += cost_inr; total_emissions_kg += emissions_kg
            if is_flyover: num_flyovers += 1
            if highway in ["motorway", "trunk"]: highway_len_m += length_m

        routes.append({
            "route_id": i+1, "path": path, "segments": segments,
            "total_distance_km": total_length_m/1000.0, "total_time_min": total_time_min,
            "total_cost_inr": total_cost_inr, "total_emissions_kg": total_emissions_kg,
            "num_segments": len(segments), "has_flyover": num_flyovers > 0,
            "highway_share_pct": (highway_len_m/total_length_m*100.0) if total_length_m>0 else 0.0,
            "data_source": data_source or ("osm" if OSMNX_AVAILABLE else "demo"),
        })
    return routes

def summarize_routes(routes: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(routes)
    if df.empty: return df
    metrics = df[["total_distance_km","total_time_min","total_cost_inr","total_emissions_kg"]].values
    ranks = _pareto_ranks(metrics); df["dominance_rank"] = ranks; df["optimized"] = ["candidate"]*len(df)
    return df

def recommend_route(df_routes: pd.DataFrame, objective: str):
    if df_routes is None or len(df_routes)==0: return 0, "optimized:none (no routes)"
    if objective == 'pareto':
        df_nd = df_routes[df_routes['dominance_rank']==0].copy()
        if df_nd.empty: df_nd = df_routes.copy()
        idx_rel, _ = _best_balanced_index(df_nd)
        idx = df_nd.index[idx_rel]
        tag = f"optimized:pareto-balanced (nondominated set size={len(df_nd)})"
    elif objective in ['min_distance','min_time','min_cost','min_emissions']:
        m = {'min_distance':'total_distance_km','min_time':'total_time_min','min_cost':'total_cost_inr','min_emissions':'total_emissions_kg'}
        idx = int(df_routes[m[objective]].idxmin()); tag = f"optimized:{objective}"
    else:
        idx, _ = _best_balanced_index(df_routes); tag = "optimized:balanced"
    df_routes.loc[:, 'optimized'] = 'candidate'; df_routes.loc[idx, 'optimized'] = 'recommended'
    return idx, tag

def _best_balanced_index(df: pd.DataFrame):
    cols = ["total_distance_km","total_time_min","total_cost_inr","total_emissions_kg"]
    X = df[cols].values.astype(float); mins = X.min(axis=0); maxs = X.max(axis=0)
    denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
    scores = ((X - mins) / denom).sum(axis=1)
    idx_rel = int(np.argmin(scores)); tag = f"optimized:balanced(score={scores[idx_rel]:.3f})"
    return idx_rel, tag

def _pareto_ranks(values: np.ndarray) -> List[int]:
    n = values.shape[0]; ranks = np.zeros(n, dtype=int)
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j: continue
            if np.all(values[j] <= values[i]) and np.any(values[j] < values[i]):
                dominated = True; break
        ranks[i] = 0 if not dominated else 1
    return ranks.tolist()
