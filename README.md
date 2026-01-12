
# Road Route Optimization (India) — Streamlit App

**Features**
- Not using Google Maps. Uses **OpenStreetMap** via **OSMnx** (when online) or a small offline demo graph.
- Computes up to *k* alternative **road routes** between selected lat/lon points in India.
- **Multi-criteria optimization**: distance, time, cost, emissions. Choose primary objective or view Pareto-optimal set.
- **ML speed model (optional)** with scikit-learn to adjust segment speeds.
- Displays a **table** of all route alternatives with a **recommended** route tagged, plus a **map** highlighting the optimized path.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scriptsctivate
pip install -r requirements.txt
streamlit run app.py
```

## Cities included (From/To)
Kolkata, Bhubaneswar, Ranchi, Patna, Delhi, Mumbai, Bengaluru, Hyderabad, Chennai, Pune, Ahmedabad, Jaipur, Surat, Lucknow, Kanpur, Nagpur, Indore, Bhopal, Coimbatore, Kochi, Thiruvananthapuram, Visakhapatnam, Vijayawada, Guwahati, Siliguri, Durgapur, Asansol, Jamshedpur, Dhanbad, Cuttack, Puri, Gaya, Varanasi.

## Notes & Limitations
- For real roads, names, small streets, highways, and flyovers, disable **Offline Demo** and ensure internet to fetch OSM data.
- The offline demo graph is **synthetic** and only for illustrating the UI and optimization flow.
- The map draws polylines based on per-segment node coordinates. You can enhance it to follow exact edge geometry if available.
- ML model: drop a trained model at `models/speed_model.pkl` (e.g., `RandomForestRegressor`) that predicts segment speed (kph) from features.

## Repository structure
```
app.py               # Streamlit UI with expanded city list
optimization.py      # Graph loading, k-route generation, metrics, recommendation (MultiDiGraph -> DiGraph collapse; non-negative weights)
map_utils.py         # Folium map rendering
requirements.txt     # Python dependencies
models/              # (optional) place speed_model.pkl here
```

## Configuration
- Vehicle & environment parameters are adjustable in the sidebar (fuel price, consumption, emissions).
- Preferences: *Prefer highways*, *Avoid flyovers* influence routing weights.

## Data & Attribution
- When online, data is fetched from **OpenStreetMap** via `osmnx`.
- Please respect OpenStreetMap usage policy.
