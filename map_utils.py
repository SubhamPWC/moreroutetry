
# map_utils.py
import folium

def make_map(routes, recommended_idx, origin, dest):
    center_lat = (origin[0] + dest[0]) / 2.0
    center_lon = (origin[1] + dest[1]) / 2.0
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=6, control_scale=True, tiles='OpenStreetMap')

    folium.Marker([origin[0], origin[1]], popup="Origin", tooltip="Origin", icon=folium.Icon(color='green')).add_to(fmap)
    folium.Marker([dest[0], dest[1]], popup="Destination", tooltip="Destination", icon=folium.Icon(color='red')).add_to(fmap)

    for i, route in enumerate(routes):
        color = 'red' if i == recommended_idx else 'blue'
        weight = 6 if i == recommended_idx else 3
        coords = []
        for seg in route.get('segments', []):
            coords += [[seg['u_lat'], seg['u_lon']], [seg['v_lat'], seg['v_lon']]]
        dedup = []
        for c in coords:
            if not dedup or (dedup[-1][0] != c[0] or dedup[-1][1] != c[1]):
                dedup.append(c)
        if len(dedup) >= 2:
            folium.PolyLine(dedup, color=color, weight=weight, opacity=0.8,
                            tooltip=f"Route {route['route_id']}: {route['total_distance_km']:.1f} km, {route['total_time_min']:.1f} min").add_to(fmap)
    return fmap
