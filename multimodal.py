
# multimodal.py
import pandas as pd
import numpy as np

EMISSION_FACTORS = {"road": 0.170, "train": 0.041, "air": 0.115}

def load_flights(origin_city: str, dest_city: str) -> pd.DataFrame:
    try:
        df = pd.read_csv("data/flights.csv")
        return df[(df["origin_city"] == origin_city) & (df["dest_city"] == dest_city)].copy()
    except Exception:
        return pd.DataFrame()

def load_trains(origin_city: str, dest_city: str) -> pd.DataFrame:
    try:
        df = pd.read_csv("data/trains.csv")
        return df[(df["origin_city"] == origin_city) & (df["dest_city"] == dest_city)].copy()
    except Exception:
        return pd.DataFrame()

def summarize_mode(df_routes: pd.DataFrame, mode: str, distance_km: float) -> pd.DataFrame:
    if df_routes is None or len(df_routes) == 0: return pd.DataFrame()
    df = df_routes.copy()
    if mode == "air":
        df["total_time_min"] = (df["duration_h"] * 60).astype(float)
        df["total_cost_inr"] = df["typical_price_inr"].astype(float)
        df["emissions_kg"] = float(distance_km) * EMISSION_FACTORS["air"]
        df["label"] = df["airline"]
    elif mode == "train":
        df["total_time_min"] = (df["duration_h"] * 60).astype(float)
        df["total_cost_inr"] = df["price_3A_inr"].fillna(df["price_2A_inr"]).astype(float)
        df["emissions_kg"] = float(distance_km) * EMISSION_FACTORS["train"]
        df["label"] = df["train_name"]
    else:
        raise ValueError("Unsupported mode")
    cols = ["label","total_time_min","total_cost_inr","emissions_kg"]
    return df[cols + ([c for c in df.columns if c not in cols])]

def recommend(df: pd.DataFrame, objective: str) -> int:
    if df is None or len(df)==0: return -1
    if objective=='min_time': return int(df['total_time_min'].idxmin())
    if objective=='min_cost': return int(df['total_cost_inr'].idxmin())
    if objective=='min_emissions': return int(df['emissions_kg'].idxmin())
    X = df[['total_time_min','total_cost_inr','emissions_kg']].values.astype(float)
    mins = X.min(axis=0); maxs = X.max(axis=0); denom = (maxs-mins); denom[denom==0] = 1.0
    score = ((X - mins) / denom).sum(axis=1)
    return int(score.argmin())
