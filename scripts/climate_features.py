from __future__ import annotations
import os
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from pyproj import Transformer

@dataclass
class ClimateConfig:
    weather_csv: str                        # S-DoT measurements
    stations_csv: str                       # station data
    var_cols: List[str]                     # features to use
    start_time_dt: pd.Timestamp             # align to movement start   
    bin_seconds: int                        # bin size in seconds
    max_gap_minutes: int = 90               # temporal fill window
    node_xy_crs: str = "EPSG:5174"          # CRS for node_xy
    station_ll_crs: str = "EPSG:5174"       # CRS of station lon/lat columns
    cache_dir: str = "./climate_cache"      # directory to store cached node-climate arrays
    # column names in station file
    station_id_col: str = "SerialNum"
    station_lon_col: str = "Xlon"
    station_lat_col: str = "Ylat"
    # weather file columns
    weather_id_col: str = "serial"
    weather_time_col: str = "datetime"

    # ------------------------- preparation -------------------------

def _read_weather(cfg: ClimateConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.weather_csv)
    if cfg.weather_id_col not in df.columns and "SerialNum" in df.columns:
        df = df.rename(columns={"SerialNum": cfg.weather_id_col})
    df[cfg.weather_time_col] = pd.to_datetime(df[cfg.weather_time_col], errors="coerce", utc=True)\
                                   .dt.tz_convert(cfg.start_time_dt.tz)
    df = df.dropna(subset=[cfg.weather_id_col, cfg.weather_time_col]).sort_values(
        [cfg.weather_id_col, cfg.weather_time_col]
    ).reset_index(drop=True)

    cols_present = [c for c in cfg.var_cols if c in df.columns]
    if not cols_present:
        raise ValueError("None of cfg.var_cols found in weather CSV.")
    return df[[cfg.weather_id_col, cfg.weather_time_col] + cols_present].copy()


def _read_stations(cfg: ClimateConfig) -> pd.DataFrame:
    s = pd.read_csv(cfg.stations_csv, encoding="cp949")

    if cfg.station_id_col not in s.columns:
        raise ValueError(f"Station id column '{cfg.station_id_col}' not in stations CSV.")

    transformer = Transformer.from_crs(cfg.station_ll_crs, cfg.node_xy_crs, always_xy=True)
    x, y = transformer.transform(s[cfg.station_lon_col].values, s[cfg.station_lat_col].values)
    out = pd.DataFrame({
        "serial": s[cfg.station_id_col].astype(str),  # use str ids
        "sx": x,
        "sy": y,
    })
    return out


def _hash_key(cfg: ClimateConfig, nodes_xy: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(str(cfg.start_time_dt.value).encode())
    h.update(str(cfg.bin_seconds).encode())
    h.update(",".join(sorted(cfg.var_cols)).encode())
    h.update(str(nodes_xy.shape).encode())
    h.update(str(int(nodes_xy.sum()*1e-9)).encode())
    return h.hexdigest()[:12]




# ------------------------- temporal gap fill -------------------------

def _fill_short_gaps_linear(df: pd.DataFrame, cfg: ClimateConfig) -> pd.DataFrame:
    """
    For each station & feature:
      - if current row has a value, keep it
      - else: if prev & next valid within max_gap_minutes, linear interpolate in time
              elif only one side within window, use that side (nearest in time)
              else, remain NaN
    """
    df = df.sort_values([cfg.weather_id_col, cfg.weather_time_col]).copy()
    t = df[cfg.weather_time_col]

    for col in cfg.var_cols:
        if col not in df.columns:
            continue
        vals = df[col]
        g = df.groupby(cfg.weather_id_col)
        prev_val  = g[col].ffill()
        next_val  = g[col].bfill()
        prev_time = t.where(vals.notna()).groupby(df[cfg.weather_id_col]).ffill()
        next_time = t.where(vals.notna()).groupby(df[cfg.weather_id_col]).bfill()

        dprev = (t - prev_time).dt.total_seconds() / 60.0
        dnext = (next_time - t).dt.total_seconds() / 60.0

        has_prev = prev_val.notna() & dprev.notna() & (dprev <= cfg.max_gap_minutes)
        has_next = next_val.notna() & dnext.notna() & (dnext <= cfg.max_gap_minutes)

        filled = vals.copy()

        need = vals.isna()
        both = need & has_prev & has_next
        if both.any():
            tot = dprev[both] + dnext[both]
            w_prev = dnext[both] / tot
            w_next = dprev[both] / tot
            filled.loc[both] = w_prev * prev_val[both] + w_next * next_val[both]

        only_prev = need & has_prev & ~has_next
        if only_prev.any():
            filled.loc[only_prev] = prev_val[only_prev]

        only_next = need & has_next & ~has_prev
        if only_next.any():
            filled.loc[only_next] = next_val[only_next]

        df[col + "_filled"] = filled

    return df



# ------------------------- binning -------------------------

def _assign_bins(df: pd.DataFrame, cfg: ClimateConfig) -> pd.DataFrame:
    tz_dt = cfg.start_time_dt if cfg.start_time_dt.tzinfo else cfg.start_time_dt.tz_localize("Asia/Seoul")
    t0 = int(tz_dt.timestamp())
    secs = (df[cfg.weather_time_col].view("int64") // 10**9) - t0
    df = df.copy()
    df["time_bin"] = (secs // cfg.bin_seconds).astype(int)
    return df


def _aggregate_bins(df: pd.DataFrame, cfg: ClimateConfig) -> pd.DataFrame:
    filled_cols = [c + "_filled" for c in cfg.var_cols if (c + "_filled") in df.columns]
    if not filled_cols:
        raise ValueError("No *_filled columns found. Did you run _fill_short_gaps_linear first?")
    grp = (df.groupby([cfg.weather_id_col, "time_bin"], as_index=False)[filled_cols]
             .mean(numeric_only=True))
    grp = grp.rename(columns={c: c.replace("_filled", "") for c in filled_cols})
    return grp  # columns: serial, time_bin, <features>




# ------------------------- interpolation per bin -------------------------

def _interpolate_bin(
    stations_xy: pd.DataFrame,          # ["serial","sx","sy"]
    bin_slice: pd.DataFrame,            # ["serial","time_bin", features...], single bin
    nodes_xy: np.ndarray,               # shape [N, 2]
    feat_cols: List[str],
) -> np.ndarray:
    """
    Returns matrix [N_nodes, N_features] for this bin.
    """
    # join station positions
    df = bin_slice.merge(stations_xy, on="serial", how="inner")
    out = np.full((nodes_xy.shape[0], len(feat_cols)), np.nan, dtype=float)

    if df.empty:
        return out

    P = df[["sx", "sy"]].to_numpy()

    for j, col in enumerate(feat_cols):
        z = df[col].to_numpy()
        valid = np.isfinite(z)
        if valid.sum() == 0:
            continue
        P_valid = P[valid]
        z_valid = z[valid]

        if P_valid.shape[0] >= 3:
            lin = LinearNDInterpolator(P_valid, z_valid, rescale=False)  # builds its own Delaunay
            vals = lin(nodes_xy)

            if np.isnan(vals).any():
                near = NearestNDInterpolator(P_valid, z_valid)
                nanmask = np.isnan(vals)
                vals[nanmask] = near(nodes_xy[nanmask])
        else:
            near = NearestNDInterpolator(P_valid, z_valid)
            vals = near(nodes_xy)

        out[:, j] = vals

    return out



# ------------------------- final function to call -------------------------

def compute_node_climate_by_bin(
    cfg: ClimateConfig,
    nodes_xy: np.ndarray,           
    use_cache: bool = True,
    force_recompute: bool = False,
) -> Tuple[Dict[int, np.ndarray], List[str], str]:
    """
    Returns:
      - climate_by_bin: dict {time_bin: ndarray [N_nodes, N_features]}
      - feature_names: same order as columns
      - cache_path: file used/created (parquet)
    """
    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
    key = _hash_key(cfg, nodes_xy)
    cache_path = os.path.join(cfg.cache_dir, f"node_climate_bins_{key}.parquet")

    if use_cache and (not force_recompute) and os.path.exists(cache_path):
        tbl = pd.read_parquet(cache_path)
        # reconstruct dict
        bins = sorted(tbl["time_bin"].unique())
        feat_cols = [c for c in tbl.columns if c not in ("time_bin", "node_id")]
        n_nodes = tbl["node_id"].max() + 1
        climate_by_bin = {}
        for tb in bins:
            sub = tbl[tbl["time_bin"] == tb].sort_values("node_id")
            M = sub[feat_cols].to_numpy()
            # ensure shape [n_nodes, n_feats]
            if M.shape[0] != n_nodes:
                # sparse saved? fall back to re-compute
                break
            climate_by_bin[tb] = M
        else:
            return climate_by_bin, feat_cols, cache_path
        # fallthrough to recompute if any inconsistency

    # load + prep
    weather = _read_weather(cfg)
    stations = _read_stations(cfg)
    # unify id types
    weather[cfg.weather_id_col] = weather[cfg.weather_id_col].astype(str)

    # temporal fill
    weather_filled = _fill_short_gaps_linear(weather, cfg)

    # binning & aggregation
    weather_filled = _assign_bins(weather_filled, cfg)
    agg = _aggregate_bins(weather_filled, cfg)  # serial, time_bin, features

    # build per-bin node climate
    bins = sorted(agg["time_bin"].unique())
    feat_cols = [c for c in cfg.var_cols if c in agg.columns]
    climate_by_bin: Dict[int, np.ndarray] = {}

    for tb in bins:
        slice_tb = agg[agg["time_bin"] == tb][[cfg.weather_id_col, "time_bin"] + feat_cols]\
                        .rename(columns={cfg.weather_id_col: "serial"})
        climate_by_bin[tb] = _interpolate_bin(stations, slice_tb, nodes_xy, feat_cols)

    # save cache as long table: (time_bin, node_id, features...)
    rows = []
    for tb, M in climate_by_bin.items():
        n_nodes, n_feats = M.shape
        df_row = pd.DataFrame({
            "time_bin": np.full(n_nodes, tb, dtype=int),
            "node_id": np.arange(n_nodes, dtype=int),
        })
        for j, col in enumerate(feat_cols):
            df_row[col] = M[:, j]
        rows.append(df_row)
    cache_tbl = pd.concat(rows, ignore_index=True)
    # write with minimal metadata
    cache_tbl.attrs["config"] = json.dumps({
        "bin_seconds": cfg.bin_seconds,
        "start_time": str(cfg.start_time_dt),
        "var_cols": feat_cols,
        "node_xy_crs": cfg.node_xy_crs,
        "hash_key": key,
    })
    cache_tbl.to_parquet(cache_path, index=False)

    return climate_by_bin, feat_cols, cache_path