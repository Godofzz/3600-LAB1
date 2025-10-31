from __future__ import annotations
from pathlib import Path
from typing import List
import math
import networkx as nx

GRAPH_CACHE = Path(__file__).resolve().parent / "kl_drive.graphml"

_node_ids: List = []
_node_lat: List[float] = []
_node_lon: List[float] = []

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    from math import radians, sin, cos, asin, sqrt
    dphi = radians(lat2 - lat1)
    dlmb = radians(lon2 - lon1)
    p1, p2 = radians(lat1), radians(lat2)
    a = sin(dphi/2)**2 + cos(p1)*cos(p2)*sin(dlmb/2)**2
    return 2*R*asin(sqrt(a))

def _prep_node_arrays(G: nx.MultiDiGraph) -> None:
    """Cache arrays for fast nearest-node lookup."""
    global _node_ids, _node_lat, _node_lon
    _node_ids, _node_lat, _node_lon = [], [], []
    for n, data in G.nodes(data=True):
        _node_ids.append(n)
        _node_lat.append(float(data["y"]))
        _node_lon.append(float(data["x"]))

def load_graph() -> nx.MultiDiGraph:
    if not GRAPH_CACHE.exists():
        raise FileNotFoundError(f"Missing {GRAPH_CACHE.name}. Provide a pre-cached GraphML for KL roads.")
    G = nx.read_graphml(GRAPH_CACHE)
    if not isinstance(G, nx.MultiDiGraph):
        G = nx.MultiDiGraph(G)

    # Force-cast node coords to float
    for _, d in G.nodes(data=True):
        if "y" in d: d["y"] = float(d["y"])
        if "x" in d: d["x"] = float(d["x"])

    # Make sure edges have numeric 'length'
    for u, v, k, d in G.edges(keys=True, data=True):
        if "length" in d:
            try:
                d["length"] = float(d["length"])
            except Exception:
                y1, x1 = float(G.nodes[u]["y"]), float(G.nodes[u]["x"])
                y2, x2 = float(G.nodes[v]["y"]), float(G.nodes[v]["x"])
                d["length"] = _haversine_m(y1, x1, y2, x2)
        else:
            y1, x1 = float(G.nodes[u]["y"]), float(G.nodes[u]["x"])
            y2, x2 = float(G.nodes[v]["y"]), float(G.nodes[v]["x"])
            d["length"] = _haversine_m(y1, x1, y2, x2)

    # 🔑 Keep only the largest weakly-connected component to avoid islands
    if G.number_of_nodes() > 0:
        wccs = list(nx.weakly_connected_components(G))
        if len(wccs) > 1:
            largest = max(wccs, key=len)
            G = G.subgraph(largest).copy()

    _prep_node_arrays(G)
    return G

def nearest_node(G: nx.MultiDiGraph, lat: float, lon: float):
    """Pure-Python nearest node using haversine (no sklearn/osmnx)."""
    best_idx, best_d = 0, float("inf")
    for i in range(len(_node_ids)):
        # arrays are floats already
        d = _haversine_m(lat, lon, _node_lat[i], _node_lon[i])
        if d < best_d:
            best_d, best_idx = d, i
    return _node_ids[best_idx]
